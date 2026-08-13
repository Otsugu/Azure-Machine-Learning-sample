"""Azure Machine Learning 上で強化学習を実行する学習スクリプト。

このスクリプト 1 本でベースライン実験と改善実験の両方をまかなう。
**変更するのは引数だけ**にすることで、「何を変えたから結果が変わったのか」を
追跡可能にしている。

--------------------------------------------------------------------------
記録の方針（「実験管理と MLflow の基本」に対応）
--------------------------------------------------------------------------
* Azure ML のジョブとして実行されると MLflow の run は自動的に開始されるため
  ``mlflow.start_run()`` は呼ばない。
    出典: https://learn.microsoft.com/azure/machine-learning/how-to-log-view-metrics?view=azureml-api-2
* メトリック名に ``/`` を使わず ``eval_success_rate`` のように平坦にする。
  → Sweep Job の ``primary_metric`` にそのまま指定できるようにするため。
    出典: https://learn.microsoft.com/azure/machine-learning/how-to-tune-hyperparameters?view=azureml-api-2
* **成功率と平均報酬を必ず両方記録し、評価動画も残す。**
  報酬ハッキング（報酬は上がるが望ましくない動作）を検出するために必須。
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import mlflow
import numpy as np

from evaluate import evaluate_policy_with_success, record_video
from rl_env_factory import REWARD_MODES, describe_env, make_env, make_env_fn, resolve_env_id

ALGOS = ("sac", "td3", "ppo")
#: HER が使えるのはオフポリシー手法のみ（リプレイバッファの仕組みであるため）
OFF_POLICY_ALGOS = ("sac", "td3")


# ======================================================================
# 引数
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Azure ML 上での強化学習ジョブ")

    # --- 課題設定 ---
    p.add_argument("--env-id", type=str, default="PandaPickAndPlace-v3")
    p.add_argument("--reward-mode", type=str, default="sparse", choices=list(REWARD_MODES))
    p.add_argument("--shaping-weight", type=float, default=0.1,
                   help="報酬シェーピングの重み（reward-mode が sparse_shaped / sparse_time_penalty のとき有効）")

    # --- アルゴリズム ---
    p.add_argument("--algo", type=str, default="sac", choices=list(ALGOS))
    p.add_argument("--use-her", type=int, default=1, choices=[0, 1],
                   help="1 なら HER を使う。Sweep Job から渡しやすいよう int にしている")
    p.add_argument("--n-sampled-goal", type=int, default=4)
    p.add_argument("--goal-selection-strategy", type=str, default="future",
                   choices=["future", "final", "episode"])

    # --- ハイパーパラメーター（演習2 の主役） ---
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=1_000)

    # --- 実験条件 ---
    p.add_argument("--total-timesteps", type=int, default=50_000,
                   help="短時間で確実に終わる値を既定にしている")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1)

    # --- 評価 ---
    p.add_argument("--eval-freq", type=int, default=5_000, help="何タイムステップごとに評価するか")
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--final-eval-episodes", type=int, default=100,
                   help="最終評価のエピソード数。多いほど成功率の推定が安定する")
    p.add_argument("--record-video", type=int, default=1, choices=[0, 1])
    p.add_argument("--video-episodes", type=int, default=3)

    # --- 出力 ---
    p.add_argument("--output-dir", type=str, default="./outputs",
                   help="Azure ML では ./outputs 配下が自動的にジョブへ収集される")
    p.add_argument("--reward-fn-version", type=str, default="v1",
                   help="報酬関数のバージョン。**変更したら必ず上げること**（比較の追跡に使う）")

    return p.parse_args()


# ======================================================================
# MLflow へのバージョン記録（再現性の要）
# ======================================================================
def log_environment_versions() -> None:
    import gymnasium as gym
    import panda_gym
    import stable_baselines3 as sb3
    import torch

    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("numpy_version", np.__version__)
    mlflow.log_param("gymnasium_version", gym.__version__)
    mlflow.log_param("panda_gym_version", panda_gym.__version__)
    mlflow.log_param("sb3_version", sb3.__version__)
    mlflow.log_param("torch_version", torch.__version__)

    # panda-gym は setup.py で numpy<2 を要求している
    if not np.__version__.startswith("1."):
        raise RuntimeError(
            f"panda-gym は numpy<2 を要求します。現在 numpy=={np.__version__}。"
            " conda.yaml の 'numpy<2' 指定を確認してください。"
        )


def save_pip_freeze(output_dir: str) -> str:
    """実際に解決されたパッケージ一覧を残す（「どの版で回したか」の唯一の証跡）。"""
    path = os.path.join(output_dir, "pip_freeze.txt")
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120
        ).stdout
    except Exception as exc:  # noqa: BLE001
        freeze = f"# pip freeze に失敗しました: {exc}\n"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(freeze)
    return path


# ======================================================================
# 学習中に定期評価して MLflow へ記録するコールバック
# ======================================================================
def build_eval_callback(eval_env, args):
    from stable_baselines3.common.callbacks import BaseCallback

    class MlflowEvalCallback(BaseCallback):
        """一定タイムステップごとに評価し、MLflow へ記録する。

        **学習中の報酬（rollout）と、評価時の成功率（eval）を分けて記録する。**
        この 2 つが食い違うときが、報酬ハッキングを疑うべき瞬間。
        """

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.best_success_rate = -1.0

        def _on_step(self) -> bool:
            if args.eval_freq <= 0 or self.n_calls % args.eval_freq != 0:
                return True

            step = int(self.num_timesteps)

            # --- 学習中の統計（Monitor が集めたもの） ---
            buf = getattr(self.model, "ep_info_buffer", None)
            if buf:
                mlflow.log_metric("rollout_ep_rew_mean",
                                  float(np.mean([e["r"] for e in buf])), step=step)
                mlflow.log_metric("rollout_ep_len_mean",
                                  float(np.mean([e["l"] for e in buf])), step=step)
                if "is_success" in buf[0]:
                    mlflow.log_metric("rollout_success_rate",
                                      float(np.mean([float(e["is_success"]) for e in buf])),
                                      step=step)

            # --- 評価 ---
            metrics = evaluate_policy_with_success(
                self.model, eval_env,
                n_episodes=args.n_eval_episodes,
                deterministic=True,
                seed=10_000 + args.seed,
            )
            for key, value in metrics.items():
                mlflow.log_metric(f"eval_{key}", value, step=step)

            self.best_success_rate = max(self.best_success_rate, metrics["success_rate"])
            mlflow.log_metric("eval_best_success_rate", self.best_success_rate, step=step)

            print(
                f"[eval] steps={step:>8}  success_rate={metrics['success_rate']:.3f}"
                f"  mean_reward={metrics['mean_reward']:.2f}"
                f"  mean_len={metrics['mean_episode_length']:.1f}"
            )
            return True

    return MlflowEvalCallback()


# ======================================================================
# モデルの構築
# ======================================================================
def build_model(args, train_env):
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3 import HerReplayBuffer

    use_her = bool(args.use_her) and args.algo in OFF_POLICY_ALGOS
    if bool(args.use_her) and not use_her:
        print(f"[WARN] {args.algo} はオンポリシー手法のため HER を使えません。HER なしで実行します。")

    # 観測が辞書型（observation / achieved_goal / desired_goal）なので MultiInputPolicy を使う
    policy = "MultiInputPolicy"

    common = dict(
        policy=policy,
        env=train_env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        verbose=1,
    )

    if args.algo == "ppo":
        model = PPO(batch_size=args.batch_size, **common)
    else:
        off_policy = dict(
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
        )
        if use_her:
            off_policy["replay_buffer_class"] = HerReplayBuffer
            off_policy["replay_buffer_kwargs"] = dict(
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy=args.goal_selection_strategy,
            )
        model = (SAC if args.algo == "sac" else TD3)(**common, **off_policy)

    return model, use_her


# ======================================================================
# メイン
# ======================================================================
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(args.seed)

    # ---------- 1) 実験条件をすべて記録する ----------
    log_environment_versions()
    for key, value in sorted(vars(args).items()):
        mlflow.log_param(key, value)
    mlflow.log_param("actual_env_id", resolve_env_id(args.env_id, args.reward_mode))
    mlflow.set_tag("experiment_kind", "reinforcement-learning")
    mlflow.set_tag("reward_fn_version", args.reward_fn_version)

    # ---------- 2) 環境を作る ----------
    #   学習用: Monitor でエピソード統計を集める（info_keywords で成功判定も拾う）
    #   ※ make_env_fn には seed を渡さない。make_vec_env が各環境に seed + rank を
    #      設定するため、二重に渡すと意図が不明瞭になる。
    train_env = make_vec_env(
        make_env_fn(args.env_id, args.reward_mode, args.shaping_weight, render=False, seed=None),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_kwargs={"info_keywords": ("is_success",)},
    )

    # ---------------------------------------------------------------
    #   評価用: **常にベースラインの sparse 報酬で評価する。**
    #
    #   これは本ハンズオンで最も重要な設計判断のひとつ。
    #   学習時の報酬設計（sparse / dense / shaped ...）をそのまま評価に使うと、
    #   条件ごとに「ものさし」が変わるため **eval_mean_reward を条件間で比較できなくなる**。
    #   「同一の評価指標で比較する」を守るため、評価は常に sparse に固定する。
    #   （成功率は info["is_success"] 由来なので報酬設計に依存せずもともと比較可能。）
    # ---------------------------------------------------------------
    EVAL_REWARD_MODE = "sparse"
    mlflow.log_param("eval_reward_mode", EVAL_REWARD_MODE)
    eval_env = make_env(args.env_id, EVAL_REWARD_MODE, shaping_weight=0.0,
                        render=False, seed=1_000 + args.seed)

    spec = describe_env(eval_env)
    for key, value in spec.items():
        if value is not None:
            mlflow.log_param(f"env_{key}", value)
    print("環境仕様:", json.dumps(spec, ensure_ascii=False))

    # ---------- 3) 学習 ----------
    model, use_her = build_model(args, train_env)
    mlflow.log_param("her_effective", int(use_her))

    started = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=build_eval_callback(eval_env, args),
        progress_bar=False,
    )
    train_seconds = time.time() - started
    mlflow.log_metric("train_seconds", train_seconds)
    mlflow.log_metric("train_minutes", train_seconds / 60.0)
    print(f"学習時間: {train_seconds / 60.0:.1f} 分")

    # ---------- 4) 最終評価 ----------
    final = evaluate_policy_with_success(
        model, eval_env,
        n_episodes=args.final_eval_episodes,
        deterministic=True,
        seed=20_000 + args.seed,
    )
    for key, value in final.items():
        mlflow.log_metric(f"final_{key}", value)
    print("最終評価:", json.dumps(final, ensure_ascii=False, indent=2))

    # ---------- 5) 成果物 ----------
    model_path = os.path.join(args.output_dir, "model.zip")
    model.save(model_path)
    mlflow.log_artifact(model_path)
    #   ⚠ HER を使ったモデルは load 時に env が必須:  SAC.load(path, env=env)
    #      env.compute_reward() にアクセスできないと復元できないため。

    if args.record_video:
        #   動画も評価と同じ条件（sparse）で収録する
        video_env = make_env(args.env_id, EVAL_REWARD_MODE, shaping_weight=0.0,
                             render=True, seed=30_000 + args.seed)
        video_path = os.path.join(args.output_dir, "eval_video.mp4")
        ok = record_video(model, video_env, video_path,
                          n_episodes=args.video_episodes, seed=30_000 + args.seed)
        video_env.close()
        mlflow.log_metric("video_recorded", float(ok))
        if ok:
            mlflow.log_artifact(video_path)

    mlflow.log_artifact(save_pip_freeze(args.output_dir))

    summary = {
        "params": vars(args),
        "env_spec": spec,
        "her_effective": use_her,
        "train_seconds": train_seconds,
        "final_metrics": final,
    }
    summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2, default=str)
    mlflow.log_artifact(summary_path)

    train_env.close()
    eval_env.close()
    print("TRAINING DONE")


if __name__ == "__main__":
    main()
