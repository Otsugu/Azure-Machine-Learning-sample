"""専門家 PPO を学習し、そのデモンストレーションと評価基準を保存する。

出力（--output-dir 配下）:
  demos.npz         : 専門家の軌跡。BC / DAgger / GAIL の入力
  expert_policy.zip : 専門家方策。DAgger が学習中に問い合わせるため必要
  scores.json       : 正規化リターンの計算に使う基準値
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mlflow
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from il_common import DEFAULT_ENV_ID, evaluate, log_pip_freeze, make_env
from imitation.data import rollout, serialize

N_ENVS = 8
N_EVAL_EPISODES = 20


def evaluate_uniform_random(env_id: str, n_episodes: int, seed: int) -> tuple[float, float]:
    """一様ランダム行動の平均リターンを返す。

    正規化リターンの基準に使う。未学習の PPO は初期化次第で高い値を出しばらつきも
    大きいため、基準に適さない（実測値は ../調査レポート_模倣学習.md の 4.4.2）。
    """
    env = gym.make(env_id)
    returns = []
    for episode in range(n_episodes):
        env.reset(seed=seed + episode)
        total = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--expert-timesteps", type=int, default=100_000)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # output-dir は実行ごとに変わる一時パスなので、実験比較のパラメーターには含めない
    mlflow.log_params(
        {k: v for k, v in vars(args).items() if k != "output_dir"}
    )

    venv = make_env(args.env_id, N_ENVS, args.seed)

    random_mean, random_std = evaluate_uniform_random(
        args.env_id, N_EVAL_EPISODES, args.seed
    )
    print(f"uniform random  mean={random_mean:8.2f} std={random_std:7.2f}")

    expert = PPO(policy=MlpPolicy, env=venv, seed=args.seed, verbose=0)
    expert.learn(args.expert_timesteps)
    expert_mean, expert_std = evaluate(expert, venv, N_EVAL_EPISODES)
    print(f"expert PPO      mean={expert_mean:8.2f} std={expert_std:7.2f}")

    # 専門家がランダム以下だと (expert_mean - random_mean) が 0 以下になり、
    # 後段の normalized_return が反転して意味を失う。ここで止める。
    if expert_mean <= random_mean:
        raise ValueError(
            f"専門家がランダム行動を上回っていません "
            f"(expert={expert_mean:.2f} <= random={random_mean:.2f})。"
            f" --expert-timesteps を増やしてください（現在 {args.expert_timesteps}）。"
        )

    rollouts = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=args.n_episodes),
        rng=np.random.default_rng(args.seed),
    )
    print(f"collected {len(rollouts)} trajectories")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serialize.save(str(output_dir / "demos.npz"), rollouts)
    expert.save(str(output_dir / "expert_policy.zip"))
    (output_dir / "scores.json").write_text(
        json.dumps(
            {
                "env_id": args.env_id,
                "seed": args.seed,
                "expert_timesteps": args.expert_timesteps,
                "n_trajectories": len(rollouts),
                "random_mean": random_mean,
                "random_std": random_std,
                "expert_mean": expert_mean,
                "expert_std": expert_std,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mlflow.log_metrics(
        {
            "random_mean": random_mean,
            "random_std": random_std,
            "expert_mean": expert_mean,
            "expert_std": expert_std,
            "n_trajectories": len(rollouts),
        }
    )
    log_pip_freeze()


if __name__ == "__main__":
    main()
