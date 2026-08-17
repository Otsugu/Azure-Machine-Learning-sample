"""スクリプト専門家のデモンストレーションと評価基準を保存する。

出力（--output-dir 配下）:
  demos.npz   : 専門家の軌跡。BC / DAgger / GAIL の入力
  scores.json : 正規化リターンの計算に使う基準値

専門家は [scripted_expert.py](scripted_expert.py) に手続きとして書かれています。
学習は行わないため、このジョブは短時間で終わります。
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
from imitation.policies.base import RandomPolicy

from il_common import DEFAULT_ENV_ID, evaluate, log_pip_freeze, make_env, set_seed
from imitation.data import rollout, serialize
from scripted_expert import ScriptedExpertPolicy, rollout_policy

N_ENVS = 8
N_EVAL_EPISODES = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--n-episodes", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    # output-dir は実行ごとに変わる一時パスなので、実験比較のパラメーターには含めない
    mlflow.log_params({k: v for k, v in vars(args).items() if k != "output_dir"})

    venv = make_env(args.env_id, N_ENVS, args.seed)

    random_policy = RandomPolicy(venv.observation_space, venv.action_space)
    random_mean, random_std, random_success = evaluate(random_policy, venv, N_EVAL_EPISODES)
    print(f"uniform random  return={random_mean:8.2f} +/- {random_std:6.2f}  success={random_success:.2f}")

    expert_policy = ScriptedExpertPolicy(venv.observation_space, venv.action_space)
    expert_mean, expert_std, expert_success = evaluate(expert_policy, venv, N_EVAL_EPISODES)
    print(f"scripted expert return={expert_mean:8.2f} +/- {expert_std:6.2f}  success={expert_success:.2f}")

    # 専門家がランダム以下だと (expert_mean - random_mean) が 0 以下になり、
    # 後段の normalized_return が反転して意味を失う。ここで止める。
    if expert_mean <= random_mean:
        raise ValueError(
            f"専門家がランダム行動を上回っていません "
            f"(expert={expert_mean:.2f} <= random={random_mean:.2f})。"
            f" scripted_expert.py の制御パラメーターを確認してください。"
        )

    rollouts = rollout.rollout(
        rollout_policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=args.n_episodes),
        rng=np.random.default_rng(args.seed),
    )
    transitions = rollout.flatten_trajectories(rollouts)
    print(f"collected {len(rollouts)} trajectories -> {len(transitions)} transitions")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serialize.save(str(output_dir / "demos.npz"), rollouts)
    (output_dir / "scores.json").write_text(
        json.dumps(
            {
                "env_id": args.env_id,
                "seed": args.seed,
                "n_trajectories": len(rollouts),
                "n_transitions": len(transitions),
                "random_mean": random_mean,
                "random_std": random_std,
                "random_success_rate": random_success,
                "expert_mean": expert_mean,
                "expert_std": expert_std,
                "expert_success_rate": expert_success,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mlflow.log_metrics(
        {
            "random_mean": random_mean,
            "random_std": random_std,
            "random_success_rate": random_success,
            "expert_mean": expert_mean,
            "expert_std": expert_std,
            "expert_success_rate": expert_success,
            "n_trajectories": len(rollouts),
            "n_transitions": len(transitions),
        }
    )
    log_pip_freeze()


if __name__ == "__main__":
    main()
