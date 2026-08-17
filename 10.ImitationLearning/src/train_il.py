"""専門家デモから方策を学習する（BC / DAgger / GAIL）。

Sweep Job の主要メトリックは normalized_return です。
  normalized_return = (方策のリターン - 一様ランダムのリターン) / (専門家のリターン - 一様ランダムのリターン)
基準値は collect_demos.py が出力した scores.json から読み込みます。
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from il_common import evaluate, log_pip_freeze, make_env
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import rollout, serialize
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

N_ENVS = 8
N_EVAL_EPISODES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=("bc", "dagger", "gail"), required=True)
    parser.add_argument("--demos-dir", required=True)
    parser.add_argument("--n-demo-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    # BC / DAgger 用
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    # DAgger 用
    parser.add_argument("--dagger-timesteps", type=int, default=8_000)
    # GAIL 用
    parser.add_argument("--gail-timesteps", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # demos-dir と output-dir は実行ごとに変わるパスなので、比較用パラメーターに含めない
    mlflow.log_params(
        {k: v for k, v in vars(args).items() if k not in ("demos_dir", "output_dir")}
    )

    demos_dir = Path(args.demos_dir)
    scores = json.loads((demos_dir / "scores.json").read_text(encoding="utf-8"))
    env_id = scores["env_id"]
    random_mean = scores["random_mean"]
    expert_mean = scores["expert_mean"]

    rollouts = serialize.load(str(demos_dir / "demos.npz"))[: args.n_demo_episodes]
    transitions = rollout.flatten_trajectories(rollouts)
    print(f"{len(rollouts)} trajectories -> {len(transitions)} transitions")
    mlflow.log_metric("n_transitions", len(transitions))

    venv = make_env(env_id, N_ENVS, args.seed)

    if args.algo == "gail":
        learner = PPO(
            env=venv,
            policy=MlpPolicy,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0004,
            gamma=0.95,
            n_epochs=5,
            seed=args.seed,
            verbose=0,
        )
        GAIL(
            demonstrations=rollouts,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=8,
            venv=venv,
            gen_algo=learner,
            reward_net=BasicRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                normalize_input_layer=RunningNorm,
            ),
        ).train(args.gail_timesteps)
        policy = learner.policy
    else:
        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=transitions,
            batch_size=args.batch_size,
            rng=np.random.default_rng(args.seed),
        )
        if args.algo == "bc":
            epoch = 0

            def on_epoch_end() -> None:
                # Sweep の早期終了ポリシーが判断できるよう、途中経過を記録する。
                # 最終値 eval_return_mean とは別名にして、系列が混ざらないようにする。
                nonlocal epoch
                epoch += 1
                mean, _ = evaluate(bc_trainer.policy, venv, N_EVAL_EPISODES)
                mlflow.log_metric("epoch_eval_return_mean", mean, step=epoch)

            bc_trainer.train(
                n_epochs=args.epochs, on_epoch_end=on_epoch_end, progress_bar=False
            )
            policy = bc_trainer.policy
        else:
            expert = PPO.load(str(demos_dir / "expert_policy.zip"), env=venv)
            with tempfile.TemporaryDirectory() as scratch:
                dagger = SimpleDAggerTrainer(
                    venv=venv,
                    scratch_dir=scratch,
                    expert_policy=expert,
                    bc_trainer=bc_trainer,
                    rng=np.random.default_rng(args.seed),
                )
                dagger.train(args.dagger_timesteps)
            policy = bc_trainer.policy

    eval_mean, eval_std = evaluate(policy, venv, N_EVAL_EPISODES)
    normalized = (eval_mean - random_mean) / (expert_mean - random_mean)
    print(f"eval mean={eval_mean:8.2f} std={eval_std:7.2f} normalized={normalized:6.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy, output_dir / "policy.pt")

    mlflow.log_metrics(
        {
            "eval_return_mean": eval_mean,
            "eval_return_std": eval_std,
            "normalized_return": normalized,
            "random_mean": random_mean,
            "expert_mean": expert_mean,
        }
    )
    log_pip_freeze()


if __name__ == "__main__":
    main()
