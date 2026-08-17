"""collect_demos.py と train_il.py が共有するヘルパー。

Azure ML 固有のコードは書きません。記録は MLflow に統一します。
出典: https://learn.microsoft.com/azure/machine-learning/migrate-to-v2-execution-hyperdrive?view=azureml-api-2
"""

import random
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from pick_place_env import (  # noqa: F401  # import すると環境が gymnasium に登録される
    DEFAULT_ENV_ID,
    HORIZON,
    VARIABLE_HORIZON_ENV_ID,
)


def set_seed(seed: int) -> None:
    """実験を再現できるよう乱数を固定する。

    imitation の `rng` 引数はデータのシャッフルだけを控制し、
    **ニューラルネットワークの初期値は PyTorch のグローバル乱数が決めます**。
    これを固定しないと、同じ --seed でも実行のたびに結果が変わります。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, n_envs: int, seed: int) -> VecEnv:
    """学習・評価に使うベクトル化環境を作る。

    RolloutInfoWrapper は imitation の rollout 収集に必要。
    """
    return make_vec_env(
        env_id,
        rng=np.random.default_rng(seed),
        n_envs=n_envs,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )


def evaluate(policy, venv: VecEnv, n_episodes: int) -> tuple[float, float, float]:
    """平均リターン・標準偏差・成功率を返す。

    本ハンズオンの環境は 1 エピソードが必ず HORIZON ステップで終わるため、
    HORIZON ステップ進めるごとに全環境のエピソードが同時に完了する。
    """
    returns: list[float] = []
    successes: list[float] = []
    while len(returns) < n_episodes:
        obs = venv.reset()
        episode_return = np.zeros(venv.num_envs)
        episode_success = np.zeros(venv.num_envs, dtype=bool)
        for _ in range(HORIZON):
            actions, _ = policy.predict(obs, deterministic=True)
            obs, rewards, _dones, infos = venv.step(actions)
            episode_return += rewards
            episode_success |= np.array([bool(i.get("is_success", False)) for i in infos])
        returns.extend(episode_return.tolist())
        successes.extend(episode_success.astype(float).tolist())

    returns = returns[:n_episodes]
    successes = successes[:n_episodes]
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(successes))


def log_pip_freeze() -> None:
    """導入済みパッケージ一覧を MLflow アーティファクトとして保存する。

    社内の OSS ライセンス審査には、この pip_freeze.txt を提出します。
    """
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pip_freeze.txt"
        path.write_text(frozen, encoding="utf-8")
        mlflow.log_artifact(str(path))
