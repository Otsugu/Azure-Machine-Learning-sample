"""collect_demos.py と train_il.py が共有するヘルパー。

Azure ML 固有のコードは書きません。記録は MLflow に統一します。
出典: https://learn.microsoft.com/azure/machine-learning/migrate-to-v2-execution-hyperdrive?view=azureml-api-2
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import seals  # noqa: F401  # import すると seals/* 環境が gymnasium に登録される
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

#: 本ハンズオンの既定環境。seals 版は 1 エピソードが必ず 500 ステップで終わる。
#: 可変ホライズン環境では GAIL / AIRL の評価が成立しないため、固定ホライズン版を使う。
#: 出典: https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html
DEFAULT_ENV_ID = "seals:seals/CartPole-v0"


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


def evaluate(policy, venv: VecEnv, n_episodes: int) -> tuple[float, float]:
    """方策の平均リターンと標準偏差を返す。"""
    mean, std = evaluate_policy(policy, venv, n_eval_episodes=n_episodes)
    return float(mean), float(std)


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
