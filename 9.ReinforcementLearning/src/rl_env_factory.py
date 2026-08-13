"""強化学習の環境を生成するファクトリ。

演習1（報酬関数の改善実験）では、**このファイルだけを変更**して条件を切り替える。

--------------------------------------------------------------------------
【最重要】HER と報酬ラッパーの整合性について
--------------------------------------------------------------------------
Stable-Baselines3 の HER (HerReplayBuffer) は、リプレイバッファからサンプルする
たびに「ゴールを差し替えて報酬を再計算」する。その再計算には環境の
``compute_reward()`` を使う。

  出典（参考情報・OSS 公式ドキュメント）:
    https://stable-baselines3.readthedocs.io/en/master/modules/her.html
    「HER requires the environment to ... have a vectorized implementation of
     compute_reward()」

一方 gymnasium の ``Wrapper`` は、未定義の属性アクセスをラップ元へ委譲する。
つまり ``step()`` の報酬だけをラッパーで書き換えると、

    step() が返す報酬  : 変更後の報酬関数
    HER が再計算する報酬: 変更前（環境本来）の報酬関数

という **不整合** が起きる。しかもエラーにならず、静かに学習信号が壊れる。

そこで本ファイルの ``ShapedRewardWrapper`` は
**``step()`` と ``compute_reward()`` の両方を同じ式で差し替える**。
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np

import panda_gym  # noqa: F401  import すると PandaReach-v3 などの環境 ID が登録される

#: 本ハンズオンで比較する報酬設計の一覧
REWARD_MODES = (
    "sparse",              # 既定。成功なら 0、それ以外は -1
    "dense",               # panda-gym の Dense 版環境を使う（報酬 = -距離）
    "sparse_shaped",       # sparse + 距離に応じた補助報酬（報酬シェーピング）
    "sparse_time_penalty",  # sparse + 1 ステップごとの時間ペナルティ
)


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ゴール間のユークリッド距離。単一サンプルにもバッチにも対応する。

    HER は (N, 3) の配列をまとめて渡してくるため、``axis=-1`` で縮約する。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b, axis=-1)


class ShapedRewardWrapper(gym.Wrapper):
    """報酬を差し替えるラッパー（HER 整合性を保つ）。

    Args:
        env: ラップ対象の環境。
        mode: ``sparse_shaped`` または ``sparse_time_penalty``。
        weight: 追加項の重み。**この値の大小が実験の主役になる。**

    Note:
        panda-gym は ``compute_reward`` を **インスタンス属性** として持つ
        (``self.compute_reward = self.task.compute_reward``)。
        一方このクラスは ``compute_reward`` を **クラスのメソッド** として定義する。

        Python の属性解決では、`wrapper.compute_reward` はまずラッパー自身の
        クラス属性が見つかるため、``Wrapper.__getattr__``（ラップ元への委譲）は
        呼ばれない。したがって **ラッパーの実装が優先される**。

        さらに SB3 の VecEnv 経由（``env_method("compute_reward", ...)``）でも、
        Monitor など上位ラッパーの ``__getattr__`` がラップ元へ委譲していき、
        最初に定義を持つ本ラッパーで解決される。

        ただしこの挙動はライブラリのバージョンに依存しうるため、
        :func:`verify_reward_consistency` で **実行時に必ず検証する**。
    """

    def __init__(self, env: gym.Env, mode: str, weight: float) -> None:
        super().__init__(env)
        if mode not in ("sparse_shaped", "sparse_time_penalty"):
            raise ValueError(f"ShapedRewardWrapper が対応しない mode です: {mode}")
        self.mode = mode
        self.weight = float(weight)

    # ------------------------------------------------------------------
    # HER から呼ばれる。必ずベクトル化されていること。
    # ------------------------------------------------------------------
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Any = None,
    ) -> np.ndarray:
        base = np.asarray(
            self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info),
            dtype=np.float32,
        )
        if self.mode == "sparse_shaped":
            # 目標に近いほど報酬を上げる（距離のマイナスを重み付きで加算）
            return (base - self.weight * _distance(achieved_goal, desired_goal)).astype(np.float32)
        # sparse_time_penalty: 1 ステップ経過するごとに一定量を引く
        return (base - self.weight).astype(np.float32)

    def step(self, action):  # type: ignore[override]
        obs, _reward, terminated, truncated, info = self.env.step(action)
        reward = float(
            np.asarray(
                self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
            ).reshape(-1)[0]
        )
        return obs, reward, terminated, truncated, info


def resolve_env_id(env_id: str, reward_mode: str) -> str:
    """報酬モードに応じて実際に使う環境 ID を決める。

    panda-gym は Dense 版を別の環境 ID として登録している
    （例: ``PandaPickAndPlace-v3`` に対して ``PandaPickAndPlaceDense-v3``）。
      出典: https://panda-gym.readthedocs.io/en/latest/usage/environments.html
    """
    if reward_mode != "dense":
        return env_id
    if "Dense" in env_id:
        return env_id
    if not env_id.endswith("-v3"):
        raise ValueError(f"想定外の環境 ID です: {env_id}")
    return env_id[: -len("-v3")] + "Dense-v3"


def verify_reward_consistency(env: gym.Env, n_checks: int = 3) -> None:
    """``step()`` の報酬と ``compute_reward()`` の戻り値が一致することを検証する。

    **なぜ必要か**: HER はリプレイ時に ``compute_reward()`` で報酬を再計算する。
    ここが ``step()`` と食い違うと、**エラーを出さずに学習信号だけが壊れる**。
    最も発見しにくいバグなので、環境生成のたびに検証する。

    Raises:
        RuntimeError: 不一致を検出した場合。
    """
    obs, _info = env.reset(seed=12345)
    for _ in range(n_checks):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        recomputed = np.asarray(
            env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        ).reshape(-1)[0]
        if not np.isclose(float(reward), float(recomputed), atol=1e-5):
            raise RuntimeError(
                "step() の報酬と compute_reward() の戻り値が一致しません。\n"
                f"  step()          = {reward}\n"
                f"  compute_reward()= {recomputed}\n"
                "HER を使うと学習信号が壊れます。ShapedRewardWrapper の実装、"
                "および gymnasium / stable-baselines3 のバージョンを確認してください。"
            )
        if terminated or truncated:
            obs, _info = env.reset(seed=12345)


def make_env(
    env_id: str,
    reward_mode: str = "sparse",
    shaping_weight: float = 0.1,
    render: bool = False,
    seed: int | None = None,
) -> gym.Env:
    """1 個の環境インスタンスを生成する。

    Args:
        env_id: 例 ``PandaPickAndPlace-v3``。
        reward_mode: :data:`REWARD_MODES` のいずれか。
        shaping_weight: 報酬シェーピングの重み。
        render: True なら ``render_mode="rgb_array"`` で作る（動画収録用）。
        seed: 乱数シード。
    """
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"未知の reward_mode です: {reward_mode} (使えるのは {REWARD_MODES})")

    actual_id = resolve_env_id(env_id, reward_mode)
    kwargs: dict[str, Any] = {}
    if render:
        # "Tiny" は PyBullet のソフトウェア レンダラー。GPU も X サーバーも不要なため、
        # ヘッドレスな Azure ML のコンピューティングではこれが必須。
        kwargs["render_mode"] = "rgb_array"
        kwargs["renderer"] = "Tiny"

    try:
        env = gym.make(actual_id, **kwargs)
    except TypeError:
        # panda-gym のバージョン差で renderer 引数を受け付けない場合のフォールバック
        kwargs.pop("renderer", None)
        env = gym.make(actual_id, **kwargs)

    if reward_mode in ("sparse_shaped", "sparse_time_penalty"):
        env = ShapedRewardWrapper(env, mode=reward_mode, weight=shaping_weight)
        # ラッパーを付けたときだけ検証する（HER との整合性が崩れるのはこの場合だけ）
        verify_reward_consistency(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def make_env_fn(
    env_id: str,
    reward_mode: str = "sparse",
    shaping_weight: float = 0.1,
    render: bool = False,
    seed: int | None = None,
) -> Callable[[], gym.Env]:
    """Stable-Baselines3 の ``make_vec_env`` などに渡す生成関数を返す。"""

    def _init() -> gym.Env:
        return make_env(
            env_id=env_id,
            reward_mode=reward_mode,
            shaping_weight=shaping_weight,
            render=render,
            seed=seed,
        )

    return _init


def describe_env(env: gym.Env) -> dict[str, Any]:
    """環境の仕様を辞書で返す（MLflow に記録して再現性を担保するため）。

    値に ``None`` を含めない（MLflow のパラメーターとして意味を持たないため）。
    """
    spec = getattr(env, "spec", None)
    obs_space = env.observation_space
    keys = sorted(obs_space.spaces.keys()) if hasattr(obs_space, "spaces") else []
    return {
        "env_id": getattr(spec, "id", "unknown"),
        "max_episode_steps": getattr(spec, "max_episode_steps", None) or 50,
        "observation_keys": ",".join(keys) if keys else "not-a-dict-space",
        "action_dim": int(env.action_space.shape[0]),
        "action_low": float(np.min(env.action_space.low)),
        "action_high": float(np.max(env.action_space.high)),
    }
