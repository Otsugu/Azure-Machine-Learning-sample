"""ピックアンドプレース環境の生成。

素の `PandaPickAndPlace-v3` は **成功した瞬間にエピソードが終わる**（可変ホライズン）ため、
そのままでは GAIL / AIRL の評価が成立しません。
  出典: https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html

そこで seals が MountainCar を固定ホライズン化しているのと同じ構成にします。
  内側の環境 → AbsorbAfterDoneWrapper → 外側の TimeLimit
  出典: https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/classic_control.py
"""

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation

import panda_gym  # noqa: F401  # import すると Panda 系の環境が gymnasium に登録される
from seals.util import AbsorbAfterDoneWrapper

#: 素のタスク環境。成功した瞬間に terminated=True になる（可変ホライズン）。
BASE_ENV_ID = "PandaPickAndPlace-v3"

#: 本ハンズオンの既定環境。固定ホライズン化し、観測を 1 本のベクトルに平坦化したもの。
DEFAULT_ENV_ID = "il/PandaPickAndPlace-v0"

#: 素の環境を平坦化しただけのもの。07 章で「可変ホライズンだと GAIL が止まる」ことを見るために使う。
VARIABLE_HORIZON_ENV_ID = "il/PandaPickAndPlaceVariable-v0"

#: panda-gym の登録値と同じ。1 エピソードは必ずこのステップ数で終わる。
HORIZON = 50

_probe = gym.make(BASE_ENV_ID)
#: 平坦化前の辞書型観測空間。平坦化した観測を元に戻すときに使う。
DICT_OBSERVATION_SPACE = _probe.observation_space
_probe.close()


class SuccessRecorderWrapper(gym.Wrapper):
    """エピソード中に一度でも成功したかを `info["is_success"]` に保持する。

    AbsorbAfterDoneWrapper は吸収状態に入ると info を空の辞書にするため、
    そのままでは成功した事実が次のステップ以降に伝わらない。
      出典: https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/util.py
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._success = bool(info.get("is_success", False))
        info["is_success"] = self._success
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._success = self._success or bool(info.get("is_success", False))
        info["is_success"] = self._success
        return obs, reward, terminated, truncated, info


class ClipObservationWrapper(gym.ObservationWrapper):
    """観測を observation_space の範囲に収める。

    panda-gym の観測空間は `Box(-10.0, 10.0, ...)` と宣言されていますが、
    実際の値はクリップされていません。学習途中の方策がロボットを激しく動かすと
    速度などが ±10 を超え、`observation_space.contains()` を前提にする実装が失敗します。
    実例: imitation の `NonTrainablePolicy._predict` は
    `assert self.observation_space.contains(...)` を行うため、DAgger が AssertionError で止まります。
    """

    def observation(self, observation):
        space = self.observation_space
        return np.clip(observation, space.low, space.high).astype(space.dtype)


def _make_fixed_horizon(**kwargs):
    env = gym.make(BASE_ENV_ID, **kwargs)
    env = AbsorbAfterDoneWrapper(env)      # 成功しても止めず、吸収状態にする
    env = SuccessRecorderWrapper(env)      # 吸収状態で消える is_success を保持する
    env = FlattenObservation(env)          # 辞書型観測を 1 本の 25 次元ベクトルにする
    return ClipObservationWrapper(env)     # 宣言された範囲に収める


def _make_variable_horizon(**kwargs):
    return ClipObservationWrapper(FlattenObservation(gym.make(BASE_ENV_ID, **kwargs)))


#  外側の TimeLimit は gym.register の max_episode_steps が掛ける（seals と同じ構成）
if DEFAULT_ENV_ID not in gym.registry:
    gym.register(
        id=DEFAULT_ENV_ID,
        entry_point=_make_fixed_horizon,
        max_episode_steps=HORIZON,
    )
if VARIABLE_HORIZON_ENV_ID not in gym.registry:
    gym.register(id=VARIABLE_HORIZON_ENV_ID, entry_point=_make_variable_horizon)


def unflatten_observation(flat_observation) -> dict:
    """平坦化された観測を、元の 3 つのキーを持つ辞書に戻す。

    並び順を手で決め打ちしないために Gymnasium の公式 API を使う。
    """
    from gymnasium import spaces

    return spaces.unflatten(
        DICT_OBSERVATION_SPACE, np.asarray(flat_observation, dtype=np.float32)
    )
