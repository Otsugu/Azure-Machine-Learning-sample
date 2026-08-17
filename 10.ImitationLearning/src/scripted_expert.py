"""スクリプト専門家（人が手続きとして書いたピックアンドプレースのお手本）。

**記憶を持ちません。** 行動は「いまの観測」だけで決まります。
BC は「観測 → 行動」の対応を学ぶため、専門家が履歴に依存していると原理的に模倣できません。

行動は 4 次元 `Box(-1, 1)` で、手先の移動量は行動 × 0.05、指の開閉は行動 × 0.2 を現在の幅に加算します。
  出典: https://github.com/qgallouedec/panda-gym/blob/master/panda_gym/envs/robots/panda.py
"""

import numpy as np
import torch
from imitation.policies import base as imitation_policies

from pick_place_env import unflatten_observation

#: 物体の真上に構える高さ [m]
APPROACH_HEIGHT = 0.05
#: 「真上に来た」と判定する水平距離 [m]
XY_TOLERANCE = 0.01
#: 「掴める位置に来た」と判定する距離 [m]
GRASP_TOLERANCE = 0.015
#: 「掴んだまま」と判定する距離 [m]。掴んだ後は物体が手先と一緒に動く
HOLD_TOLERANCE = 0.03
#: 指がこの幅より狭ければ「閉じている」とみなす [m]（全開は 0.08）
HOLD_WIDTH = 0.045
#: 手先の移動量は行動 × 0.05 なので、1 / 0.05 = 20 が「目標まで一度に動く」ゲイン
GAIN = 20.0


def scripted_action(flat_observation) -> np.ndarray:
    """平坦化された観測 1 件から行動を返す。

    判定は **「掴んでいるか → 掴める位置か → それ以外」の順**でなければなりません。
    順序を逆にすると、運搬中に把持判定が外れてグリッパーを開き、物体を落とします。
    """
    obs = unflatten_observation(flat_observation)
    ee = np.asarray(obs["observation"][0:3], dtype=np.float64)      # 手先の位置
    finger_width = float(obs["observation"][6])                      # 指の開き幅
    obj = np.asarray(obs["achieved_goal"], dtype=np.float64)         # 物体の位置
    goal = np.asarray(obs["desired_goal"], dtype=np.float64)         # 目標の位置

    distance = np.linalg.norm(ee - obj)
    if finger_width < HOLD_WIDTH and distance < HOLD_TOLERANCE:
        target, grip = ee + (goal - obj), -1.0                       # 掴んでいる → 運ぶ
    elif distance < GRASP_TOLERANCE:
        target, grip = obj, -1.0                                     # 掴める位置 → 閉じる
    elif np.linalg.norm(ee[:2] - obj[:2]) > XY_TOLERANCE:
        target, grip = obj + np.array([0.0, 0.0, APPROACH_HEIGHT]), 1.0   # 物体の真上へ
    else:
        target, grip = obj, 1.0                                      # 真上から降下

    delta = target - ee
    return np.concatenate([np.clip(delta * GAIN, -1.0, 1.0), [grip]]).astype(np.float32)


def scripted_actions(flat_observations) -> np.ndarray:
    return np.stack([scripted_action(obs) for obs in np.asarray(flat_observations)])


def rollout_policy(observations, states, episode_starts):
    """imitation の `rollout.rollout` に渡す方策。"""
    return scripted_actions(observations), states


class ScriptedExpertPolicy(imitation_policies.NonTrainablePolicy):
    """DAgger が学習中に問い合わせるための専門家。

    `NonTrainablePolicy` は「手書き方策」のための抽象クラスとして imitation が用意しています。
      出典: https://imitation.readthedocs.io/en/latest/_api/imitation.policies.base.html
    """

    def _choose_action(self, obs):
        array = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
        if array.ndim == 1:
            return scripted_action(array)
        return scripted_actions(array)
