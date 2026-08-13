"""学習した方策の評価と、評価動画の収録。

【なぜ独自の評価関数を書くのか】
Stable-Baselines3 の ``evaluate_policy`` は報酬とエピソード長は返すが、
**成功率（success rate）は返さない**。

強化学習では「報酬が上がっても望ましい動作とは限らない」（＝報酬ハッキング）ため、
**成功率と平均報酬を必ず両方記録する**必要がある。
そのため成功判定 ``info["is_success"]`` を明示的に集計する評価関数を用意する。

``info["is_success"]`` は panda-gym が返す（検証済み）。
  出典（参考情報・OSS 公式ソース）:
    https://github.com/qgallouedec/panda-gym/blob/master/panda_gym/envs/core.py
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np


def evaluate_policy_with_success(
    model: Any,
    env: Any,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: int | None = None,
) -> dict[str, float]:
    """方策を評価し、成功率・平均報酬・エピソード長などを返す。

    Args:
        model: Stable-Baselines3 のモデル。
        env: 評価用の（ベクトル化されていない）環境。
        n_episodes: 評価エピソード数。**少なすぎると成功率の推定が荒くなる。**
        deterministic: True なら確率的な探索をせず決定的に行動する。
        seed: 評価の再現性のためのシード。

    Returns:
        メトリックの辞書。すべて float。
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    successes: list[float] = []

    for ep in range(n_episodes):
        reset_seed = None if seed is None else seed + ep
        obs, info = env.reset(seed=reset_seed)
        done = False
        total_reward = 0.0
        steps = 0
        is_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            # 成功した瞬間を取りこぼさないよう、エピソード中に一度でも True なら成功とする
            is_success = is_success or bool(info.get("is_success", False))
            done = bool(terminated or truncated)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(1.0 if is_success else 0.0)

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(episode_rewards)),
        # ばらつきは「安定性」の評価指標そのもの。平均だけで判断しないために必ず記録する。
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        # 成功率の標準誤差。「成功率 0.30 ± 0.06」のように幅で語るために使う。
        "success_rate_stderr": float(
            statistics.pstdev(successes) / max(np.sqrt(len(successes)), 1.0)
        ),
        "n_eval_episodes": float(n_episodes),
    }


def record_video(
    model: Any,
    env: Any,
    video_path: str,
    n_episodes: int = 3,
    fps: int = 20,
    deterministic: bool = True,
    seed: int | None = None,
) -> bool:
    """評価の様子を mp4 に書き出す。

    **報酬ハッキングを検出する最も確実な手段が「動画を見ること」** なので、
    すべての実験でこれを残す。

    Returns:
        書き出しに成功したら True。失敗しても学習全体は止めないため例外は投げない。
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[WARN] imageio が無いため動画を収録できません。conda.yaml を確認してください。")
        return False

    frames: list[np.ndarray] = []
    try:
        for ep in range(n_episodes):
            reset_seed = None if seed is None else seed + ep
            obs, _info = env.reset(seed=reset_seed)
            done = False
            while not done:
                #   reset 直後の初期状態も 1 フレーム目として意図的に収録する
                #   （ロボットとキューブの初期配置が分からないと動作を評価できないため）
                frame = env.render()
                if frame is None:
                    print("[WARN] env.render() が None を返しました。"
                          "render_mode='rgb_array' で環境を作っているか確認してください。")
                    return False
                frames.append(np.asarray(frame))
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

        if not frames:
            return False

        with imageio.get_writer(video_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"評価動画を書き出しました: {video_path} ({len(frames)} フレーム)")
        return True

    except Exception as exc:  # noqa: BLE001  動画は補助情報なので学習を止めない
        print(f"[WARN] 動画の書き出しに失敗しました: {type(exc).__name__}: {exc}")
        return False
