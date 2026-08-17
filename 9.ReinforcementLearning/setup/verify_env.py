"""conda 環境 rl-local が正しく構築できたかを検証する。

setup.ps1 / setup.sh の最後に自動実行されます。
このファイルがある setup フォルダーで、単体で実行することもできます。

    conda activate rl-local
    python verify_env.py

検証に失敗した場合は終了コード 1 を返します。
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys

#: 検証対象の配布パッケージ名。environment-local.yml の内容と対応している。
PACKAGES = (
    "numpy",
    "scipy",
    "pybullet",
    "gymnasium",
    "panda-gym",
    "stable-baselines3",
    "imageio",
    "imageio-ffmpeg",
    "pandas",
    "matplotlib",
    "azure-ai-ml",
    "azure-identity",
    "mlflow",
    "azureml-mlflow",
    "ipykernel",
)

RETRY_HINT = "セットアップ スクリプト（setup.ps1 / setup.sh）を再実行してください。"


def fail(message: str) -> None:
    print("", file=sys.stderr)
    print(f"NG: {message}", file=sys.stderr)
    sys.exit(1)


def show_platform() -> None:
    print("[1/4] 実行環境")
    print(f"  os         : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  python     : {sys.version.split()[0]}")
    print(f"  executable : {sys.executable}")


def check_packages() -> None:
    print("")
    print("[2/4] パッケージの導入状況")
    missing = []
    for name in PACKAGES:
        try:
            print(f"  {name:<20} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"  {name:<20} （未導入）")
            missing.append(name)
    if missing:
        fail(f"次のパッケージが見つかりません: {', '.join(missing)}\n    {RETRY_HINT}")

    import numpy

    if int(numpy.__version__.split(".")[0]) >= 2:
        fail(
            f"numpy {numpy.__version__} が導入されています。panda-gym は numpy<2 を要求します。\n"
            f"    {RETRY_HINT}"
        )


def check_rl_env() -> None:
    print("")
    print("[3/4] panda-gym の動作確認（PandaReach-v3）")

    import gymnasium as gym
    import numpy as np
    import panda_gym  # noqa: F401  # import すると Panda 系の環境が gymnasium に登録される

    # renderer="Tiny" は PyBullet の DIRECT 接続。ウィンドウを開かずに画像だけ取得する
    env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="Tiny")
    try:
        obs, _info = env.reset(seed=0)
        env.step(env.action_space.sample())
        frame = np.asarray(env.render())
        print(f"  observation shape : {obs['observation'].shape}")
        print(f"  action_space      : {env.action_space}")
        print(f"  render frame      : {frame.shape} {frame.dtype}")
    finally:
        env.close()


def check_azure_sdk() -> None:
    print("")
    print("[4/4] Azure ML SDK と MLflow の import 確認")

    import mlflow
    from azure.ai.ml import MLClient  # noqa: F401
    from azure.identity import DefaultAzureCredential  # noqa: F401

    print(f"  mlflow tracking uri : {mlflow.get_tracking_uri()}")
    print("  ※ ローカル実行では、この追跡 URI をワークスペースのものに差し替える必要があります")
    print("     （docs/03_AzureML環境構築.md の 3.6）")


def main() -> None:
    show_platform()
    check_packages()
    check_rl_env()
    check_azure_sdk()

    print("")
    print("OK: ローカル環境の検証に成功しました。")


if __name__ == "__main__":
    main()
