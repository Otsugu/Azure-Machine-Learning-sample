"""conda 環境 il-panda が正しく構築できたかを検証する。

setup.ps1 / setup.sh の最後に自動実行されます。
このファイルがある setup フォルダーで、単体で実行することもできます。

    conda run -n il-panda --no-capture-output python verify_env.py

検証に失敗した場合は終了コード 1 を返します。
"""

import importlib.metadata as metadata
import platform
import sys
from pathlib import Path

#: 検証対象の配布パッケージ名。environment-local.yml の内容と対応している。
PACKAGES = (
    "numpy",
    "scipy",
    "pybullet",
    "torch",
    "gymnasium",
    "panda-gym",
    "seals",
    "stable-baselines3",
    "imitation",
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


def check_fixed_horizon() -> None:
    print("")
    print("[3/4] ピックアンドプレース環境の動作確認（エピソード長が固定であること）")

    #  ../src の環境定義をそのまま使う（検証と本番で定義がずれないように）
    sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

    import gymnasium as gym

    from pick_place_env import DEFAULT_ENV_ID, HORIZON  # noqa: E402
    from scripted_expert import scripted_action  # noqa: E402

    env = gym.make(DEFAULT_ENV_ID)
    lengths, successes = [], []
    try:
        for episode in range(3):
            obs, _ = env.reset(seed=episode)
            steps, success = 0, False
            while True:
                obs, _, terminated, truncated, info = env.step(scripted_action(obs))
                steps += 1
                success = success or bool(info.get("is_success", False))
                if terminated or truncated:
                    break
            lengths.append(steps)
            successes.append(success)
        print(f"  env_id            : {DEFAULT_ENV_ID}")
        print(f"  observation_space : {env.observation_space}")
        print(f"  action_space      : {env.action_space}")
        print(f"  episode lengths   : {lengths}")
        print(f"  expert successes  : {successes}")
    finally:
        env.close()

    if set(lengths) != {HORIZON}:
        fail(
            f"エピソード長が {HORIZON} で一定ではありません: {lengths}\n"
            "    固定ホライズンでないと GAIL / AIRL の評価が成立しません。\n"
            f"    {RETRY_HINT}"
        )
    if not any(successes):
        fail(
            "スクリプト専門家が 1 度も成功しませんでした。\n"
            f"    物理エンジンの導入が不完全な可能性があります。\n    {RETRY_HINT}"
        )


def check_azure_sdk() -> None:
    print("")
    print("[4/4] Azure ML SDK と MLflow の import 確認")

    import mlflow
    from azure.ai.ml import MLClient  # noqa: F401
    from azure.identity import DefaultAzureCredential  # noqa: F401

    print(f"  mlflow tracking uri : {mlflow.get_tracking_uri()}")
    print("  ※ ローカル実行では、この追跡 URI をワークスペースのものに差し替える必要があります")
    print("     （docs/03_AzureML環境構築.md）")


def main() -> None:
    show_platform()
    check_packages()
    check_fixed_horizon()
    check_azure_sdk()

    print("")
    print("OK: ローカル環境の検証に成功しました。")


if __name__ == "__main__":
    main()
