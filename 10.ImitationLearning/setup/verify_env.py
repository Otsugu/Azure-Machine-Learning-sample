"""conda 環境 il-local が正しく構築できたかを検証する。

setup.ps1 / setup.sh の最後に自動実行されます。
このファイルがある setup フォルダーで、単体で実行することもできます。

    conda run -n il-local --no-capture-output python verify_env.py

検証に失敗した場合は終了コード 1 を返します。
"""

import importlib.metadata as metadata
import platform
import sys

#: 検証対象の配布パッケージ名。environment-local.yml の内容と対応している。
PACKAGES = (
    "numpy",
    "torch",
    "gymnasium",
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
    print("[3/4] seals/CartPole-v0 の動作確認（エピソード長が固定であること）")

    import gymnasium as gym
    import seals  # noqa: F401  # import すると seals/* 環境が gymnasium に登録される

    env = gym.make("seals:seals/CartPole-v0")
    lengths = []
    try:
        for episode in range(3):
            env.reset(seed=episode)
            steps = 0
            while True:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                steps += 1
                if terminated or truncated:
                    break
            lengths.append(steps)
        print(f"  observation_space : {env.observation_space}")
        print(f"  action_space      : {env.action_space}")
        print(f"  episode lengths   : {lengths}")
    finally:
        env.close()

    if len(set(lengths)) != 1:
        fail(
            f"エピソード長が一定ではありません: {lengths}\n"
            "    固定ホライズンでないと GAIL / AIRL の評価が成立しません。\n"
            f"    {RETRY_HINT}"
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
