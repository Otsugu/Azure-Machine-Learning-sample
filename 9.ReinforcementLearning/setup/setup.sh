#!/usr/bin/env bash
# =====================================================================
#  9.ReinforcementLearning ローカル環境セットアップ（macOS / Linux）
#
#  実行方法:
#      bash setup.sh
#
#  このスクリプトが行うこと:
#      1. Miniforge (conda) の導入          … 未導入の場合のみ
#      2. conda 環境 rl-local の作成 / 更新
#      3. Jupyter カーネル「Python (rl-local)」の登録
#      4. Azure CLI と ml 拡張の導入        … 未導入の場合のみ
#      5. verify_env.py による動作確認
#
#  Azure のワークスペースやコンピューティングの作成は行いません。
#  それらは docs/03_AzureML環境構築.md と notebooks/01_setup_azureml.ipynb で実施します。
#
#  何度実行しても同じ結果になります（既に導入済みのものはスキップされます）。
# =====================================================================
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SETUP_DIR/environment-local.yml"
VERIFY_SCRIPT="$SETUP_DIR/verify_env.py"
ENV_NAME="rl-local"
MINIFORGE_DIR="$HOME/miniforge3"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

step() {
    printf '\n=== %s ===\n' "$1"
}

find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
    elif [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
        # conda init 済みの環境では PATH ではなく CONDA_EXE で示されていることがある
        echo "$CONDA_EXE"
    elif [ -x "$MINIFORGE_DIR/bin/conda" ]; then
        echo "$MINIFORGE_DIR/bin/conda"
    fi
}

get_env_python() {
    # conda run は子プロセスの出力を返さないことがあるため、環境の python を直接実行する
    local prefix
    prefix="$("$CONDA" env list | awk -v name="$ENV_NAME" '$1 == name { print $NF }')"
    if [ -n "$prefix" ] && [ -x "$prefix/bin/python" ]; then
        echo "$prefix/bin/python"
    fi
}

# --- 1. Miniforge (conda) -------------------------------------------
step "conda を確認します"
CONDA="$(find_conda)"

if [ -n "$CONDA" ]; then
    echo "  導入済みの conda を使用します: $CONDA"
else
    if ! command -v curl >/dev/null 2>&1; then
        echo "  curl が見つかりません。curl を導入してから、このスクリプトを再実行してください。" >&2
        exit 1
    fi

    echo "  conda が見つかりません。Miniforge を導入します（大きなダウンロードが発生します）。"
    echo "  $MINIFORGE_URL"
    installer="$(mktemp)"
    curl -fsSL -o "$installer" "$MINIFORGE_URL"
    bash "$installer" -b -p "$MINIFORGE_DIR"
    rm -f "$installer"

    CONDA="$(find_conda)"
    if [ -z "$CONDA" ]; then
        echo "  Miniforge を導入しましたが conda が見つかりません。$MINIFORGE_DIR を確認してください。" >&2
        exit 1
    fi
    echo "  導入しました: $CONDA"
fi

# --- 2. conda 環境 ---------------------------------------------------
step "conda 環境 $ENV_NAME を準備します（初回は多数のパッケージを取得するため時間がかかります）"
if [ -n "$(get_env_python)" ]; then
    echo "  既存の $ENV_NAME を environment-local.yml の内容に合わせて更新します。"
    "$CONDA" env update --name "$ENV_NAME" --file "$ENV_FILE" --prune
else
    echo "  $ENV_NAME を新規作成します。"
    "$CONDA" env create --file "$ENV_FILE"
fi

ENV_PYTHON="$(get_env_python)"
if [ -z "$ENV_PYTHON" ]; then
    echo "  conda 環境 $ENV_NAME の Python が見つかりません。" >&2
    exit 1
fi

# --- 3. Jupyter カーネル ---------------------------------------------
step "Jupyter カーネルを登録します"
"$ENV_PYTHON" -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# --- 4. Azure CLI ----------------------------------------------------
# 導入に失敗してもローカル環境の構築は続行する（Azure CLI は後から手動でも導入できるため）。
# そのため、この関数は失敗しても 0 を返す。
step "Azure CLI を確認します"
install_azure_cli() {
    if command -v az >/dev/null 2>&1; then
        echo "  Azure CLI は導入済みです。"
    elif [ "$(uname)" = "Darwin" ]; then
        if ! command -v brew >/dev/null 2>&1; then
            echo "  Homebrew が見つかりません。次のページから手動で導入してください。"
            echo "      https://learn.microsoft.com/cli/azure/install-azure-cli-macos"
            return 0
        fi
        echo "  Homebrew で Azure CLI を導入します。"
        if ! { brew update && brew install azure-cli; }; then
            echo "  警告: Azure CLI の導入に失敗しました。ローカル環境の構築は続行します。" >&2
            return 0
        fi
    elif [ -r /etc/os-release ] && grep -qiE '^(ID|ID_LIKE)=.*(debian|ubuntu)' /etc/os-release; then
        echo "  Azure CLI を導入します。sudo のパスワードを求められます。"
        if ! curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash; then
            echo "  警告: Azure CLI の導入に失敗しました。ローカル環境の構築は続行します。" >&2
            return 0
        fi
    else
        echo "  このディストリビューションでは自動導入を行いません。次のページから手動で導入してください。"
        echo "      https://learn.microsoft.com/cli/azure/install-azure-cli-linux"
        return 0
    fi

    # ここに到達するのは az が使える場合だけ
    if az extension add --name ml --upgrade --only-show-errors; then
        echo "  ml 拡張を最新にしました。"
    else
        echo "  警告: ml 拡張の追加に失敗しました（本ハンズオンでは任意の機能です）。" >&2
    fi
    return 0
}

install_azure_cli

# --- 5. 検証 ---------------------------------------------------------
step "導入結果を検証します"
"$ENV_PYTHON" "$VERIFY_SCRIPT"

# --- 6. 次の手順 -----------------------------------------------------
cat <<EOF

=== セットアップが完了しました ===

次の手順に進んでください。

  1. Notebook を開く
         VS Code   : notebooks/01_setup_azureml.ipynb を開き、
                     右上のカーネルで「Python ($ENV_NAME)」を選ぶ
         JupyterLab: conda activate $ENV_NAME
                     jupyter lab

  2. Notebook の「1. セットアップ」から順にセルを実行する
     ※ Azure へのサインインも Notebook の中で行うため、az login は不要です
        （az login を済ませてあれば、その資格情報がそのまま使われます）

  3. ロボットが動く様子を GUI で見る（ローカル実行だけの特典です）
         docs/04_RL環境を触って理解する.md の 4.6

  ⚠ Azure のリソースは起動している間ずっと課金されます。
     終わったら docs/09_評価・コスト・後片付け.md の後片付けを必ず実施してください。
EOF
