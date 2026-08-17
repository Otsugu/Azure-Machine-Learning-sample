# =====================================================================
#  10.ImitationLearning ローカル環境セットアップ（Windows）
#
#  実行方法（PowerShell 7 以降）:
#      pwsh -NoProfile -File .\setup.ps1
#
#  このスクリプトが行うこと:
#      1. Miniforge (conda) の導入          … 未導入の場合のみ
#      2. conda 環境 il-local の作成 / 更新
#      3. Jupyter カーネル「Python (il-local)」の登録
#      4. Azure CLI と ml 拡張の導入        … 未導入の場合のみ
#      5. verify_env.py による動作確認
#
#  Azure のワークスペースやコンピューティングの作成は行いません。
#  それらは docs/03_AzureML環境構築.md と notebooks/01_setup_azureml.ipynb で実施します。
#
#  何度実行しても同じ結果になります（既に導入済みのものはスキップされます）。
# =====================================================================

$ErrorActionPreference = 'Stop'

$SetupDir     = $PSScriptRoot
$EnvFile      = Join-Path $SetupDir 'environment-local.yml'
$VerifyScript = Join-Path $SetupDir 'verify_env.py'
$EnvName      = 'il-local'
$MiniforgeDir = Join-Path $env:USERPROFILE 'miniforge3'
$MiniforgeUrl = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe'

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Find-Conda {
    $command = Get-Command conda -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    # conda init 済みの環境では PATH ではなく CONDA_EXE で示されていることがある
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }

    $candidate = Join-Path $MiniforgeDir 'Scripts\conda.exe'
    if (Test-Path $candidate) { return $candidate }

    return $null
}

# --- 0. 実行環境の確認 -------------------------------------------------
if ($PSVersionTable.PSEdition -ne 'Core' -or $PSVersionTable.PSVersion.Major -lt 7) {
    Write-Host 'このスクリプトは PowerShell 7 以降が必要です。' -ForegroundColor Red
    Write-Host '次のコマンドで導入し、pwsh から実行し直してください。'
    Write-Host '    winget install --exact --id Microsoft.PowerShell'
    Write-Host '    pwsh -NoProfile -File .\setup.ps1'
    exit 1
}

# Miniforge はインストール先に空白や非 ASCII 文字を含めないことを推奨している
# 出典（参考情報・サードパーティ）: https://github.com/conda-forge/miniforge#windows
if ($MiniforgeDir -match '\s' -or $MiniforgeDir -match '[^\x20-\x7E]') {
    Write-Host "既定のインストール先 $MiniforgeDir に空白または非 ASCII 文字が含まれています。" -ForegroundColor Red
    Write-Host 'Miniforge を C:\miniforge3 など短い英数字のパスへ手動で導入してから、このスクリプトを再実行してください。'
    Write-Host '    https://github.com/conda-forge/miniforge'
    exit 1
}

# --- 1. Miniforge (conda) -------------------------------------------
Write-Step 'conda を確認します'
$conda = Find-Conda

if ($conda) {
    Write-Host "  導入済みの conda を使用します: $conda"
}
else {
    Write-Host '  conda が見つかりません。Miniforge を導入します（大きなダウンロードが発生します）。'
    $installer = Join-Path $env:TEMP 'Miniforge3-Windows-x86_64.exe'
    $ProgressPreference = 'SilentlyContinue'   # 進捗バーを消すとダウンロードが大幅に速くなる
    Invoke-WebRequest -Uri $MiniforgeUrl -OutFile $installer
    $ProgressPreference = 'Continue'

    # /D は NSIS の仕様上、最後の引数かつクォートなしで渡す必要がある
    $arguments = "/InstallationType=JustMe /RegisterPython=0 /S /D=$MiniforgeDir"
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Miniforge のインストールに失敗しました (exit code = $($process.ExitCode))"
    }
    Remove-Item $installer -Force

    $conda = Find-Conda
    if (-not $conda) {
        throw "Miniforge を導入しましたが conda が見つかりません。$MiniforgeDir を確認してください。"
    }
    Write-Host "  導入しました: $conda"
}

# --- 2. conda 環境 ---------------------------------------------------
# 別の conda コマンド（別ターミナルでの env create など）が実行中だと、
# conda はロック待ちで無言のまま停止することがあります。
# 進まない場合は、他に conda を実行しているターミナルが無いか確認してください。
Write-Step "conda 環境 $EnvName を準備します（初回は多数のパッケージを取得するため時間がかかります）"
$envList = & $conda env list
if ($LASTEXITCODE -ne 0) { throw 'conda env list に失敗しました。' }

if ($envList -match "^$EnvName\s") {
    Write-Host "  既存の $EnvName を environment-local.yml の内容に合わせて更新します。"
    & $conda env update --name $EnvName --file $EnvFile --prune
}
else {
    Write-Host "  $EnvName を新規作成します。"
    & $conda env create --file $EnvFile
}
if ($LASTEXITCODE -ne 0) { throw "conda 環境 $EnvName の作成 / 更新に失敗しました。" }

# --- 3. Jupyter カーネル ---------------------------------------------
Write-Step 'Jupyter カーネルを登録します'
& $conda run --name $EnvName --no-capture-output python -m ipykernel install --user --name $EnvName --display-name "Python ($EnvName)"
if ($LASTEXITCODE -ne 0) { throw 'Jupyter カーネルの登録に失敗しました。' }

# --- 4. Azure CLI ----------------------------------------------------
Write-Step 'Azure CLI を確認します'
try {
    if (Get-Command az -CommandType Application -ErrorAction SilentlyContinue) {
        Write-Host '  Azure CLI は導入済みです。'
        az extension add --name ml --upgrade --only-show-errors
        if ($LASTEXITCODE -eq 0) {
            Write-Host '  ml 拡張を最新にしました。'
        }
        else {
            Write-Host '  警告: ml 拡張の追加に失敗しました（本ハンズオンでは任意の機能です）。' -ForegroundColor Yellow
        }
    }
    elseif (Get-Command winget -CommandType Application -ErrorAction SilentlyContinue) {
        Write-Host '  Azure CLI を導入します（ユーザー アカウント制御の確認が表示されることがあります）。'
        winget install --exact --id Microsoft.AzureCLI --accept-source-agreements --accept-package-agreements
        Write-Host '  導入しました。ml 拡張の追加は、新しいターミナルで次を実行してください。'
        Write-Host '      az extension add --name ml'
    }
    else {
        Write-Host '  Azure CLI と winget が見つかりません。次のページから手動で導入してください。'
        Write-Host '      https://learn.microsoft.com/cli/azure/install-azure-cli-windows'
    }
}
catch {
    Write-Host "  警告: Azure CLI の処理に失敗しました: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host '  ローカル環境の構築は続行します。Azure CLI は次のページから手動で導入してください。'
    Write-Host '      https://learn.microsoft.com/cli/azure/install-azure-cli-windows'
}

# --- 5. 検証 ---------------------------------------------------------
Write-Step '導入結果を検証します'
& $conda run --name $EnvName --no-capture-output python $VerifyScript
if ($LASTEXITCODE -ne 0) { throw '検証に失敗しました。上に表示されたメッセージを確認してください。' }

# --- 6. 次の手順 -----------------------------------------------------
Write-Host ''
Write-Host '=== セットアップが完了しました ===' -ForegroundColor Green
Write-Host @"

次の手順に進んでください。

  1. Azure にサインインする
         az login

  2. Notebook を開く
         VS Code   : notebooks/01_setup_azureml.ipynb を開き、
                     右上のカーネルで「Python ($EnvName)」を選ぶ
         JupyterLab: conda activate $EnvName
                     jupyter lab

  ⚠ Azure のリソースは起動している間ずっと課金されます。
     終わったら docs/09_評価・コスト・後片付け.md の後片付けを必ず実施してください。
"@
