# 02. Azure 環境の準備

[← 01. 強化学習の基礎](01_強化学習の基礎.md) ｜ [次へ: 03. Azure ML 環境構築 →](03_AzureML環境構築.md)

---

> **この章の目的**
> Azure Machine Learning のワークスペースを作る**前**に必要な準備を済ませることです。
> **最も事故が起きやすいのがここ**（権限不足・クォータ不足）なので、**必ず先に済ませてください。**

---

## 2.1 必要な権限を確認する

### なぜ必要か

Azure ML ワークスペースを作ると、**同時に複数の Azure リソースが自動作成されます**（ストレージ アカウント、Key Vault、Application Insights、Container Registry）。
そのため、**リソースを作成できる権限**が必要です。

> **出典: Microsoft Learn**
> [Manage Azure Machine Learning workspaces](https://learn.microsoft.com/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2)
> 「既存のリソース グループを使うには *共同作成者（Contributor）* または *所有者（Owner）* ロールが必要です」と記載されています。

### 手順

1. [Azure Portal](https://portal.azure.com/) にサインインする
2. 使用するサブスクリプションを開く
3. 左メニューの **［アクセス制御 (IAM)］** → **［自分のアクセス権の表示］** を選択
4. **共同作成者** または **所有者** が付与されているか確認する

### 期待される結果

- 自分のアカウントに **共同作成者** 以上のロールが表示される

> **重要**: 権限が無い場合、**この先の手順はすべて失敗します。** 事前に IT 管理者へ依頼してください。
> 依頼する内容: 「演習用リソースグループに対する **共同作成者** ロール」

---

## 2.2 リージョンを決める

### なぜ必要か

Azure のリージョン（データセンターの場所）によって、**使える VM の種類・クォータ・価格が異なります。**

### 選定の考え方

| 観点 | 説明 |
|---|---|
| **クォータ** | **最も重要。** 使いたい VM シリーズの vCPU 枠が空いているリージョンを選ぶ |
| レイテンシ | 利用者・データに近いほうがよい（日本なら Japan East など） |
| 価格 | リージョンによって単価が異なる。[Azure ML 価格](https://azure.microsoft.com/pricing/details/machine-learning/) で確認 |
| VM の可用性 | [リージョン別の提供状況](https://azure.microsoft.com/global-infrastructure/services/?products=virtual-machines) で確認 |

> ⚠ **注意**: コンピューティング クラスターをワークスペースと**別のリージョン**に作ると、**ネットワーク遅延とデータ転送コストが増える可能性があります。**
> → 出典: [Create an Azure Machine Learning compute cluster - Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/how-to-create-attach-compute-cluster?view=azureml-api-2)
>
> **原則、ワークスペースとコンピューティングは同じリージョンにしてください。**

---

## 2.3 【最重要】クォータ（vCPU 上限）を確認する

### なぜ必要か

Azure には **サブスクリプション・リージョン・VM ファミリごとに vCPU の上限（クォータ）** があります。
**クォータが足りないと、コンピューティング クラスターを作成できても、ジョブが永久に `Queued` のまま進みません。**

> **重要な仕様**（出典: [Create an Azure Machine Learning compute instance - Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/how-to-create-compute-instance?view=azureml-api-2)）
> - コンピューティング **インスタンス**と **クラスター**のクォータは**共通**です
> - **コンピューティング インスタンスを停止してもクォータは解放されません**（再起動を保証するため）
> - **クォータは「上限」であって「空き容量の保証」ではありません。** リージョンが逼迫していると、クォータ内でも起動できないことがあります

### 本ハンズオンで必要なクォータの目安

| 用途 | VM サイズの例 | vCPU | 台数 | 合計 vCPU |
|---|---|---|---|---|
| 学習ジョブ（並列4本） | `Standard_DS3_v2`（4 vCPU） | 4 | 2〜4 | **8〜16** |

> **推奨: 少なくとも 8 vCPU、演習を快適に進めるなら 16 vCPU 以上**の空きを確保してください。

### 手順（Azure Portal で確認）

1. [Azure Portal](https://portal.azure.com/) → 検索バーで **「クォータ」** を検索
2. **［クォータ］** → **［コンピューティング］** を選択
3. **サブスクリプション** と **リージョン** を絞り込む
4. 使用予定の VM ファミリ（例: `Standard DSv2 Family vCPUs`）の **使用量 / 上限** を確認

### 手順（Azure ML studio で確認）

1. [Azure Machine Learning studio](https://ml.azure.com) を開く
2. 左メニューの **［クォータ］** を選択
3. サブスクリプションとリージョンを選ぶ

### クォータが足りない場合

**［クォータ］画面から増加申請を行います。**（承認まで時間がかかるため、**必ず事前に**実施してください）

> **出典: Microsoft Learn**
> [Manage and increase quotas and limits for resources with Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2)

---

## 2.4 ローカル開発環境を準備する

### 選択肢は2つあります

| 方式 | メリット | デメリット | 本テキストの推奨 |
|---|---|---|---|
| **A. Azure ML コンピューティング インスタンス** | **Python も SDK も導入済み。参加者間の環境差異がゼロ** | インスタンス代がかかる | **✅ 推奨** |
| B. 手元の PC | 追加費用ゼロ | Python / CLI の導入が必要。**当日のトラブル要因になりやすい** | 自習向き |

> **重要**: **本ハンズオンでは方式 A（コンピューティング インスタンス）を強く推奨します。**
> 「pip が通らない」「Python のバージョンが違う」といったトラブルで時間を溶かさないためです。
> コンピューティング インスタンスの作成手順は [03 章](03_AzureML環境構築.md) の 3.4 で扱います。

### 方式 B（手元の PC）を選ぶ場合の手順

```powershell
# 1) Python の確認（3.9 以上を推奨）
python --version

# 2) 仮想環境を作る
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Azure ML SDK v2 と関連パッケージを導入
pip install azure-ai-ml azure-identity mlflow azureml-mlflow

# 4) Azure CLI と ml 拡張（任意。CLI から操作したい場合）
az extension add --name ml
```

> **出典: Microsoft Learn**
> - Azure ML Python SDK v2: https://aka.ms/sdk-v2-install

> [!IMPORTANT]
> **方式 B で「RL 環境をローカルで動かす」場合の注意（Windows ユーザー必読）**
>
> **Windows でも panda-gym は動きます。ただし `pip` だけでは入りません。**
>
> `panda-gym` が依存する `pybullet` は、**PyPI に Windows 向けのビルド済みホイールを公開していません**
> （最新版 3.2.7 の配布物は `manylinux` 向けホイールとソース配布 `.tar.gz` のみ）。
> そのため Windows で `pip install panda-gym` を実行するとソースビルドに入り、次のエラーで失敗します。
>
> ```
> error: Microsoft Visual C++ 14.0 or greater is required.
> ```
>
> 一方 **conda-forge は `win-64` 向けのビルド済み `pybullet` を配布しています**。
> こちらを使えば **C++ コンパイラを入れずに** Windows へ導入できます。
> **具体的な手順は [notebooks/01_setup_azureml.ipynb](../notebooks/01_setup_azureml.ipynb) の「1-2. （Windows のみ）ローカルで RL 環境を動かす場合の準備」にまとめてあります。**
>
> **対処の選択肢**
>
> | 選択肢 | 内容 |
> |---|---|
> | **✅ conda-forge から入れる（Windows で GUI を見たいなら推奨）** | `pybullet` をビルド済みパッケージで導入 → `panda-gym` は `pip` で追加。コンパイラ不要 |
> | **✅ 方式 A に切り替える（最も確実）** | コンピューティング インスタンスは Linux なのでこの問題が起きません。ただし **GUI は使えません** |
> | Build Tools を導入して `pip` でビルドする | [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) を入れる（容量が大きく管理者権限が必要） |
> | WSL / Linux / macOS を使う | Linux には `pybullet` のビルド済みホイール（`cp310` 向け）があります |
> | **ローカルでは RL 環境を動かさない** | ジョブ投入と MLflow 参照だけなら SDK のみで十分です |
>
> ⚠ **この制約はローカル実行だけの話です。** Azure ML のコンピューティングは Linux なので、
> [src/conda.yaml](../src/conda.yaml) の `python=3.10` のもとで `pybullet` は PyPI のホイールから導入されます。
>
> **出典（参考情報・サードパーティ）**
> - PyPI の配布ファイル一覧: https://pypi.org/project/pybullet/#files
> - conda-forge の配布パッケージ一覧（`win-64` を含む）: https://anaconda.org/conda-forge/pybullet
> - Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
> - CLI (v2) のセットアップ: [Install and set up the CLI (v2)](https://learn.microsoft.com/azure/machine-learning/how-to-configure-cli?view=azureml-api-2)
> - MLflow の前提: [Log metrics, parameters, and files with MLflow](https://learn.microsoft.com/azure/machine-learning/how-to-log-view-metrics?view=azureml-api-2)（`mlflow` と `azureml-mlflow` が必要と明記）

### 実行場所は二段構えです（**ローカル → Azure ML Compute**）

本テキストは、**同じ学習スクリプト [src/train_rl.py](../src/train_rl.py) を 2 か所で実行できる**ように設計しています。

| 段階 | 実行場所 | 目的 | MLflow の記録先 |
|---|---|---|---|
| **段階 1** | **手元の PC / Mac** | 環境を「目で見て」理解する。コードのバグを潰す。**GUI シミュレーターが使えるのはここだけ**（→ [04 章](04_RL環境を触って理解する.md) 4.6） | **追跡 URI を明示設定する必要がある** |
| **段階 2** | **Azure ML Compute（クラスター）** | 長時間学習・並列実験・Sweep | **自動設定（設定作業不要）** |

> **出典: Microsoft Learn** — [Azure Machine Learning 用に MLflow を構成する](https://learn.microsoft.com/azure/machine-learning/how-to-use-mlflow-configure-tracking?view=azureml-api-2)
> - 「**Azure コンピューティング インフラストラクチャを使用する場合、追跡 URI を構成する必要はありません。自動的に設定されるようになっています。** 自動構成を持つ環境には、Azure Machine Learning ノートブック、Azure Machine Learning コンピューティング インスタンスでホストされている Jupyter Notebook、**Azure Machine Learning コンピューティング クラスターで実行されるジョブ**が含まれます」
> - 「**ただし、Azure Machine Learning の外部で作業する場合は、ワークスペースを指すように MLflow を構成する必要があります。影響を受ける環境には、ローカル コンピューター**、Azure Synapse Analytics、Azure Databricks が含まれます」

> **重要**: この追跡 URI は、**ワークスペースを作った後でないと取得できません。**
> 実際の取得・設定手順は [03 章](03_AzureML環境構築.md) の 3.6 で扱います。
> ここでは「**方式 B にはこの一手間が要る**」ことだけ覚えてください。**方式 A（コンピューティング インスタンス）ならこの設定は不要です。**

> ⚠ **ローカル実行は「下見」と位置づけてください。**
> ローカル実行は、ジョブと違って**コードのスナップショットと環境バージョンがワークスペースに残りません**。
> **比較や報告に使う実験は、必ずジョブ（段階 2）として実行してください。**

---

## 2.5 認証（サインイン）

### なぜ必要か

SDK から Azure にアクセスするには**認証情報**が必要です。本テキストでは `DefaultAzureCredential` を使います。

`DefaultAzureCredential` は、**複数の認証方法を上から順に自動で試す**仕組みです。
（環境変数 → マネージド ID → Azure CLI のログイン情報 …）

> **出典: Microsoft Learn**
> [DefaultAzureCredential クラス](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential)

### 手順

```powershell
# Azure CLI でサインイン（ブラウザーが開きます）
az login

# 複数サブスクリプションがある場合は、使うものを既定に設定
az account set --subscription "<サブスクリプション名 または ID>"

# 確認
az account show
```

### 期待される結果

- `az account show` で、演習に使うサブスクリプションの情報が表示される

---

## 2.6 命名規則とタグを決める（後のコスト集計に効きます）

### なぜ必要か

後から **「このハンズオンで結局いくらかかったのか」** を集計するには、**タグ**を付けておくのが最も簡単です。

> **出典: Microsoft Learn**
> [Manage Azure Machine Learning workspaces（タグ）](https://learn.microsoft.com/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2)
> 「タグは名前と値のペアで、リソースを分類し、**同じタグを複数のリソースやリソース グループに適用して請求を統合して表示**するために使います」と記載されています。

### 推奨する命名規則（例）

| リソース | 命名例 |
|---|---|
| リソース グループ | `rg-rl-workshop-<yourname>` |
| ワークスペース | `mlw-rl-workshop-<yourname>` |
| コンピューティング クラスター | `cpu-cluster` |
| 環境（Environment） | `rl-panda-gym-env` |
| 実験（Experiment） | `rl-baseline` / `rl-reward-exp` / `rl-hparam-sweep` |
| ジョブ（表示名） | `baseline_pickandplace_sac-her_seed0_v1` |

### 推奨するタグ

| タグ キー | 値の例 | 用途 |
|---|---|---|
| `project` | `rl-workshop` | **コスト集計の主キー** |
| `owner` | `<your-alias>` | 担当者 |
| `purpose` | `training` / `evaluation` | 用途の区別 |
| `delete-after` | `2026-08-20` | **後片付けの期限**（消し忘れ防止） |

> **重要**: `delete-after` タグは**課金事故を防ぐ最も安価な保険**です。必ず付けてください。

---

## ⚠ うまくいかないときは（Step 02）

| # | 症状 | 主な原因 | 確認手順 | 対処 |
|---|---|---|---|---|
| 1 | `az login` でブラウザーが開かない／ハングする | ヘッドレス環境、またはブラウザー未設定 | ターミナルの出力を確認 | `az login --use-device-code` を実行し、表示されたコードを別端末のブラウザーで入力する |
| 2 | `az account show` に**別のサブスクリプション**が表示される | 既定のサブスクリプションが違う | `az account list --output table` で一覧を確認 | `az account set --subscription "<名前 or ID>"` で切り替える |
| 3 | サブスクリプションが一覧に出ない | 別のテナントにサインインしている | `az account list --all --output table` | `az login --tenant <テナントID>` でテナントを指定して再サインイン |
| 4 | `az extension add --name ml` が失敗する | 古い Azure CLI | `az version` でバージョン確認 | Azure CLI を最新に更新してから再実行（`az upgrade`） |
| 5 | ワークスペース作成で **`AuthorizationFailed`** | **権限不足**（2.1 参照） | Portal の［アクセス制御 (IAM)］→［自分のアクセス権の表示］ | IT 管理者に**共同作成者**ロールを依頼する |
| 6 | クラスターは作れたのに**ジョブが `Queued` から進まない** | **クォータ不足、またはリージョンの容量不足**（2.3 参照） | studio の［クォータ］、および Portal の［クォータ］→［コンピューティング］ | クォータ増加を申請する。急ぐ場合は**より小さい VM サイズ**か**別リージョン**を検討 |
| 7 | `QuotaExceeded` / `Not enough quota` エラー | 同上 | 同上 | 同上。`max_instances` を減らして再作成する |
| 8 | `pip install azure-ai-ml` が失敗する | Python が古い／プロキシ環境 | `python --version`、`pip config list` | Python を更新する。企業プロキシ環境では `pip install --proxy=<proxy>` を使う |
| 9 | どのリージョンを選べばよいか分からない | 判断材料不足 | 2.2 の表を確認 | **クォータに空きがあるリージョンを最優先**。迷ったら `japaneast` で始め、クォータ不足なら別リージョンへ |

**参考（出典）**

- [クォータと上限の管理・増加申請 - Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2)
- [ワークスペースの管理 - Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2)
- [CLI (v2) のセットアップ - Microsoft Learn](https://learn.microsoft.com/azure/machine-learning/how-to-configure-cli?view=azureml-api-2)

---

## ✅ この章のチェックリスト（Azure 実験環境確認票 その1）

成果物「Azure 実験環境確認票」に対応します。**すべて ✅ になってから次章へ進んでください。**

- [ ] Azure Portal にサインインできた
- [ ] 使用するサブスクリプションを特定した
- [ ] **共同作成者（または所有者）ロール**があることを確認した
- [ ] 使用するリージョンを決めた
- [ ] **そのリージョンの vCPU クォータに 8 以上の空きがある**ことを確認した
- [ ] （不足していた場合）クォータ増加を申請した
- [ ] `az login` に成功し、`az account show` で正しいサブスクリプションが表示された
- [ ] リソース命名規則とタグ（`project` / `owner` / `delete-after`）を決めた
- [ ] **実行場所の二段構え**（段階1 = 手元の PC ／ 段階2 = Azure ML Compute）を理解した
- [ ] 方式 B（手元の PC）を選んだ場合、**MLflow の追跡 URI 設定が別途必要**であることを把握した（[03 章](03_AzureML環境構築.md) 3.6）

---

[← 01. 強化学習の基礎](01_強化学習の基礎.md) ｜ [次へ: 03. Azure ML 環境構築 →](03_AzureML環境構築.md)
