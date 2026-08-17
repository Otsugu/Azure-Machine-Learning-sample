# A2. OSS ライセンス一覧

[← A1. トラブルシューティング](A1_トラブルシューティング.md) ｜ [A3. 出典一覧 →](A3_出典一覧.md)

---

> **本ページの位置づけ**
> 本ハンズオンが利用する OSS のライセンスを、**実際に導入されたパッケージのメタデータと同梱 LICENSE ファイルを直接読んで**確認した結果です。
> **確認日: 2026-08-17** ／ 確認環境: Windows 11 / Python 3.10.20 / conda 環境 `il-local`
>
> ⚠ **免責**: 以下は確認内容の要約であり、**法的助言ではありません。**
> **実際の利用可否・再配布条件は、ご所属の組織の OSS ポリシーおよび法務部門の判断に従ってください。**

---

## 1. 本ハンズオンが直接指定するパッケージ

| パッケージ | バージョン | ライセンス | 著作権表示（同梱 LICENSE から転記） |
|---|---|---|---|
| **imitation** | 1.0.1 | **MIT** | Copyright (c) 2019-2022 Center for Human-Compatible AI and Google LLC |
| **Stable-Baselines3** | 2.2.1 | **MIT** | Copyright (c) 2019 Antonin Raffin |
| **Gymnasium** | 0.29.1 | **MIT** | Copyright (c) 2016 OpenAI<br>Copyright (c) 2022 Farama Foundation |
| **seals** | 0.2.1 | **MIT** | Copyright (c) 2020 Center for Human-Compatible AI |
| **NumPy** | 2.2.6 | **BSD** | Copyright (c) 2005-2024, NumPy Developers |
| **pandas** | 2.3.3 | **BSD 3-Clause** | — |
| **matplotlib** | 3.10.9 | **PSF ベースのライセンス**（Trove: Python Software Foundation License） | — |
| **MLflow** | 3.15.1 | **Apache-2.0** | Copyright 2018 Databricks, Inc. |
| **azureml-mlflow** | 1.60.0 | ⚠ **Other/Proprietary License** | https://aka.ms/azureml-sdk-license |

### MIT ライセンスの要点

| | 内容 |
|---|---|
| **許可** | 商用利用、改変、配布、私的利用 |
| **条件** | **著作権表示とライセンス表示の保持** |
| **制限** | 責任・保証は負わない |

### Apache License 2.0 の要点

| | 内容 |
|---|---|
| **許可** | 商用利用、改変、配布、**特許利用**、私的利用 |
| **条件** | 著作権表示とライセンス表示の保持、**変更点の明示** |
| **制限** | **商標の使用許諾は含まれない**、責任・保証は負わない |

---

## 2. ⚠ 特に注意が必要な 2 件

### 2-1. `azureml-mlflow` は OSS ライセンスではありません

パッケージのメタデータで **`License :: Other/Proprietary License`** と分類されており、
ライセンス欄には **https://aka.ms/azureml-sdk-license** が示されています。

> **これは MIT や Apache-2.0 のような寛容型 OSS ライセンスではありません。**
> Azure Machine Learning に MLflow の記録を送るために本ハンズオンで使用しますが、
> **再配布や Azure 以外での利用を検討する場合は、必ず上記 URL のライセンス条項を確認してください。**

### 2-2. `pygame` は LGPL（コピーレフト系）です

`gymnasium[classic-control]` の依存として **`pygame` 2.6.1** が導入されます。
Trove 分類は **`License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)`** です。

| 事実 | 内容 |
|---|---|
| なぜ入るのか | `imitation` が `gymnasium[classic-control]` を要求し、その extra が `pygame>=2.1.3` を要求するため |
| 本ハンズオンでの用途 | **CartPole の画面描画のみ**。本ハンズオンのコードは `pygame` を直接 import しません |
| 確認できたこと | パッケージに **LICENSE ファイルが同梱されていません**（メタデータの Trove 分類のみが根拠） |

> ⚠ **LGPL はコピーレフト系のライセンスです。**
> 本ハンズオンのように「pip で導入して動的に利用するだけ」であれば通常は問題になりませんが、
> **成果物を再配布する場合は、ご所属組織の OSS ポリシーに照らして必ず確認してください。**
>
> 描画が不要であれば `pygame` なしで動作するかを検証する余地がありますが、
> **本ハンズオンでは検証していません。**

---

## 3. 間接的に導入される主なパッケージ

`imitation` は次を必須依存として宣言しています（出典: https://pypi.org/pypi/imitation/json ）。

| パッケージ | バージョン | ライセンス（Trove 分類） |
|---|---|---|
| PyTorch | 2.13.0 | メタデータに記載なし（配布元の LICENSE を確認してください） |
| sacred | 0.8.7 | MIT |
| optuna | 4.9.0 | MIT |
| datasets | 5.0.1 | Apache-2.0 |
| huggingface-sb3 | 3.0 | Apache |
| scikit-learn | 1.7.2 | メタデータに記載なし |
| scipy | 1.15.3 | BSD |
| tensorboard | 2.21.0 | Apache-2.0 |
| **pygame** | 2.6.1 | **LGPL**（§2-2 を参照） |

> **上表は網羅的ではありません。** 完全な一覧は各ジョブが記録する `pip_freeze.txt` を参照してください。

---

## 4. 配布・再利用するときの実務チェックリスト

本テキストや `src/` のコードを社内で再配布する場合の確認項目です。

- [ ] **[../src/NOTICE.md](../src/NOTICE.md) を同梱する**（第三者 OSS の著作権表示）
- [ ] 各パッケージの LICENSE 全文を入手し、社内の指定場所に保管する
- [ ] **`pip_freeze.txt`（MLflow アーティファクト）で実際に導入された全パッケージを洗い出す**
- [ ] 洗い出したパッケージのライセンスを社内ポリシーに照らして確認する
- [ ] **⚠ `pygame`（LGPL）の扱いを確認する**（§2-2）
- [ ] **⚠ `azureml-mlflow`（Proprietary）の条項を確認する**（§2-1）
- [ ] 法務部門のレビューを受ける

---

## 5. 免責（再掲）

- 本ページの記載は、パッケージのメタデータおよび同梱 LICENSE ファイルの内容の要約であり、**法的助言ではありません。**
- ライセンスは変更される可能性があります。**必ず配布元で最新の内容を確認してください。**
- **実際の利用可否は、ご所属の組織の OSS ポリシーおよび法務部門の判断に従ってください。**

---

[← A1. トラブルシューティング](A1_トラブルシューティング.md) ｜ [A3. 出典一覧 →](A3_出典一覧.md)
