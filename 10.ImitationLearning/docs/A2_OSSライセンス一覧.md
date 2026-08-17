# A2. OSS ライセンス一覧

[← A1. トラブルシューティング](A1_トラブルシューティング.md) ｜ [A3. 出典一覧 →](A3_出典一覧.md)

---

> **本ページの位置づけ**
> 本ハンズオンが利用する OSS のライセンスを、**実際に導入されたパッケージのメタデータと同梱 LICENSE ファイルを直接読んで**確認した結果です。
> **確認日: 2026-08-18** ／ 確認環境: Windows 10 / Python 3.10 / conda 環境 `il-panda`
>
> ⚠ **免責**: 以下は確認内容の要約であり、**法的助言ではありません。**
> **実際の利用可否・再配布条件は、ご所属の組織の OSS ポリシーおよび法務部門の判断に従ってください。**

> [!IMPORTANT]
> **本章はロボットのピックアンドプレースを扱うため、9 章（強化学習）や本章の旧版（CartPole）にはなかった依存が加わります。**
> 具体的には **panda-gym** と **PyBullet**、そして **PyBullet に同梱される 3D モデル データ**です。
> **3D モデル データはパッケージ本体と別のライセンスです**（§3・§4）。

---

## 1. 本ハンズオンが直接指定するパッケージ

[../src/conda.yaml](../src/conda.yaml)（Azure ML ジョブ用）と
[../setup/environment-local.yml](../setup/environment-local.yml)（ローカル用）で明示的にバージョンを指定しているものです。

| パッケージ | バージョン | ライセンス | 著作権表示（同梱 LICENSE から転記） |
|---|---|---|---|
| **imitation** | 1.0.1 | **MIT** | Center for Human-Compatible AI and Google |
| **Stable-Baselines3** | 2.2.1 | **MIT** | Antonin Raffin |
| **Gymnasium** | 0.29.1 | **MIT** | — |
| **seals** | 0.2.1 | **MIT** | Center for Human-Compatible AI |
| **panda-gym** | 3.0.7 | **MIT** | **Copyright (c) 2020 Quentin Gallouédec** |
| **PyBullet** | 3.2.5 | **zlib**（Trove: zlib/libpng License） | Erwin Coumans, Yunfei Bai, Jasmine Hsu |
| **NumPy** | 1.26.4 | **BSD** | Copyright (c) 2005-2023, NumPy Developers |
| **SciPy** | 1.15.2 | **BSD** | Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers |
| **pandas** | 2.3.3 | **BSD 3-Clause** | Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team<br>Copyright (c) 2011-2023, Open source contributors |
| **matplotlib** | 3.10.9 | **PSF ベースのライセンス**（Trove: Python Software Foundation License） | Copyright (c) 2012- Matplotlib Development Team |
| **MLflow** | 3.15.1 | **Apache-2.0** | Copyright 2018 Databricks, Inc. |
| **azureml-mlflow** | 1.60.0 | ⚠ **Other/Proprietary License** | https://aka.ms/azureml-sdk-license （§2-1） |

ノートブックからローカルで使うもの（環境定義には含めず、手元にだけ導入します）:

| パッケージ | 確認したバージョン | ライセンス | 備考 |
|---|---|---|---|
| **azure-ai-ml** | 1.34.1 | **MIT**（`License-Expression: MIT`） | Azure ML v2 SDK |
| **azure-identity** | 1.25.3 | **MIT** | Copyright (c) Microsoft Corporation. |

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

### zlib ライセンスの要点（PyBullet）

同梱の `LICENSE.txt` の冒頭は次のとおりです（実測）。

> "The files in this repository are licensed under the zlib license, **except for the files under 'Extras' and examples/ThirdPartyLibs**."

| | 内容 |
|---|---|
| **許可** | 商用利用、改変、配布、私的利用 |
| **条件** | **改変版では「オリジナルではない」と明示すること**、ライセンス表示を削除しないこと |
| **制限** | 責任・保証は負わない |
| ⚠ 注意 | **リポジトリ全体が zlib ではありません。** 除外部分と同梱データ（§3）を個別に確認してください |

---

## 2. ⚠ 特に注意が必要な 2 件（パッケージ本体）

### 2-1. `azureml-mlflow` 1.60.0 は OSS ライセンスではありません

本ハンズオンが固定している **1.60.0** のメタデータは次のとおりでした（実測）。

```
Version: 1.60.0
License: https://aka.ms/azureml-sdk-license
Classifier: License :: Other/Proprietary License
```

同梱 `LICENSE.txt` の冒頭も、**Azure プレビューの追加使用条件への同意を条件とする**内容です。

> **これは MIT や Apache-2.0 のような寛容型 OSS ライセンスではありません。**
> Azure Machine Learning に MLflow の記録を送るために本ハンズオンで使用しますが、
> **再配布や Azure 以外での利用を検討する場合は、必ず上記 URL のライセンス条項を確認してください。**

> [!IMPORTANT]
> **⚠ バージョンによってライセンス表記が変わります。**
> 同じ日に確認した **1.62.0.post5** は `License: MIT` ／ `Classifier: License :: OSI Approved :: MIT License` でした。
> **「azureml-mlflow は◯◯ライセンス」と一般化せず、使うバージョンのメタデータを必ず自分で確認してください。**

### 2-2. `pygame` は LGPL（コピーレフト系）です

`gymnasium[classic-control]` の依存として **`pygame` 2.6.1** が導入されます。
Trove 分類は **`License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)`** です。

| 事実 | 内容 |
|---|---|
| なぜ入るのか | `imitation` が `gymnasium[classic-control]` を要求し、その extra が `pygame` を要求するため |
| 本ハンズオンでの用途 | **使用しません。** 本章の描画は PyBullet が行い、本ハンズオンのコードは `pygame` を直接 import しません |
| 確認できたこと | パッケージの dist-info に **LICENSE ファイルが同梱されていません**（メタデータの Trove 分類のみが根拠） |

> ⚠ **LGPL はコピーレフト系のライセンスです。**
> 本ハンズオンのように「pip で導入して動的に利用するだけ」であれば通常は問題になりませんが、
> **成果物を再配布する場合は、ご所属組織の OSS ポリシーに照らして必ず確認してください。**
>
> `gymnasium` を extra 無しで導入すれば `pygame` は不要になる可能性がありますが、
> **`imitation` が extra 付きで要求するため、本ハンズオンでは外せるかを検証していません。**

---

## 3. ⚠ PyBullet に同梱される 3D モデル データは別ライセンスです

**これは本章から新しく加わる、見落としやすい論点です。**

`pybullet_data` フォルダーには、シミュレーション用の URDF モデルとメッシュが同梱されています。
**その一部には、PyBullet 本体（zlib）とは異なるライセンスが個別に添付されています。**

実測した結果、`pybullet_data` 配下のライセンス ファイルは **4 件**でした。

| 場所 | ライセンス | 本ハンズオンでの使用 |
|---|---|---|
| `pybullet_data/franka_panda/LICENSE.txt` | **Apache License 2.0** | ✅ **使用します**（§3-1） |
| `pybullet_data/bicycle/LICENSE.txt` | ⚠ **CC BY-SA 3.0**（Attribution-ShareAlike） | ❌ 使用しません |
| `pybullet_data/domino/license.txt` | 独自表記（**PyBullet への引用を要請**） | ❌ 使用しません |
| `pybullet_data/laikago/license.txt` | 独自表記（メッシュは **Unitree の許諾**に基づく） | ❌ 使用しません |

### 3-1. 本ハンズオンが使うロボット モデル

panda-gym は、ロボット本体として **`franka_panda/panda.urdf`** を読み込みます。

```python
# panda_gym/envs/robots/panda.py（実測）
file_name="franka_panda/panda.urdf",
```

このファイルの探索パスは `pybullet_data.getDataPath()` に設定されているため、
**実際に読み込まれるのは `pybullet_data/franka_panda/panda.urdf`** です。

```python
# panda_gym/pybullet.py（実測）
self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
```

**したがって本ハンズオンが実際に使用するロボット モデルは Apache License 2.0 です。**

> ⚠ **Apache-2.0 は「変更点の明示」を条件に含みます。**
> URDF やメッシュを改変して再配布する場合は、その旨を明示してください。

### 3-2. ⚠ 使っていないデータにも注意が必要な理由

**`pybullet_data` はパッケージ全体が導入されます。**
本ハンズオンで使うのは `franka_panda` と地面のモデルだけですが、
**`pip install pybullet` を行った時点で、CC BY-SA 3.0 のデータも一緒に手元に入ります。**

> ⚠ **CC BY-SA は「継承（ShareAlike）」条件を持つライセンスです。**
> **成果物として環境イメージやコンテナーを再配布する場合、同梱データのライセンスが問題になり得ます。**
> **本ハンズオンでは「導入されるだけで使用していない」ことまでしか確認していません。**
> 再配布の可否については、ご所属組織の OSS ポリシーおよび法務部門の判断に従ってください。

---

## 4. 間接的に導入される主なパッケージ

| パッケージ | 確認したバージョン | ライセンス（メタデータ実測） |
|---|---|---|
| **PyTorch** | 2.13.0 | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| tensorboard | 2.21.0 | Apache 2.0 |
| **pygame** | 2.6.1 | **LGPL**（§2-2 を参照） |

> ⚠ **PyTorch のライセンス表記は単一ではありません。**
> メタデータは **複数ライセンスの AND 結合**として宣言しており、dist-info には
> `licenses/third_party/` 配下に多数の第三者ライセンスが同梱されています。
> **「PyTorch は BSD」と単純化しないでください。**

> **上表は網羅的ではありません。** 完全な一覧は各ジョブが記録する `pip_freeze.txt` を参照してください
> （[../src/il_common.py](../src/il_common.py) の `log_pip_freeze()` が MLflow アーティファクトとして保存します）。

---

## 5. 配布・再利用するときの実務チェックリスト

本テキストや `src/` のコードを社内で再配布する場合の確認項目です。

- [ ] **[../src/NOTICE.md](../src/NOTICE.md) を同梱する**（第三者 OSS の著作権表示）
- [ ] 各パッケージの LICENSE 全文を入手し、社内の指定場所に保管する
- [ ] **`pip_freeze.txt`（MLflow アーティファクト）で実際に導入された全パッケージを洗い出す**
- [ ] 洗い出したパッケージのライセンスを社内ポリシーに照らして確認する
- [ ] **⚠ `azureml-mlflow` の条項を確認する**（§2-1。**バージョンごとに表記が変わります**）
- [ ] **⚠ `pygame`（LGPL）の扱いを確認する**（§2-2）
- [ ] **⚠ `pybullet_data` 同梱データのライセンスを確認する**（§3。**CC BY-SA 3.0 が含まれます**）
- [ ] **⚠ 使用するロボット モデル（`franka_panda`, Apache-2.0）の変更点明示義務を確認する**（§3-1）
- [ ] 法務部門のレビューを受ける

---

## 6. 免責（再掲）

- 本ページの記載は、パッケージのメタデータおよび同梱 LICENSE ファイルの内容の要約であり、**法的助言ではありません。**
- **バージョンによってライセンス表記が変わることを確認しています**（§2-1）。**必ず自分が使うバージョンで確認してください。**
- ライセンスは変更される可能性があります。**必ず配布元で最新の内容を確認してください。**
- **実際の利用可否は、ご所属の組織の OSS ポリシーおよび法務部門の判断に従ってください。**

---

[← A1. トラブルシューティング](A1_トラブルシューティング.md) ｜ [A3. 出典一覧 →](A3_出典一覧.md)
