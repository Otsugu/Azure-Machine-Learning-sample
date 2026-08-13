# NOTICE — 第三者 OSS のライセンス表記

本ディレクトリ（`src/`）のコードは、以下のオープンソース ソフトウェアを利用します。
**すべて商用利用が可能な寛容型ライセンス**です。

各ライセンスの全文は、記載の URL（各リポジトリの LICENSE ファイル）を参照してください。

---

## 直接利用するパッケージ

| パッケージ | 著作権表示 | ライセンス | ライセンス全文 |
|---|---|---|---|
| **Gymnasium** | Copyright (c) 2016 OpenAI<br>Copyright (c) 2022 Farama Foundation | **MIT** | https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE |
| **Stable-Baselines3** | Copyright (c) 2019 Antonin Raffin | **MIT** | https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE |
| **panda-gym** | Copyright (c) 2020 Quentin Gallouédec | **MIT** | https://github.com/qgallouedec/panda-gym/blob/master/LICENSE |
| **PyBullet / Bullet Physics** | Bullet Continuous Collision Detection and Physics Library | **zlib** | https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt |

> ⚠ **PyBullet（Bullet3）の注意点**
> LICENSE.txt に次の但し書きがあります。
> 「このリポジトリのファイルは zlib ライセンスの下にあります。**ただし `Extras` 配下と `examples/ThirdPartyLibs` を除きます**」
> 本ハンズオンは PyPI の `pybullet` パッケージを利用するのみで、`Extras` 等のソースを再配布しません。

---

## 間接的に導入されるもの

`pip install` により、上記パッケージの依存として PyTorch・NumPy・SciPy などが導入されます。
**実際に導入されたパッケージとバージョンの完全な一覧は、各ジョブの成果物 `pip_freeze.txt` に記録されます。**
（`src/train_rl.py` が `pip freeze` の結果を MLflow アーティファクトとして保存します。）

---

## Microsoft 提供のドキュメント／サンプル

本ハンズオンは Azure Machine Learning の公式ドキュメント（Microsoft Learn）を出典として参照しています。
また、Azure ML のジョブ記述の一次情報として次のリポジトリを参照できます。

| リポジトリ | 著作権表示 | ライセンス |
|---|---|---|
| **Azure/azureml-examples** | Copyright (c) Microsoft Corporation | **MIT** ( https://github.com/Azure/azureml-examples/blob/main/LICENSE ) |

---

## 免責

本ファイルの記載は、各リポジトリの LICENSE ファイルの内容を要約したものであり、**法的助言ではありません。**
**実際の利用可否・再配布条件は、ご所属の組織の OSS ポリシーおよび法務部門の判断に従ってください。**

参照日: 2026-08-13
