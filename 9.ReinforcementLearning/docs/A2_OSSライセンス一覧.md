# A2. OSS ライセンス一覧

[← A1. トラブルシューティング総合索引](A1_トラブルシューティング総合索引.md) ｜ [A3. 出典一覧 →](A3_出典一覧.md)

---

> **本ページの位置づけ**
> 本ハンズオンが利用する OSS のライセンスを、**各リポジトリの LICENSE ファイルを直接参照して**確認した結果です。
> **参照日: 2026-08-13**
>
> ⚠ **免責**: 以下は LICENSE ファイルの記載内容の要約であり、**法的助言ではありません。**
> **実際の利用可否・再配布条件は、ご所属の組織の OSS ポリシーおよび法務部門の判断に従ってください。**

---

## 1. 本ハンズオンで実際に使うパッケージ

| パッケージ | 提供元 | ライセンス | 著作権表示 | LICENSE ファイル |
|---|---|---|---|---|
| **Gymnasium** | Farama Foundation | **MIT** | Copyright (c) 2016 OpenAI<br>Copyright (c) 2022 Farama Foundation | https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE |
| **Stable-Baselines3** | DLR-RM | **MIT** | Copyright (c) 2019 Antonin Raffin | https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE |
| **panda-gym** | Quentin Gallouédec | **MIT** | Copyright (c) 2020 Quentin Gallouédec | https://github.com/qgallouedec/panda-gym/blob/master/LICENSE |
| **PyBullet / Bullet Physics** | Bullet Physics | **zlib** | Bullet Continuous Collision Detection and Physics Library | https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt |

### MIT ライセンスの要点

| | 内容 |
|---|---|
| **許可** | 商用利用、改変、配布、私的利用 |
| **条件** | **著作権表示とライセンス表示の保持** |
| **制限** | 責任・保証は負わない |

### zlib ライセンスの要点（PyBullet）

| | 内容 |
|---|---|
| **許可** | **商用利用を含む任意の目的での利用、改変、再配布** |
| **条件** | ① 出所を偽らない ② 改変版はその旨を明示 ③ この告知を削除・改変しない |
| **制限** | 責任・保証は負わない |

> ⚠ **PyBullet（Bullet3）の重要な但し書き**
> LICENSE.txt に次の記載があります。
> 「このリポジトリのファイルは zlib ライセンスの下にあります。**ただし `Extras` 配下と `examples/ThirdPartyLibs` を除きます。**」
>
> **本ハンズオンは PyPI の `pybullet` パッケージを利用するのみで、`Extras` 等のソースを再配布しません。**

---

## 2. 採用しなかった代替案（参考）

**いずれも寛容型ライセンスですが、導入コストが高いため本テキストでは採用していません。**

| パッケージ | 提供元 | ライセンス | LICENSE ファイル | 採用しなかった理由 |
|---|---|---|---|---|
| **Gymnasium-Robotics**<br>（Fetch 系のピックアンドプレイス環境） | Farama Foundation | **MIT**<br>Copyright (c) 2022 Farama Foundation | https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/LICENSE | **MuJoCo が別途必要**で依存が増える |
| **MuJoCo** | Google DeepMind | **Apache-2.0** | https://github.com/google-deepmind/mujoco/blob/main/LICENSE | 上記の依存 |
| **Ray / RLlib** | ray-project | **Apache-2.0** | https://github.com/ray-project/ray/blob/master/LICENSE | 学習用途にはオーバースペック。Azure ML の `RayDistribution` は**実験的クラス** |
| **sb3-contrib**<br>（TQC など追加アルゴリズム） | Stable-Baselines Team | **MIT**<br>Copyright (c) 2020 Stable-Baselines Team | https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/LICENSE | 本テキストは SAC + HER に絞るため |

### Apache License 2.0 の要点

| | 内容 |
|---|---|
| **許可** | 商用利用、改変、配布、**特許利用**、私的利用 |
| **条件** | 著作権表示とライセンス表示の保持、**変更点の明示** |
| **制限** | **商標の使用許諾は含まれない**（ライセンスの対象外。使用したい場合は別途権利者の許諾が必要）、責任・保証は負わない |

---

## 3. Microsoft が提供する OSS（参考）

| リポジトリ | ライセンス | 著作権表示 | 用途 |
|---|---|---|---|
| **Azure/azureml-examples** | **MIT** | Copyright (c) Microsoft Corporation<br>https://github.com/Azure/azureml-examples/blob/main/LICENSE | Azure ML のジョブ記述の一次情報 |
| **microsoft/maro** | **MIT** | Copyright (c) Microsoft Corporation<br>https://github.com/microsoft/maro/blob/master/LICENSE | 参考：産業オペレーション向け多エージェント RL |
| **microsoft/learning-loop** | **MIT** | Copyright (c) Microsoft Corporation<br>https://github.com/microsoft/learning-loop | 参考：オンライン RL ループ。Personalizer の移行先として公式推奨 |

---

## 4. 間接的に導入される依存パッケージ

`pip install` により、上記パッケージの依存として **PyTorch・NumPy・SciPy・cloudpickle** などが導入されます。

> **重要**: **実際に導入されたパッケージとバージョンの完全な一覧は、各ジョブの成果物 `pip_freeze.txt` に記録されます。**
> [src/train_rl.py](../src/train_rl.py) が `pip freeze` の結果を MLflow アーティファクトとして保存します。
>
> **社内の OSS ライセンス審査には、この `pip_freeze.txt` を提出してください。**

---

## 5. 配布・再利用するときの実務チェックリスト

本テキストや `src/` のコードを社内で再配布する場合の確認項目です。

- [ ] **[src/NOTICE.md](../src/NOTICE.md) を同梱する**（第三者 OSS の著作権表示）
- [ ] 各パッケージの LICENSE 全文を入手し、社内の指定場所に保管する
- [ ] `pip_freeze.txt` で**実際に導入された全パッケージ**を洗い出す
- [ ] 洗い出したパッケージのライセンスを社内ポリシーに照らして確認する
- [ ] **コピーレフト系（GPL / AGPL など）のパッケージが混入していないか確認する**
- [ ] 法務部門のレビューを受ける

> ⚠ **本テキストが直接指定するパッケージにコピーレフト系はありません**（MIT / zlib のみ）。
> ただし**間接依存には別のライセンスが含まれる可能性があります。** 必ず `pip_freeze.txt` から確認してください。

---

## 6. 免責（再掲）

- 本ページの記載は、各リポジトリの LICENSE ファイルの内容の要約であり、**法的助言ではありません。**
- ライセンスは変更される可能性があります。**必ず記載の URL で最新の内容を確認してください。**
- **実際の利用可否は、ご所属の組織の OSS ポリシーおよび法務部門の判断に従ってください。**

---

[← A1. トラブルシューティング総合索引](A1_トラブルシューティング総合索引.md) ｜ [A3. 出典一覧 →](A3_出典一覧.md)
