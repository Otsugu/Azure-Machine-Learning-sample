# 08. Sweep で改善する

[← 07. GAIL と評価の落とし穴](07_GAILと評価の落とし穴.md) ｜ [09. 評価・コスト・後片付け →](09_評価・コスト・後片付け.md)

---

> **この章の目的**
> Azure ML の **Sweep Job** でハイパーパラメーターを自動探索し、
> **「何を最大化させるか」の設計が結果を決める**ことを理解します。
> 対応するノートブック: [notebooks/06_sweep_compare_cleanup.ipynb](../notebooks/06_sweep_compare_cleanup.ipynb) の 1〜2

> [!WARNING]
> **本章の Azure ジョブは Azure 上で実行検証していません。**
> 探索の結果として得られる数値は記載していません。**あなたの環境で確認してください。**

---

## 8.1 Sweep Job とは

**「この引数をこの候補の中で振って、何本も実行して」**と Azure ML に丸投げする仕組みです。

| 手作業（[05 章](05_BCを動かす.md)でやったこと） | Sweep Job |
|---|---|
| 条件を自分で列挙してジョブを 1 本ずつ投入する | **探索空間を宣言するだけ** |
| 結果を自分で集計する | **主要メトリックで自動的に順位付けされる** |
| 上限を自分で管理する | `max_total_trials` / `timeout` で上限を宣言できる |

> 出典（Microsoft 公式）: [モデルのハイパーパラメーター調整 (v2)](https://learn.microsoft.com/azure/machine-learning/how-to-tune-hyperparameters?view=azureml-api-2)

書き方は「**通常の Command Job を作り、振りたい引数を `inputs` として宣言し、`.sweep()` を呼ぶ**」だけです。

```python
sweep_job = trial_job(
    epochs=Choice(values=[37, 150, 600]),
    batch_size=Choice(values=[16, 32, 64]),
).sweep(...)
```

コマンド側では **`${{inputs.epochs}}`** のように参照します。

---

## 8.2 主要メトリックの設計

**`primary_metric` は「Sweep が良し悪しを判断する物差し」です。ここを間違えると、探索そのものが無意味になります。**

### ⚠ 検証損失を使ってはいけない

BC の検証損失は **「専門家の行動をどれだけ正確に当てられたか」** です。
しかし私たちが知りたいのは **「ロボットが実際に物体を運べるか」** です。**この 2 つは一致しません。**

> "we derive a series of lessons including ... **the variability based on the stopping criteria due to the different objectives in training and evaluation.**"
>
> 出典（参考情報・学術論文）: Mandlekar et al.,
> *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*, arXiv:2108.03298（CoRL 2021 Oral）
> https://arxiv.org/abs/2108.03298

**これはロボット操作の実証研究から得られた知見であり、まさに本章の題材そのものです。**

### だから環境で測った値を使います

[src/train_il.py](../src/train_il.py) は、学習後の方策を環境で 20 エピソード動かして
**`eval_success_rate`（成功率）** と **`normalized_return`（正規化リターン）** の両方を記録します。

| 主要メトリックの候補 | 長所 | 短所 |
|---|---|---|
| **`normalized_return`**（本章の既定） | 「ランダム 0 / 専門家 1」で解釈しやすい。**成功の速さも反映される** | 成功率そのものではない |
| `eval_success_rate` | **最も直感的**。ロボットの合否そのもの | **0.05 刻み**（20 エピソード評価）なので同点が出やすい |

> ⚠ **`primary_metric` は、学習スクリプトが記録するメトリック名と完全一致していなければなりません。**

> ⚠ **比較する全試行が同じ `scores.json`（同じデータ資産のバージョン）を使っていることが前提です。**
> 基準値が違うと、`normalized_return` どうしを比べられません（[05 章 5.8](05_BCを動かす.md)）。

---

## 8.3 探索空間の設計

[05 章](05_BCを動かす.md)で得た教訓は **「効いたのはデモ数ではなく勾配更新回数だった」** でした。
そこで探索するのも **学習量を決める 2 つの値**にします。

| 引数 | 意味 |
|---|---|
| `--epochs` | データを何周するか |
| `--batch-size` | 1 回の更新に使う件数 |

$$
\text{勾配更新の回数} \;\approx\; \frac{\text{デモの総ステップ数}}{\text{batch\_size}} \times \text{epochs}
$$

つまり **この 2 つを振ることは、実質的に「更新回数」を振ること**です。

### サンプリング方法

| 方法 | いつ使うか |
|---|---|
| **`grid`** | 候補が離散で少ないとき。**全組み合わせを試す**（本章はこれ） |
| `random` | 候補が多い／連続値のとき |
| `bayesian` | 試行回数を抑えて良い解を探したいとき |

> ⚠ **`max_total_trials` と `timeout` を必ず設定してください。**
> 指定しないと探索空間の全組み合わせが実行され、**費用の上限が無くなります。**

> ⚠ **本章のジョブは [05 章](05_BCを動かす.md) より重くなります。**
> ロボットの物理シミュレーションが入るため、**1 試行あたりの時間が CartPole のような題材とは桁違いです。**
> ローカル実測では、**約 46,000 回の勾配更新に 100〜175 秒**かかりました（[09 章 9.3](09_評価・コスト・後片付け.md)）。

---

## 8.4 ⚠ 早期終了ポリシーは、このままでは働きません

見込みの薄い試行を途中で打ち切る **早期終了ポリシー**（`BanditPolicy` など）があります。
しかし **本章の構成では機能しません。** 理由は仕様にあります。

> **早期終了ポリシーは、学習スクリプトが主要メトリックをログに記録するたびに「1 区間」と数えます。**
>
> 出典（Microsoft 公式）: [モデルのハイパーパラメーター調整 (v2) — 早期終了](https://learn.microsoft.com/azure/machine-learning/how-to-tune-hyperparameters?view=azureml-api-2#early-termination)

`normalized_return` は **学習の最後に 1 回だけ**記録されるため、区間が 1 つしかなく、打ち切る余地がありません。

### 効かせたい場合

[src/train_il.py](../src/train_il.py) は、BC の**各エポック終了時**に
**`epoch_eval_return_mean` と `epoch_eval_success_rate` を step 付きで**記録しています。

```python
def on_epoch_end() -> None:
    nonlocal epoch
    epoch += 1
    mean, _, success = evaluate(bc_trainer.policy, venv, N_EVAL_EPISODES)
    mlflow.log_metric("epoch_eval_return_mean", mean, step=epoch)
    mlflow.log_metric("epoch_eval_success_rate", success, step=epoch)
```

**主要メトリックをこちらに変えれば、`BanditPolicy` が区間ごとに判断できるようになります。**
書き方は [notebooks/06_sweep_compare_cleanup.ipynb](../notebooks/06_sweep_compare_cleanup.ipynb) に記載しています。

> ⚠ **`delay_evaluation` を必ず入れてください。** 学習の序盤は成績が安定しません。

> ⚠ **エポックごとの評価は無料ではありません。**
> 1 回の評価で **20 エピソード × 50 ステップ = 1,000 ステップ**の物理シミュレーションが走ります。
> **エポック数が多い設定では、評価のほうが学習より重くなることがあります。**

---

## 8.5 ⚠ デモ収集を Sweep の中に入れてはいけない

**Sweep の各試行は、毎回ゼロからやり直します。**

| | デモ収集の回数 | 問題 |
|---|---|---|
| Sweep の中に入れる | **試行数だけ** | 時間の無駄。**しかも試行ごとに基準値が変わり、比較できない** |
| **Sweep の外に出す（本ハンズオン）** | **1 回** | 全試行が同じ `scores.json` を使う |

**費用の問題だけではありません。** 試行ごとに `random_mean` / `expert_mean` が変われば、
**`normalized_return` の物差しそのものが試行ごとに変わってしまいます**（[05 章 5.8](05_BCを動かす.md)）。

> **だから [04 章 4.7](04_専門家デモを作る.md) で `uri_folder` のデータ資産として登録しました。**

---

## 8.6 結果の読み方

[notebooks/06_sweep_compare_cleanup.ipynb](../notebooks/06_sweep_compare_cleanup.ipynb) の **2.** で確認します。

| 見るもの | どこで |
|---|---|
| 試行ごとの `normalized_return` / `eval_success_rate` | ノートブックの表 |
| **どの引数が効いているか** | **studio の［平行座標プロット］** |
| 学習量と成績の関係 | ノートブックのグラフ |

> [!IMPORTANT]
> **本章の Sweep は `seed` を固定しています。**
> [05 章 5.6](05_BCを動かす.md) で見たとおり、**この題材ではシードを変えるだけで成功率が大きく動きます。**
>
> **上位の試行が僅差なら、「差は確認できなかった」が正しい結論です。**
> どうしても優劣を判定したい場合は、**上位の設定だけを 3 シードで再実行**してください。

---

## ⚠ うまくいかないときは

| 症状 | 対処 |
|---|---|
| Sweep が始まらない／ずっと `Queued` | `max_concurrent_trials` がクラスターの `max_instances` や vCPU クォータを超えています |
| 「primary metric が見つからない」旨のエラー | メトリック名の綴り違いです。`train_il.py` の `mlflow.log_metrics` と完全一致させてください |
| 一部の試行だけ `Failed` | studio で該当する子ジョブの `user_logs/std_log.txt` を読む |
| 費用が想定より大きい | `max_total_trials` / `timeout` / `max_concurrent_trials` を見直す（[09 章](09_評価・コスト・後片付け.md)） |

---

## この章のまとめ

- Sweep Job は **「探索空間を宣言するだけ」**。上限（`max_total_trials` / `timeout`）は必ず設定する
- **主要メトリックは検証損失ではなく、環境で測った `normalized_return` / 成功率**
- **早期終了は「主要メトリックが記録された回数」で区間を数える。** 最後に 1 回だけ記録する値では働かない
- **デモ収集は Sweep の外に出す。** 費用だけでなく、**物差しを固定するため**
- **1 位の設定だけで判断しない。** 僅差はシードのばらつきに埋もれている

---

[← 07. GAIL と評価の落とし穴](07_GAILと評価の落とし穴.md) ｜ [09. 評価・コスト・後片付け →](09_評価・コスト・後片付け.md)
