# NOTICE

本フォルダー（`10.ImitationLearning/src`）のコードは、次の第三者ソフトウェアを利用します。
各ソフトウェアの著作権は、それぞれの権利者に帰属します。

**著作権表示は、実際に導入されたパッケージに同梱されている LICENSE ファイルから転記しました（確認日: 2026-08-17）。**

---

## imitation (MIT License)

```
Copyright (c) 2019-2022 Center for Human-Compatible AI and Google LLC
```

https://github.com/HumanCompatibleAI/imitation/blob/master/LICENSE

## Stable-Baselines3 (The MIT License)

```
Copyright (c) 2019 Antonin Raffin
```

https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE

## Gymnasium (The MIT License)

```
Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation
```

https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE

## seals (MIT License)

```
Copyright (c) 2020 Center for Human-Compatible AI
```

https://github.com/HumanCompatibleAI/seals/blob/master/LICENSE

---

## MIT License の全文

上記 4 パッケージはいずれも MIT License です。全文は次のとおりです。

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 間接的に導入されるパッケージについて

`pip install` により、上記の依存として多数のパッケージが導入されます。
**それらのライセンスは MIT だけではありません**（BSD / Apache-2.0 / LGPL / Proprietary を含みます）。

- 一覧と注意点: [../docs/A2_OSSライセンス一覧.md](../docs/A2_OSSライセンス一覧.md)
- **実際に導入された全パッケージとバージョンは、各ジョブの成果物 `pip_freeze.txt` に記録されます**
  （[il_common.py](il_common.py) の `log_pip_freeze()` が MLflow アーティファクトとして保存します）

社内の OSS ライセンス審査には、この `pip_freeze.txt` を提出してください。
