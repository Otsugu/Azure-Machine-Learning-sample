# NOTICE

本フォルダー（`10.ImitationLearning/src`）のコードは、次の第三者ソフトウェアを利用します。
各ソフトウェアの著作権は、それぞれの権利者に帰属します。

**著作権表示は、実際に導入されたパッケージに同梱されている LICENSE ファイルから転記しました（確認日: 2026-08-18）。**

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

## panda-gym (MIT License)

```
Copyright (c) 2020 Quentin Gallouédec
```

https://github.com/qgallouedec/panda-gym/blob/master/LICENSE

---

## PyBullet / Bullet Physics (zlib License)

ロボットの物理シミュレーションに使います。**MIT ではなく zlib ライセンス**です。

同梱の `LICENSE.txt` から転記（確認日: 2026-08-18 / pybullet 3.2.5）:

```
The files in this repository are licensed under the zlib license, except for the files under
'Extras' and examples/ThirdPartyLibs.

Bullet Continuous Collision Detection and Physics Library
http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the
   original software. If you use this software in a product, an acknowledgment in the product
   documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being
   the original software.
3. This notice may not be removed or altered from any source distribution.
```

著作権者の一覧は同梱の `AUTHORS.txt` にあります。先頭には次のとおり記載されています。

```
Bullet Physics is created by Erwin Coumans with contributions from the following authors /
copyright holders: AMD, Apple, Yunfei Bai, Steve Baker, Gino van den Bergen, ... (以下略)
```

> [!WARNING]
> **`pybullet` には、zlib 以外のライセンスのデータファイルが同梱されています。**
> `pybullet_data` 配下のライセンス ファイルは 4 件でした（実測で確認）。
>
> | 場所 | ライセンス | 本ハンズオンでの使用 |
> |---|---|---|
> | `pybullet_data/franka_panda/LICENSE.txt` | **Apache License 2.0** | ✅ **使用します**（下記） |
> | `pybullet_data/bicycle/LICENSE.txt` | **CC BY-SA 3.0** | ❌ 使用しません |
> | `pybullet_data/domino/license.txt` | 独自表記（PyBullet への引用を要請） | ❌ 使用しません |
> | `pybullet_data/laikago/license.txt` | 独自表記（メッシュは Unitree の許諾） | ❌ 使用しません |
>
> **使用しないデータも `pip install pybullet` で一緒に手元へ入ります。**
> 再配布する場合は個別に確認してください。

https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt

---

## Franka Emika Panda ロボット モデル (Apache License 2.0)

本ハンズオンが動かすロボットの 3D モデルです。**PyBullet 本体（zlib）とは別のライセンスです。**

panda-gym は次のファイルを読み込みます（実測）。

```python
# panda_gym/envs/robots/panda.py
file_name="franka_panda/panda.urdf",

# panda_gym/pybullet.py
self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
```

したがって実体は **`pybullet_data/franka_panda/panda.urdf`** であり、
同じフォルダーの `LICENSE.txt` は **Apache License 2.0** です（確認日: 2026-08-18 / pybullet 3.2.5）。

> [!IMPORTANT]
> **Apache-2.0 は「変更点の明示」を条件に含みます。**
> URDF やメッシュを改変して再配布する場合は、その旨を明示してください。
> **本ハンズオンはこれらのファイルを改変していません。**

https://github.com/bulletphysics/bullet3/blob/master/data/franka_panda/LICENSE.txt

---

## MIT License の全文

上記のうち **imitation / Stable-Baselines3 / Gymnasium / seals / panda-gym** の 5 つは MIT License です。全文は次のとおりです。

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
