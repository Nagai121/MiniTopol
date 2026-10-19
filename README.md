# MiniTopol

GRO＋XTCの教師トラジェクトリを逐次変換し、分子表現と記述子を管理するPythonパッケージです。
基本表現Aは溶媒を除く全重原子で、Zarr v3に原座標・全フレームの時刻・ボックスを保存します。
原子表と記述子結果はParquet、所在・対応関係の管理と横断参照はDuckDBを使用します。

**実装状況：保存・変換・Registry・結果管理の基盤を実装中です。12種類の組み込み記述子は、合意済み一覧と計算定義の引き継ぎ待ちで未実装です。PBC補正と結合取得も未確定のため未実装です。全機能の完成版ではありません。**

## インストールと起動

Python 3.11以上を使用します。リポジトリのルートで実行します。

```bash
python -m pip install -e ".[test]"
python -m minitopol init
python -m minitopol convert teacher.gro teacher.xtc --solvent SOL WAT HOH
python -m minitopol list
python -m pytest -q
```

`--solvent` は実際の入力に含まれる溶媒残基名を指定します。溶媒がない場合も `--solvent` 自体は付け、名前を省略します。
GROの原子名からの元素推定が適さない場合は、全入力原子分の元素配列をJSONで用意し、`--elements elements.json` で指定します。

初回利用時に `BaseTraj_Workspace/` を自動生成します。生成データはGitに含めません。
別の保存先を使う場合は `python -m minitopol --workspace /data/MiniTopol init` のように指定できます。
環境変数 `MINITOPOL_WORKSPACE` でも指定でき、既存Workspace内のデータは更新時に保持します。

## Pythonからの利用

```python
from minitopol import Workspace, Registry, convert

workspace = Workspace("BaseTraj_Workspace")
trajectory_id = convert(workspace, "teacher.gro", "teacher.xtc",
                        solvent_resnames=["SOL"])
registry = Registry(workspace)

for block in registry.blocks(trajectory_id):
    print(block.start, block.stop, block.positions.shape)
```

初期状態ではAのみです。Cの登録と全軌跡の生成は分離しています。
`Registry.register` に対象軌跡・変換規則・版・記述子定義を渡して登録し、必要な範囲だけ `registry.blocks(trajectory_id, "C1")` から生成します。
現実装はAの点番号と明示的な重みによる変換を提供し、C座標を重複保存しません。

## ファイル構成

| 場所 | 用途 |
|---|---|
| `src/minitopol/` | Converter・Registry・Workspace・記述子の定義／計算／結果管理 |
| `definitions/` | 科学的な計算定義 |
| `tests/` | 実形式の変換・中断復旧・競合・結果管理の検証 |
| `docs/` | データ構造と処理契約 |
| `BaseTraj_Workspace/` | 利用時に生成するデータ。配布物に含めない |

仕様、再開方法、実装済みの範囲と制約は [docs/schema.md](docs/schema.md) に記載しています。
このリポジトリには実教師データを含めていません。テストは一時ディレクトリ内でGRO・XTCを生成します。
数百GB規模の実教師データでの長時間運用は未検証です。
