# MiniTopol

GRO＋XTCの教師トラジェクトリを逐次変換し、分子表現と記述子を管理するPythonパッケージです。
基本表現Aは溶媒を除く全重原子で、Zarr v3に原座標・全フレームの時刻・ボックスを保存します。
原子表と記述子結果はParquet、所在・対応関係の管理と横断参照はDuckDBを使用します。

**Aの10記述子、要求時のPBC取得、逐次・時間集計、分割ParquetとDuckDBへの登録を実装しています。** 計算方式は [承認済み定義 A-1](definitions/A-proposal.md) に従います。化学結合情報・水素結合・内部幾何は対象外です。

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

初回利用時にプロジェクト内の `BaseTraj_Workspace/` を自動生成します。起動ディレクトリを変えても保存先は変わりません。生成データはGitに含めません。
ソースディレクトリを持たないwheelインストールでは、ユーザーホームの `MiniTopol/BaseTraj_Workspace/` が既定先です。
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

10記述子を計算するには、鎖・連続区間の確実な情報を登録し、座標の扱いを明示します。GROの並びから鎖の連続性は推測しません。

```python
from minitopol.descriptors import analyze

# 例：Aのresidue_indexが0～4で、同じ鎖の連続する5残基だと確認済みの場合
registry.annotate(trajectory_id, chains=[{"id": "A", "segments": [[0, 1, 2, 3, 4]]}])

# 初期配置が意図した像にまとまり、保存フレーム間の移動が半ボックス未満の場合
report = analyze(workspace, trajectory_id, coordinate_mode="nojump",
                 initial_whole=True, half_box_steps=True, memory_mb=128)
print(report["status"], report["results"])
```

非周期入力は `coordinate_mode="nonperiodic"`、既に連続化した周期入力は `coordinate_mode="continuous", initial_whole=True` を明示します。
入力条件を満たさない記述子は理由付きで失敗を報告し、正常な結果として登録しません。未定義の角度・ゼロ分散は理由付きnullで保存します。
CLI、注釈JSON、出力列、中断再開と計算資源については [docs/calculators.md](docs/calculators.md) を参照してください。

初期状態ではAのみです。Cの登録と全軌跡の生成は分離しています。
`Registry.register` に対象軌跡・変換規則・版・記述子定義を渡して登録し、必要な範囲だけ `registry.blocks(trajectory_id, "C1")` から生成します。
現実装はAの点番号と明示的な重みによる変換を提供し、C座標を重複保存しません。

## ファイル構成

| 場所 | 用途 |
|---|---|
| `src/minitopol/` | Converter・Registry・Workspace・記述子の定義／計算／結果管理 |
| `definitions/` | 科学的な計算定義 |
| `tests/` | 実形式の変換・10記述子の数学的検証・中断復旧・競合・結果管理 |
| `docs/` | データ構造と処理契約 |
| `BaseTraj_Workspace/` | 利用時に生成するデータ。配布物に含めない |

仕様、再開方法、実装済みの範囲と制約は [docs/schema.md](docs/schema.md) に記載しています。
このリポジトリには実教師データを含めていません。テストは一時ディレクトリ内でGRO・XTCを生成します。
数百GB規模の実教師データでの長時間運用は未検証です。
