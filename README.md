# MiniTopol

GRO＋XTCの教師トラジェクトリを逐次変換し、分子表現と記述子を管理するPythonパッケージです。
基本表現Aは溶媒を除く全重原子で、Zarr v3に原座標・全フレームの時刻・ボックスを保存します。
原子表と通常の記述子結果はParquet、SASAは整数counts、主鎖角は検証付きrecipeで保存します。所在・対応関係の管理と横断参照はDuckDBを使用します。各形式はWorkspace APIとMCPから論理表として取得できます。

**Aの10記述子、要求時のPBC取得、逐次・時間集計、分割保存とDuckDBへの登録を実装しています。** 計算方式は [承認済み定義 A-1](definitions/A-proposal.md) に従います。TPRからの静的な結合表現は独立して追加され、10記述子・A/C座標の定義には影響しません。水素結合・内部幾何の追加計算は対象外です。[検証範囲と性能値の制約](docs/validated-processing.md)、[保存形式とrecipeの環境固定](docs/schema.md)も参照してください。

## インストールと起動

Python 3.11以上を使用します。リポジトリのルートで実行します。

```bash
python -m pip install -e ".[test]"
python -m minitopol init
python -m minitopol convert teacher.gro teacher.xtc --solvent SOL WAT HOH
python -m minitopol list
python -m pytest -q
```

CPUの高速計算には `.[acceleration,lossless]`、独立軌跡の並列実行には
`.[parallel]` を追加します。`lossless` はPython 3.12以上で使用します。
NVIDIA GPUを利用する場合は `.[gpu,lossless,parallel]` を使用します。
対応するNVIDIAドライバーが必要です。CUDAライブラリはインストール済みの
`CUDA_HOME` / `CUDA_PATH`、またはGPU extraのNVIDIAパッケージから検出します。
MiniTopol自体がドライバーの変更やダウンロードを行うことはありません。
GPUを使わない場合は環境変数 `MINITOPOL_SASA_DEVICE=cpu` を指定します。

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

独立した複数入力は `python -m minitopol run jobs.json` またはMCPの
`run_trajectory_pipelines` で処理できます。ジョブごとに別Workspaceを指定し、
変換・明示した注釈・任意のTPR結合復元・記述子計算を順に実行します。
既定は最大6プロセス、各2 nativeスレッド、最大2ジョブでGPUを使用し、
親・子プロセス合計RSSを4GiBに制限します。入力形式と再開方法は
[計算資源と並列実行](docs/calculators.md#独立軌跡の並列実行)を参照してください。

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

## ATLASとMCP

ATLASからのメタデータ取得・ZIPダウンロード・簡易TSV台帳、構成チェッカー、Replica別のA変換、TPR由来の独立した結合表現を利用できます。`atlas-check` / `atlas-import` またはMCP経由で明示的に実行します。既存のダウンロード操作だけでは変換は始まりません。詳細は[ATLAS・MCPの使い方](docs/atlas-mcp.md)を参照してください。

## v0.2.0: 元データと静的な系情報

ATLASの元データは `BaseTraj_Workspace/sources/ATLAS/<dataset>-<kind>/`、
ダウンロード台帳は `sources/ATLAS/downloads.tsv` に保存します。
既存Workspaceは旧配置の読み出しが可能です。検証付きの明示的な移行は
`python -m minitopol migrate-atlas-sources` で実行します。mainの既存データを
自動移動・削除しません。

変換後の軌道ごとに `system_conditions.json` を作成し、取得可能な
シミュレーション条件を保存します。値が不明な項目はnullです。TPRと検証済み
原子対応があれば、元の全系について分子・原子タイプ・部分電荷・分子所属を
`system/molecules.parquet` と `system/atom_properties.parquet` に保存できます。

既存の軌道へ追加情報を記録する例：

```bash
python -m minitopol system-metadata <trajectory-id> --mdp input.mdp
python -m minitopol system-metadata <trajectory-id> --tpr input.tpr --mapping mapping.json
```

水素結合PR #7の機能を前提とした開発ブランチです。
