# MiniTopol Storage: 内部補助・外部補助・Provider先読み

## 基本構造

科学データの保存形式やReaderの従来の `request()` 契約は変更しません。
ソースチェックアウト時の既定Workspaceは、リポジトリと同じ親ディレクトリにある
`MiniTopol Storage/` です。wheelの場合はユーザーホームの同名ディレクトリです。
既定保存先は `~/.minitopol/storage-location.json` に初回登録します。
登録済みの保存先がなくなった場合、空の保存領域を作り直さずエラーにします。

内部補助の `MiniTopol Storage` に管理DB `integrated.duckdb` と
`storage_index.sqlite3`（正本・複製の台帳）を置きます。従来の
`trajectories/`、`descriptors/`、`sources/`、`work/`、`locks/` を維持し、
転送用に `staging/` を用意します。外部補助は独立した `MiniTopol Storage` に
`trajectories/`、`descriptors/`、`sources/` を保持します。

外部へ配置できる論理単位は次のとおりです。

- `trajectories/<ID>`：A.zarr、原子表、系情報、マニフェスト等
- `descriptors/<NAME>/<TRAJECTORY_ID>`：該当記述子の全パート・記録
- `sources/<GROUP>/<ID>`：元TPR/XTC等と関連する原データ

配置の正本をSQLite台帳で明示し、転送はSHA-256検証後にのみ公開します。
コピーと検証を終えてから正本の所在を更新します。外部の正本は
キャッシュ解放で削除しません。内部のキャッシュも内容が変化していれば
消さずに保護し、Provider再起動後の初回利用ではハッシュ照合します。
科学計算による座標・記述子・結合・系情報の保存は外部退避と同じ排他ロックを使用し、
外部正本になった論理単位への変更は明示的に拒否します。
外部ディスクは識別子で照合します。

## 初期設定（多層バッチは初期状態で無効）

```bash
minitopol init
minitopol storage status
minitopol storage configure --external /media/external --enable
minitopol storage configure --cache-percent 10 --write-policy longevity --prefetch-batches 1
```

`--external` にドライブのルートを指定した場合、その配下に
`MiniTopol Storage` を明示的に作成します。既存の同名フォルダを指定しても構いません。
外部ディスクが外された場合も、内部にあるデータの通常読み出しは継続できます。
外部側への転送や外部にしかないデータの復元は拒否します。
別ディスクへのすり替わりは識別子の照合で拒否します。内部補助と同一または
親子関係にある保存先は、正本の誤削除を防ぐため指定できません。
**Linuxでは外部保存先が内部補助と別ファイルシステム上にあることを確認**します。
マウントポイントのディレクトリだけが残っている場合は、その場所への作成や
外部正本の退避を拒否します。

内部補助の一時キャッシュ容量は、Provider起動時に得た使用可能容量の初期10%です。
内部補助に正本として配置されたデータはこの枠に含めません。
`--cache-percent` は10%超（最大100%）も指定できます。
`--cache-limit-gib` でGiB単位の絶対容量を指定すると、割合指定ではなく固定容量が
目標になります。どちらかを変更すれば、その方式に切り替わります。相対方式の
起動時容量基準は、設定変更時に再計算せず、Provider再起動時に再評価します。

容量は**軟制限**です。初期状態では、目標容量の10%ぶんを追加の許容量として認めます
（`--cache-slack-percent`）。さらに`--cache-slack-gib`でGiB単位の許容量を
加算できます。いずれも設定可能で、0にするとその許容量は無効です。

- **目標容量**：割合指定または絶対容量で決めた通常の一時配置枠
- **軟制限の許容上限**：目標容量＋指定された超過許容量
- **実際のコピー可否**：その時点の空き容量を別途確認し、コピー時の追加容量を確保
- **許容上限を超える場合**：データ削除で調整できなければ外部正本を直接読み出す

転送のピーク容量や、起動時と利用時の空き容量の差を考慮した仕様です。
容量を超過しても科学データの正本は削除されません。絶対容量指定も物理ディスクの
空き容量を保証するものではありません。コピー用の最低余裕が不足した場合は
外部直接読み出しへ切り替えます。

設定例：

```bash
# 空き容量の40%を目標（初期10%を上回る値）
minitopol storage configure --cache-percent 40

# 内部一時配置の目標を絶対2 GiB、追加許容25%＋0.5 GiBとする
minitopol storage configure --cache-limit-gib 2 --cache-slack-percent 25 --cache-slack-gib 0.5

# 厳密な容量比較を行う実験設定（超過許容なし）
minitopol storage configure --cache-limit-gib 2 --cache-slack-percent 0 --cache-slack-gib 0
```

`storage status` では`temporary_cache_mode`、`temporary_cache_limit_bytes`、
`temporary_cache_soft_limit_bytes`、`available_at_start_bytes`、
`available_now_bytes`を確認できます。

書き込み抑制を重視し、既存の検証済みコピーを再利用して不要な再転送を避けます。
`storage status` では内部補助への累計転送回数・書き込み量
(`internal_copy_operations`、`internal_written_bytes`) も取得できます。
`--write-policy` の初期値は `longevity`（先行バッチ数1）。
`balanced` は2、`throughput` は4を既定の先行バッチ数とし、
`--prefetch-batches` で独立に上書きできます。書き込み削減量を実機で
保証するものではありません。

## データ管理

```bash
# 直接変換したTPR/GRO/PDB + XTCを管理下へコピーし、出典ハッシュを確認してmanifestの参照先を更新
minitopol storage ingest TRAJECTORY_ID

# 必要なデータを外部補助の正本として登録（検証済みコピー後に台帳更新）
minitopol storage offload trajectories/TRAJECTORY_ID
minitopol storage offload descriptors/sasa/TRAJECTORY_ID
minitopol storage offload sources/manual/TRAJECTORY_ID

# 外部の正本は残し、内部に残るキャッシュのみ回収
minitopol storage prune --all

# 明示的に内部補助へ再転送
minitopol storage ensure trajectories/TRAJECTORY_ID
```

`storage ingest` は軌跡を外部補助へ退避する前に実行します。
外部を正本とした後のマニフェスト変更は拒否します。

Providerの通常の `get` は必要な管理対象が外部正本なら
内部補助に準備してから、従来どおりRAMの共有メモリにデータを渡します。
Providerを介さない従来の`Registry`の座標・原子表・マニフェスト読み出しや
`Workspace.query_results()`も、必要に応じて検証済み外部正本を直接参照します。
直接読み出しで内部補助へ全量コピーすることはありません。
科学計算による書き込み先は引き続き内部の物理Workspaceとし、
外部正本への変更は既存の排他・保護規則に従って拒否します。
プリフェッチ・オフロードしただけでは科学データの内容は変更しません。
1つの論理単位がキャッシュ枠を超える場合は、検証済みの外部正本を
Providerが読み取り専用で直接参照します。SSDへの全量コピーは行わず、
内部補助の設定された軟制限を超えた全量転送を避けます。複数の一時データが同時に必要で
上限内に収まらない場合も、外部直接読み出しへ切り替えます。
内部への通常の復元と同じ科学データ形式・Provider応答を使用します。

## MolCasta CLPなどからの多層バッチ

```python
from reader import Reader
client = Reader()
client.configure_storage(enabled=True, prefetch_batches=2)

batches = [
    Reader.batch_request("trajectory-1", data=["coords", "sasa"], frames=[8, 0]),
    Reader.batch_request("trajectory-2", data=["coords"], frames=[1, 4]),
]
plan = client.plan_batches(batches)

with client.request("trajectory-1", data=["coords", "sasa"], frames=[8, 0],
                    plan_id=plan["plan_id"], plan_index=0) as lease:
    # 利用した配列の参照を解除してからLeaseを閉じること
    pass

status = client.batch_status(plan["plan_id"])
```

CLPがバッチの**供給順序**を指定します。Providerはその順序を変えずに
背景の**データ準備**を行い、先行バッチ数だけ外部→内部の転送を予約します。
`batch_status` は転送状態・失敗理由を返し、`cancel_batches` で残りを中止できます。
既存の `next_frames` ヒントも引き続き利用可能です。

## 従来のWorkspaceからの移行

旧 `BaseTraj_Workspace` は自動削除も自動移動もしません。
旧Workspaceのカタログを検出した場合は、新しい空Workspaceを勝手に生成せず
明示的な移行を求めます。Providerを停止してから実行してください。

```bash
minitopol migrate-workspace /path/to/BaseTraj_Workspace \
    --destination "/path/to/MiniTopol Storage"
```

移行は宛先が空の場合にのみ行い、全ファイルを検証したうえで公開し、
旧Workspaceを保持します。既存データへの上書きや、移行失敗時の
原本削除はしません。Providerの登録先を変更した場合は再登録してください。

## 検証上の制限

内部補助への保存単位は引き続き論理単位です。上限超過時は
外部正本を直接読み込むため、Zarrの特定フレームだけを内部補助へ
部分転送する必要はありません。外部直接読み出しではProviderからの
通常のフレーム指定をそのまま使用します。
転送中断時の動作と大規模な数百億AFの実効性能は実機で未検証です。
