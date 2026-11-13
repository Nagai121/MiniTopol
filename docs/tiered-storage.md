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
消さずに保護します。外部ディスクは識別子で照合します。

## 初期設定（多層バッチは初期状態で無効）

```bash
minitopol init
minitopol storage status
minitopol storage configure --external /media/external --enable
minitopol storage configure --cache-percent 10 --write-policy longevity --prefetch-batches 1
```

`--external` にドライブのルートを指定した場合、その配下に
`MiniTopol Storage` を明示的に作成します。既存の同名フォルダを指定しても構いません。
登録済み外部ディスクが外された場合や、別ディスクに交換された場合は拒否します。

内部補助の一時キャッシュ容量は、Provider起動時に得た使用可能容量の初期10%です。
内部補助に正本として配置されたデータはこの枠に含めません。
`--cache-percent` で割合を変更できます。設定変更時も起動時の容量基準を
維持し、Provider再起動で容量を再評価します。

書き込み抑制を重視し、既存の検証済みコピーを再利用して不要な再転送を避けます。
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

Providerの通常の `get` は必要な管理対象が外部正本なら
内部補助に準備してから、従来どおりRAMの共有メモリにデータを渡します。
プリフェッチ・オフロードしただけでは科学データの内容は変更しません。
1つの論理単位がキャッシュ枠を超える場合は明示的な容量エラーを返し、
自動的に容量制限を破りません。

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

```bash
minitopol migrate-workspace /path/to/BaseTraj_Workspace \
    --destination "/path/to/MiniTopol Storage"
```

移行は宛先が空の場合にのみ行い、全ファイルを検証したうえで公開し、
旧Workspaceを保持します。既存データへの上書きや、移行失敗時の
原本削除はしません。Providerの登録先を変更した場合は再登録してください。

## 検証上の制限

現段階では論理単位の全体転送です。Zarrの特定フレームだけを内部補助へ
部分転送する機能は含まれません。寿命保護の効果、転送中断の動作、
大規模な数百億AFの実効性能については実機測定が必要です。
