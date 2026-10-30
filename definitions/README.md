# 記述子の計算定義

各記述子の科学的な意味・対象表現・単位・参照構造・時間区間・PBC処理・版をここで管理します。
実行時には `DescriptorDefinition` に同じ内容を指定し、結果Parquetと統合カタログの両方へ保存します。

対象はRMSD、RMSF、MSD、SASA、Ramachandran、回転半径、二次構造、接触マップ、時間相関、動的相互相関の10種類です。
化学結合情報、水素結合、内部幾何は今回の実装対象から除外します。
合意済みの基本方式と承認された詳細値・対応条件は [承認済み定義 A-1](A-proposal.md) に整理しています。
10種類のCalculatorはこの定義を使用し、実行前に定義と版をDuckDBへ登録します。実行時の具体的な定義は `descriptors/builtins.py` の `definition_for` が生成し、結果に対応する定義IDで検索できます。
半径・質量の固定表と出典は `src/minitopol/data/atomic-properties.json` にあり、wheelにも同梱します。

C群は空で配布します。新しいCの登録には、明示的な点対応・重み・対象Topologyと記述子定義を指定します。
