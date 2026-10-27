# Aの10記述子の実行・出力・再開

科学的な定義は [A-1](../definitions/A-proposal.md) に従います。CalculatorはRegistryから読み出し、元GRO・XTCを直接開きません。
今回の計算対象は教師軌跡のAのみです。Cは空のままで、予測軌跡との比較や化学結合情報は扱いません。

## 実行

ネイティブ計算と座標の可逆圧縮を利用する場合は `pip install '.[mcp,acceleration,lossless]'` でインストールします。
`acceleration` はNumba、`lossless` は任意の一時配列用PCodecを導入します。新規A座標のAGX1圧縮は基本依存に含まれます。PCodecで保存した既存配列の読み出しには、読み出し側にも対応する依存ライブラリが必要です。
SASAの960球面点・原子半径・プローブ半径・遮蔽判定、全フレーム・全ラグの科学的定義は変わりません。主鎖角の中間計算はfloat64で行います。保存座標に由来する精度限界は変わりません。
`memory_mb` は作業配列の予算で、Python・ライブラリ・スレッドを含むプロセス全体のRSS上限ではありません。

```bash
python -m minitopol --workspace /data/MiniTopol convert teacher.gro teacher.xtc --solvent SOL
python -m minitopol --workspace /data/MiniTopol annotate TRAJECTORY_ID annotations.json
python -m minitopol --workspace /data/MiniTopol calculate TRAJECTORY_ID --coordinates nojump --initial-whole --half-box-steps --memory-mb 128
```

`TRAJECTORY_ID` は変換で返されたIDです。`annotations.json` の例は次のとおりです。

```json
{"chains": [{"id": "A", "segments": [[0, 1, 2, 3, 4]]}]}
```

この例はAの `residue_index` が0～4で、同一鎖の連続区間と確認できた場合だけ使用します。
原子・残基の対応は `Registry(workspace).atoms(trajectory_id)` から確認できます。
鎖が複数ある場合はchainの要素を増やし、鎖内の切断・欠損がある場合はsegmentsを分けます。
残基番号の連番や原子間距離から連続性を推測しません。GROのセグメント文字列を確かな鎖IDと読み替えることもしません。

注釈には `alpha_carbons`（Aの点番号の配列）、`masses`（点番号からu単位の質量への辞書）、`radii`（点番号からÅ単位の半径への辞書）も任意で指定できます。
JSON辞書の点番号は文字列です。元GROの番号ではなく、溶媒・水素除去後のAの0起点点番号です。
注釈の既存値は無断更新せず、異なる内容の再登録は拒否します。訂正が必要な入力は明示的に別の軌跡IDで登録します。

座標の扱いは次のいずれかを明示します。

| 指定 | 意味・必要条件 |
|---|---|
| `--coordinates nonperiodic` | 非周期入力として原座標を使う。接触や局所角にも周期距離を適用しない |
| `--coordinates continuous --initial-whole` | 対象Aの意図した像配置にまとめ、時間的に連続化済みの周期座標。周期距離には保存ボックスを使う |
| `--coordinates nojump --initial-whole --half-box-steps` | 初期配置が意図した像にまとまり、保存フレーム間の真の移動が半ボックス未満という前提で、要求時にNoJump復元する |

フラグは入力条件の宣言です。wrapped座標だけから半ボックス以上の移動や、初期配置の誤った像選択を一般に検出・修復できるとは扱いません。
周期距離は直交またはreducedなGROMACS形式の下三角ボックスを対象とします。不正・特異・欠損ボックスは周期計算で拒否します。
原子間の最小像距離と時間方向のNoJumpは別処理です。Converterは保存時にどちらも実行しません。

`--descriptors rmsd rmsf` のように対象を明示できます。省略時は10種類すべてを実行します。
結果レポートには各記述子のcomplete/failed、定義ID、結果ID、行数、未定義値数または失敗理由を含めます。
CLIは1種類でも失敗した場合に終了コード1を返します。成功した他の記述子の結果は保持します。
Python APIは `from minitopol.descriptors import analyze` から利用できます。

## 保存と検索

| 名前 | 主な識別列とvalueの意味 | 単位 |
|---|---|---|
| `rmsd` | frame、time_ps；初期フレームへの全重原子RMSD | Å |
| `rmsf` | point、residue、frames；原子の全時間RMSF | Å |
| `msd` | lag_frames、lag_ps、origins、atoms；内部運動のMSD | Å² |
| `sasa` | level=atom/residue/total、frame、point/residue；各粒度のSASA | Å² |
| `ramachandran` | frame、residue、angle=phi/psi、reason；主鎖角 | degree |
| `gyration` | frame、time_ps；質量重み付き回転半径 | Å |
| `secondary_structure` | frame、residue、reason；helix_like/extended_like/other、未定義はnull | 分類 |
| `contacts` | level=state/frequency、residue_i/j、frame、contact_frames、frames；接触状態1または頻度 | 無次元 |
| `time_correlation` | point、residue、lag_frames、lag_ps、origins、reason；正規化自己相関 | 無次元 |
| `dynamic_correlation` | residue_i/j、frames、reason；正規化相互相関 | 無次元 |

接触のstate行には `distance_angstrom` を併記します。frequency行のframe・time_ps・distance_angstromはnullです。
非接触のstate行、一度も接触しない対象ペアのfrequency行は省略します。対象ペアの定義と全フレーム数は定義・完了記録に保持します。
対象外の近接残基と接触頻度0の対象ペアは、注釈と除外規則から区別します。
φ・ψやゼロ分散で未定義になる値は理由付きnullです。「その他」や数値0へ置き換えません。
二次構造は局所的な角度領域分類であり、水素結合やDSSPの判定を含みません。

```python
table = workspace.query_results(descriptor="rmsf", trajectory=trajectory_id)
table_for_one_definition = workspace.query_results(definition_id=report["results"]["rmsf"]["definition_id"])
```

同名でも入力注釈・PBCの扱い等が異なる結果は定義IDで区別します。
複数種類を一括検索すると、数値と文字列などがDuckDBによって共通型へ変換される場合があります。個々の記述子はそれぞれのParquet型で保存しています。
`query_results` は選択結果をArrow表としてRAMへ返します。非常に大きな出力は対象記述子・定義・列を絞るか、カタログが指すParquetを外部で逐次読み出します。
start/stopの検索条件はフレーム別のframe列に対する条件です。時間差や区間集計の抽出条件には読み替えません。

## 計算資源と再開

`--memory-mb` はMiBを目安とする作業配列の分割予算であり、ライブラリ・メタデータ・アロケータを含むプロセスRSSの上限ではありません。
`--chunk-frames`、`--pair-block`、`--fft-block` でも読み出し・出力・FFTの粒度を制御できます。予算を超える指定は内部の分割上限まで抑えます。
一つのフレームの作業領域すら確保できない予算は拒否します。

位置合わせ座標は専用の `work/<job-key>/aligned.zarr` に逐次保存し、正常終了後にそのジョブ所有の中間データだけを除去します。
このディスク領域は原データと別に必要です。圧縮前の座標規模はおよそ `全フレーム数×A原子数×3×8 bytes` です。
RAMへこの配列全体を展開せず、必要なフレーム・原子範囲を読み出します。原座標やC座標の恒久的な複製には使いません。
中断時は再開のため中間領域を保持します。未知のファイルが混ざる領域は自動削除しません。

RMSFはフレームごとのWelford集計です。MSDと時間相関は時間タイルの自己・相互FFTを合算し、タイルをまたぐ全始点を数えます。
FFTタイルを小さくするとRAMを抑える代わりに時間タイル同士の組合せが増えます。Nフレーム、タイル幅Bでは、単一原子あたり概ねO(N² log B/B)のFFT処理です。
Nが予算内の1タイルに収まる場合は通常のO(N log N)のFFTになります。長時間差の打ち切りや間引きで計算を省略しません。
動的相互相関は残基ペアブロックで集計します。接触は残基ペアブロックと小さな原子ブロックで距離を評価し、密な全ペア行列を保持しません。
接触の全ペア評価や小さなFFTタイルでは計算時間・ディスクI/Oが大きくなり得ます。数百GBの実データでの処理時間は未検証です。

全記述子の定義を計算前にDuckDBへ登録します。出力は `descriptors/<name>/<trajectory>/A/<result-id>/part-*.parquet` に保存します。
パートは一時ファイルから置換し、行数・定義メタデータ・SHA-256を検証します。全パートの完了記録を作った後、単一のDBトランザクションで結果と全パートを登録します。
未完了のパートは検索結果に含めません。同じ定義の再実行は完成済み結果を検証して再利用します。

中断した未完成ジョブは同じ引数・分割設定で再実行します。完成済みのパートを再登録せず、処理途中の単位をやり直して再開します。
位置合わせはコミット済み座標を保持し、NoJumpの状態を再現するため原座標の接頭部分を再走査する場合があります。
MSDは原子ごとに二重バッファで集計をコミットし、途中の加算を再実行しても二重計上しません。
接触頻度は処理済みペアブロックを再利用します。途中のブロックや集計統計の再走査が必要な場合もあります。

```bash
python -m minitopol --workspace /data/MiniTopol recover
```

この操作は全パートが完成し完了記録まで保存された結果を、ハッシュ確認後にDBへ復旧登録します。
途中の計算自体は同じcalculateコマンドで再開します。欠損・破損ファイルを正常扱いして復旧せず、既存のユーザーデータを無断で削除・上書きしません。

## 対応条件

- Cαの不足・共線配置・役割の曖昧さがある場合、位置合わせを必要とする記述子は理由付きで失敗します。
- 鎖の連続性が不明なら該当する主鎖角・二次構造はnullです。接触で必要な近接残基除外を適用できない場合はcontactsが失敗します。
- 未定義の原子半径・質量は、SASAまたは回転半径の失敗として報告します。明示的な半径・質量の注釈で対応できます。
- 不等間隔・重複・逆行時刻はMSD・時間相関の未対応条件です。元のフレーム・時刻や他記述子の計算をその理由で省略しません。
- 真のフレーム間移動が半ボックスを超える場合や初期像が不適切な場合、NoJumpを正しい復元として使えません。結合推定による修復は実装しません。

これらは処理対象の条件として明示した制約です。全10記述子は実装済みですが、条件を満たさない入力から無理に数値を作りません。


### 位置合わせキャッシュの整合性

再開時には、保存済みの座標と時刻を処理ブロックごとのSHA-256で照合します。Zarrの内部チャンクの欠損、値の変更、検証記録の欠損はエラーとし、そのキャッシュを使った未完結果を正式登録しません。検証は一度に1ブロックを読み出し、検証記録の一覧をRAMへ全展開しません。

修正前に作られた検証記録のない途中キャッシュは、安全に再利用できないため停止します。キャッシュや既存の結果を自動削除・上書きしません。再計算する場合は既存Workspaceを保持した別のWorkspaceへ同じ入力を変換して実行できます。完成済みParquetの形式・定義ID・再利用方法は変わりません。

この検証の対象は位置合わせ用の中間キャッシュです。元のA.zarr全体やMSD等の別の中間配列に、同じ完全性検証を適用するものではありません。

## 独立軌跡の並列実行

`minitopol run jobs.json` は独立したWorkspaceで各入力を処理します。
`jobs.json` は次のような配列です。ファイル内の相対パスはJSONの所在を基準にします。
鎖・連続区間・元素・TPR原子対応は確認済みの情報を指定してください。

```json
[
  {
    "workspace": "workspaces/replica1",
    "gro": "data/replica1.gro",
    "xtc": "data/replica1.xtc",
    "solvent_resnames": ["SOL"],
    "trajectory_id": "replica1",
    "analysis": {
      "coordinate_mode": "nojump",
      "initial_whole": true,
      "half_box_steps": true,
      "memory_mb": 128
    }
  }
]
```

`annotation` には `Registry.annotate` の引数（例：`chains`）を渡せます。
`atom_elements` は全入力原子分の明示的な元素配列です。
TPR結合復元には `tpr` と確認済みの `source_to_tpr` 配列を両方指定します。
これらを省略した場合に鎖や結合を推測することはありません。
`analysis` は `analyze` の引数です。条件を満たさない記述子は理由付きで失敗します。

`estimated_work` には既知のA原子数×フレーム数などの正の見積値を指定します。6件以下は降順、7件以上は小さい2件を先に開始してから残りを降順で実行します。同じ重みでは入力順を保ちます。
`--gpu-workers` は同時にGPUを使うSASA stage数を制限します。待機中は同じworkerで独立した整列・主鎖角・二次構造を進め、SASA終了時にcontextを閉じてpermitを返します。
既定は `--workers 6 --native-threads 2 --gpu-workers 2 --rss-limit-mb 4096` です。
`native-threads` は各子プロセスのNumbaスレッド上限です。Contactsのフレーム処理などは
別途、作業メモリ予算内で少数のスレッドを使用します。
GPUや追加ライブラリがない場合、対象範囲・予算を満たさない場合はCPU経路を使います。
実行中のGPUエラーを無制限に再試行することはありません。

各ジョブの作業予算と、Python・ライブラリ・CUDA contextを含むプロセスRSSは別です。
RSS上限超過や呼び出し中断では所有する子プロセスだけを停止し、途中データを保持します。
通常のジョブ失敗では起動済みの独立ジョブを完了させ、新しいジョブを起動しません。
ログと制御情報は各Workspaceの `.pipeline/` 内に保存します。
同じJSONで再実行すると既存の完成結果を検証し、未完了部分を再開します。
同じWorkspaceを複数ジョブや複数runへ同時に割り当てることはできません。

## 保存と実行経路

SASAのGPU経路も960点、原子半径、プローブ半径、厳密な遮蔽判定を維持します。
原子順序・フレーム順序・PBCの定義を変えず、CPU経路へ戻る場合も全対象を計算します。
MSDの新規一時accumulatorは `lossless` extraがある場合PCodecを使い、
各原子の加算順を維持し、RAMに収まる経路では64原子ごとと最終原子で二重バッファを確定します。中断後は最後の確定境界から再開します。大きな系のZarr経路は原子ごとの確定を維持します。既存の一時配列は保存済みのcodecで読みます。
追加ライブラリがない場合はZarr既定の可逆codecを使います。

Contactsはメモリ予算内の順序付きfuturesで距離計算とspool書き込みを重ねます。callerが扱う結果を含む保持枠を制限し、例外・早期終了時も所有スレッドをjoinします。全spoolの整合性検品が済むまで最初の結果を公開しません。

完成結果は`Workspace.query_results`とMCPから取得します。SASAのSCT2とRamachandranのrecipeは通常のParquetではありません。論理表への展開と検品は読み出し側で行います。形式と実行環境固定の条件は[保存スキーマ](schema.md)を参照してください。
途中結果の出力分割方式が異なる版では再開を拒否します。未完了データを元の版で
完了するか、元Workspaceを保持した別のWorkspaceを使用してください。
途中状態のlayoutを書き換えて再開させることはしません。
