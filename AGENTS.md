# MiniTopol：エージェント向け案内

## MCP連携

MiniTopolには、エージェントから機能を利用するための**MCPサーバーが実装済み**です。
ローカルで標準入出力（stdio）を使う方式です。リモートHTTPサーバーは起動しません。

追加依存関係のインストールと起動方法：

```powershell
python -m pip install -e ".[mcp]"
minitopol-mcp
```

MCPクライアント側には、`minitopol-mcp` コマンドを起動するよう設定します。
作業対象のワークスペースは、必要に応じて環境変数 `MINITOPOL_WORKSPACE` で指定できます。

MCP経由で利用できる機能には、ATLASのメタデータ取得・ダウンロード、
取得済みデータの確認・取り込み、トラジェクトリ一覧、変換、記述子計算、
結果の参照などがあります。詳細は `docs/atlas-mcp.md` を参照してください。
記述子計算は同期実行です。大規模な結果を一度にMCPへ出力する設計ではありません。

## WindowsでのProvider測定

Providerには、クエリ単位の処理時間・キャッシュ統計をTXTファイルに直接追記する
測定機能があります。**Provider起動前**に、PowerShellで次を実行します。

```powershell
$env:MINITOPOL_PROVIDER_PROFILE = "1"
python -m minitopol.provider serve
```

保存先の初期値は、登録されたMiniTopolワークスペース内の
`provider-profile.txt` です。別の保存先を指定する場合は、
Providerの起動前に次の環境変数も設定します。

```powershell
$env:MINITOPOL_PROVIDER_PROFILE = "1"
$env:MINITOPOL_PROVIDER_PROFILE_PATH = "D:\MiniTopol\provider-profile.txt"
python -m minitopol.provider serve
```

上のDドライブのパスは指定例です。書き込み可能な場所を指定してください。
環境変数の変更は、既に起動しているProviderには反映されないため、再起動が必要です。

各クエリの完了時に、日時・総処理時間・工程別時間・キャッシュ統計などを
**1行1JSON形式のUTF-8テキスト**として追記します。以前の記録は上書きしません。
RAM上に履歴を保持する機能や履歴取得APIはありません。

計測時間はTXTへの書き込み開始前に確定するため、書き込み処理自体は
Provider内部の測定値に含まれません。記録処理による実際の応答遅延は生じます。
測定機能はデフォルトで無効です。Reader側のクエリ方式は変更不要です。

計測対象の工程と制約は `docs/provider-reader.md` を参照してください。
