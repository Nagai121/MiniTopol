# MiniTopol：エージェント向けメモ

## MCP
MCPサーバーを実装済みです。導入は `pip install -e ".[mcp]"`、起動は `minitopol-mcp`。
ATLAS取得・変換・記述子計算・結果参照などに対応。詳細：`docs/atlas-mcp.md`。

## Provider測定（Windows PowerShell）
Provider起動前に設定します。

```powershell
$env:MINITOPOL_PROVIDER_PROFILE="1"
$env:MINITOPOL_PROVIDER_PROFILE_PATH="D:\MiniTopol\provider-profile.txt" # 任意
python -m minitopol.provider serve
```

測定結果はTXTに1クエリ1行ずつ追記。保存先の指定を省略するとワークスペース内の `provider-profile.txt` に保存します。設定変更時はProviderを再起動。詳細：`docs/provider-reader.md`。
