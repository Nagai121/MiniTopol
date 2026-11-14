# MiniTopol：エージェントの入口

このファイルは全作業のルートです。**最初に [開発規則](docs/agent-guidelines.md) を確認**し、作業内容に応じて以下の文書を参照してください。ここには個別仕様・実行コマンド・測定手順を重複掲載しません。

| 作業内容 | 参照先 |
|---|---|
| 科学データの構造、Workspace、保存形式 | [docs/schema.md](docs/schema.md) |
| 記述子・計算の定義、科学的前提 | [docs/calculators.md](docs/calculators.md)、[definitions/A-proposal.md](definitions/A-proposal.md) |
| Provider、Reader、共有メモリ、性能計測 | [docs/provider-reader.md](docs/provider-reader.md) |
| 外部・内部補助、多層バッチ、正本・容量管理 | [docs/tiered-storage.md](docs/tiered-storage.md) |
| ATLASの取得、MCPサーバーの利用 | [docs/atlas-mcp.md](docs/atlas-mcp.md) |
| 既存の科学的検証範囲・性能値の解釈 | [docs/validated-processing.md](docs/validated-processing.md) |

**共通原則**：確定仕様を守り、科学データ・正本・既存APIを保護します。保存階層とデータ解釈はProvider側で管理し、Readerは薄い接続クライアントとします。変更後は関連テストで検証し、未検証事項を明示します。詳細は[開発規則](docs/agent-guidelines.md)に集約しています。
