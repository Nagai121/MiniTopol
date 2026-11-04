# Agent notes — MiniTopol

## Provider stage profiling on Windows

The Provider supports optional per-request stage timing. To enable it, set the
environment variable **before starting the Provider process**. In PowerShell:

```powershell
$env:MINITOPOL_PROVIDER_PROFILE = "1"
python -m minitopol.provider serve
```

By default, the Provider appends one JSON record per completed `get` request
to `provider-profile.txt` in the registered MiniTopol Workspace. No
in-memory history or profile retrieval API is used.

To choose an explicit TXT location, set the path **before starting the Provider**:

```powershell
$env:MINITOPOL_PROVIDER_PROFILE = "1"
$env:MINITOPOL_PROVIDER_PROFILE_PATH = "D:\\MiniTopol\\provider-profile.txt"
python -m minitopol.provider serve
```

The path is an example; the parent directory must be writable. These
PowerShell environment settings apply to the current PowerShell process and
its children. Restart an already-running Provider to apply changed settings.

Each line records total and per-stage Provider processing time plus cache
statistics. The timing measurement finishes before TXT serialization and
file writing. Writing still adds some actual response latency. Profiling is
off by default.

For implementation details and the timing boundaries, see
`docs/provider-reader.md`. The Reader request interface is unchanged.
