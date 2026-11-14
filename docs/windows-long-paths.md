# Windows long-path handling

MiniTopol uses the Win32 extended-length prefix (`\\?\C:\...`, or
`\\?\UNC\server\share\...` for a network share) when a Workspace path is
sufficiently deep, and for long extracted ATLAS input paths. Normal short
paths and all non-Windows paths are unchanged. The logical names stored in
ATLAS `downloads.tsv` and DuckDB are not rewritten to include this prefix.

## Git checkout

Git's Windows checkout path limit is independent of Python's filesystem
handling. If Git reports `Filename too long`, set the option for that clone:

```powershell
git -c core.longpaths=true clone https://github.com/Nagai121/MiniTopol.git
```

For an existing checkout:

```powershell
git config --local core.longpaths true
```

Run the clone command from a reasonably short parent directory if the checkout
itself already failed. A repository file cannot retroactively change Git's
configuration before checkout.

## Python and other programs

Python's standard file operations support extended paths, including
directory creation, archive streaming and atomic file replacement. This
project's ATLAS source-file paths are normalized at filesystem boundaries.
Microsoft also provides the optional system-wide `LongPathsEnabled`
setting (or "Enable Win32 long paths" group policy), which can help other
applications. Changing it may require administrator permissions.

Native programs and scientific libraries can impose their own path limits.
The extended-prefix handling is not a guarantee that *every* third-party
executable will accept any long path. This update does not change Windows
system policy, the user's registry, or Git settings automatically.

References:
- https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
- https://docs.python.org/3/using/windows.html#removing-the-max-path-limitation
