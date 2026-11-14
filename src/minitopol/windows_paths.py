"""Windows extended-length path handling for native filesystem operations.

Use this at filesystem boundaries, not to rewrite user-facing logical keys.
Windows Python supports the native \\?\ prefix even without a system-wide
long-path policy, although external software may impose separate path limits.
"""
from __future__ import annotations

import ntpath
import os
from pathlib import Path


def windows_extended_string(value: str) -> str:
    """Return an absolute Win32 extended-length path, handling UNC shares."""
    value = ntpath.abspath(value)
    if value.startswith("\\\\?\\"):
        return value
    if not ntpath.isabs(value):
        raise ValueError("Windows extended paths must be absolute")
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def filesystem_path(value):
    """Upgrade deep Windows paths, preserving normal paths on other systems."""
    path = Path(value)
    if os.name != "nt":
        return path
    text = os.fspath(path)
    if text.startswith("\\\\?\\"):
        return path
    # Leave ordinary paths unchanged. Allow room for common child file names.
    if len(ntpath.abspath(text)) < 220:
        return path
    return Path(windows_extended_string(text))
