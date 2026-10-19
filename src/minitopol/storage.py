"""Atomic publication helpers. Locks are shared by all cooperating processes."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import pyarrow.parquet as pq


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Identifiers must be 1–128 ASCII letters, digits, dots, underscores or hyphens")
    return value


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def fingerprint(path) -> dict:
    path = Path(path).resolve(strict=True)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            hasher.update(block)
    return {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size}


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical(data))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_parquet(path: Path, table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, name, compression="zstd")
        if pq.read_metadata(name).num_rows != len(table):
            raise IOError("Parquet row count verification failed")
        with open(name, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
