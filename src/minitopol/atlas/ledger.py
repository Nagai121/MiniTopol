"""Tiny atomic TSV ledger for ATLAS archives; not the DuckDB trajectory catalog."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

COLUMNS = (
    "dataset", "kind", "url", "archive", "downloaded_at",
    "bytes", "sha256", "status", "error",
)


class DownloadLedger:
    def __init__(self, workspace):
        self.workspace = workspace
        self.path = workspace.path / "atlas" / "downloads.tsv"

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames != list(COLUMNS):
                raise ValueError("ATLAS downloads.tsv schema differs from expected columns")
            return list(reader)

    def list(self) -> list[dict]:
        with self.workspace.lock("atlas-ledger"):
            return self._read()

    def get(self, dataset: str, kind: str) -> dict | None:
        with self.workspace.lock("atlas-ledger"):
            for record in self._read():
                if record["dataset"] == dataset and record["kind"] == kind:
                    return record
        return None

    def put(self, record: dict) -> None:
        if set(record) != set(COLUMNS):
            raise ValueError("ATLAS ledger record has unexpected fields")
        with self.workspace.lock("atlas-ledger"):
            rows = self._read()
            key = (record["dataset"], record["kind"])
            rows = [row for row in rows if (row["dataset"], row["kind"]) != key]
            rows.append({field: str(record[field]) for field in COLUMNS})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmpname = tempfile.mkstemp(prefix=".atlas-ledger-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter="\t")
                    writer.writeheader()
                    writer.writerows(rows)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmpname, self.path)
            finally:
                Path(tmpname).unlink(missing_ok=True)
