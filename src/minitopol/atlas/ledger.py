"""ATLAS download ledger: nine original fields; no second provenance database."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

COLUMNS = ("dataset", "kind", "url", "archive", "downloaded_at",
           "bytes", "sha256", "status", "error")


def _read_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("ATLAS downloads.tsv schema differs from expected columns")
        return list(reader)


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".atlas-ledger-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class DownloadLedger:
    def __init__(self, workspace):
        self.workspace = workspace
        self.modern = workspace.path / "sources" / "ATLAS" / "downloads.tsv"
        self.legacy = workspace.path / "atlas" / "downloads.tsv"
        # Existing Workspaces remain readable until an explicit migration.
        self.path = self.modern if self.modern.exists() or not self.legacy.exists() else self.legacy

    def _read(self):
        return _read_rows(self.path)

    def list(self):
        with self.workspace.lock("atlas-ledger"):
            return self._read()

    def get(self, dataset, kind):
        with self.workspace.lock("atlas-ledger"):
            return next((row for row in self._read()
                         if row["dataset"] == dataset and row["kind"] == kind), None)

    def put(self, record):
        if set(record) != set(COLUMNS):
            raise ValueError("ATLAS ledger record has unexpected fields")
        with self.workspace.lock("atlas-ledger"):
            rows = [row for row in self._read()
                    if (row["dataset"], row["kind"]) != (record["dataset"], record["kind"])]
            rows.append({key: str(record[key]) for key in COLUMNS})
            _write_rows(self.path, rows)


def migrate_atlas_sources(workspace):
    """Explicit, resumable migration; never overwrite existing source files.

    Existing A/descriptor folders and the DuckDB catalog are not changed.
    """
    from .client import _sha256
    modern_ledger = workspace.path / "sources" / "ATLAS" / "downloads.tsv"
    legacy_ledger = workspace.path / "atlas" / "downloads.tsv"
    moved = 0
    with workspace.lock("atlas-migrate-sources"):
        with workspace.lock("atlas-ledger"):
            ledger = DownloadLedger(workspace)
            if modern_ledger.exists() and legacy_ledger.exists():
                raise FileExistsError("Both old and new ATLAS ledgers exist; reconcile manually")
            rows = ledger._read()
            updated = []
            preflight = []
            for row in rows:
                dataset, kind = row["dataset"], row["kind"]
                # Validate identifiers before constructing filesystem paths.
                from .client import _dataset_id, _kind
                if _dataset_id(dataset) != dataset or _kind(kind) != kind:
                    raise ValueError("Invalid legacy ATLAS ledger identifier")
                old_dir = workspace.trajectory_path(f"atlas-{dataset}-{kind}") / "source"
                new_dir = workspace.path / "sources" / "ATLAS" / f"{dataset}-{kind}"
                old_rel = (old_dir / f"{dataset}_{kind}.zip").relative_to(workspace.path).as_posix()
                new_rel = (new_dir / f"{dataset}_{kind}.zip").relative_to(workspace.path).as_posix()
                if row["archive"] not in (old_rel, new_rel):
                    raise ValueError("Unexpected archive location in ledger: " + row["archive"])
                if old_dir.is_symlink() or new_dir.is_symlink():
                    raise ValueError("ATLAS source directory may not be a symbolic link")
                if old_dir.exists() and new_dir.exists():
                    raise FileExistsError("Old and new ATLAS source directories both exist")
                current = old_dir if old_dir.exists() else new_dir
                archive = current / f"{dataset}_{kind}.zip"
                if row["status"] == "complete" and not archive.is_file():
                    raise FileNotFoundError("Completed ATLAS archive missing: " + str(archive))
                if archive.exists():
                    if archive.is_symlink() or not archive.is_file():
                        raise ValueError("ATLAS archive must be a regular file")
                    if row["bytes"] and archive.stat().st_size != int(row["bytes"]):
                        raise ValueError("ATLAS archive size mismatch")
                    if row["sha256"] and _sha256(archive) != row["sha256"]:
                        raise ValueError("ATLAS archive SHA256 mismatch")
                newrow = dict(row, archive=new_rel)
                updated.append(newrow)
                if old_dir.is_dir():
                    preflight.append((old_dir, new_dir))
            for src, dst in preflight:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() or dst.is_symlink():
                    raise FileExistsError("ATLAS migration destination exists: " + str(dst))
                src.rename(dst)
                moved += 1
            if rows or ledger.path == legacy_ledger:
                _write_rows(modern_ledger, updated)
            if legacy_ledger.exists() and modern_ledger.exists():
                legacy_ledger.unlink()
    return moved
