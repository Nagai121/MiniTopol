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
        # A duplicate ledger indicates an interrupted migration. Never hide rows.
        if self.modern.exists() and self.legacy.exists():
            raise RuntimeError("Both ATLAS ledgers exist; run migrate-atlas-sources to reconcile")
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


def _relink_manifests(workspace, directory_pairs):
    """Repair only external ATLAS source paths, without changing A identity.

    An interruption after source relocation can be repaired by rerunning migration.
    Existing converter fingerprints are checked against the moved file before any
    metadata path is changed. A.zarr, descriptor files and DuckDB are untouched.
    """
    from ..storage import atomic_json, read_json
    from .client import _sha256

    root = workspace.path / "trajectories"
    for trajectory_dir in root.iterdir():
        manifest_path = trajectory_dir / "manifest.json"
        if not trajectory_dir.is_dir() or not manifest_path.is_file():
            continue
        with workspace.lock("trajectory-" + trajectory_dir.name):
            manifest = read_json(manifest_path)
            updated = False
            for field, source_kind in (("gro_path", "gro"), ("pdb_path", "pdb"),
                                       ("xtc_path", "xtc")):
                recorded = manifest.get(field)
                if not isinstance(recorded, str):
                    continue
                current = Path(recorded)
                for old_dir, new_dir in directory_pairs:
                    try:
                        suffix = current.relative_to(old_dir)
                    except ValueError:
                        continue
                    destination = new_dir / suffix
                    if destination.is_symlink() or not destination.is_file():
                        raise FileNotFoundError("Relocated trajectory source is missing: " + str(destination))
                    expected = manifest.get("options", {}).get("source", {}).get(source_kind, {})
                    if (not isinstance(expected, dict) or not expected.get("sha256")
                            or type(expected.get("bytes")) is not int):
                        raise ValueError("Cannot validate existing trajectory source fingerprint")
                    if (destination.stat().st_size != expected["bytes"]
                            or _sha256(destination) != expected["sha256"]):
                        raise ValueError("Relocated trajectory source differs from its manifest: "
                                         + str(destination))
                    manifest[field] = str(destination.resolve())
                    updated = True
                    break
            if updated:
                atomic_json(manifest_path, manifest)


def migrate_atlas_sources(workspace):
    """Explicit, resumable, non-overwriting move of tracked ATLAS source files.

    Full migration takes the same atlas-sources lock as download/check/import.
    Crashes can leave a mix of old and new source directories or two ledgers;
    retry reconciles those states, and never drops unrelated/new ledger rows.
    """
    from .client import _sha256, _dataset_id, _kind
    root = workspace.path / "sources" / "ATLAS"
    modern_ledger = root / "downloads.tsv"
    legacy_ledger = workspace.path / "atlas" / "downloads.tsv"
    moved = 0
    with workspace.lock("atlas-sources"):
        with workspace.lock("atlas-ledger"):
            modern_rows = _read_rows(modern_ledger)
            legacy_rows = _read_rows(legacy_ledger)
            by_key = {}
            for row in modern_rows + legacy_rows:
                dataset, kind = row["dataset"], row["kind"]
                if _dataset_id(dataset) != dataset or _kind(kind) != kind:
                    raise ValueError("Invalid ATLAS ledger identifier")
                # Validate EACH row before deduplication: a second ledger must
                # never hide a malicious or unrelated archive path.
                old_path = (workspace.trajectory_path(f"atlas-{dataset}-{kind}")
                            / "source" / f"{dataset}_{kind}.zip")
                new_path = root / f"{dataset}-{kind}" / f"{dataset}_{kind}.zip"
                allowed = {old_path.relative_to(workspace.path).as_posix(),
                           new_path.relative_to(workspace.path).as_posix()}
                if row["archive"] not in allowed:
                    raise ValueError("Unexpected archive location in ATLAS ledger")
                key = dataset, kind
                previous = by_key.get(key)
                if previous is not None:
                    # The only permitted difference after a partial migration
                    # is the archive's location. Never overwrite fresh status,
                    # timestamps, errors or checksums silently.
                    if any(previous[field] != row[field] for field in COLUMNS
                           if field != "archive"):
                        raise ValueError("Conflicting old/new ATLAS ledger rows for " + str(key))
                    if previous["archive"] != row["archive"] and not (
                            previous["archive"].endswith("/" + key[0] + "_" + key[1] + ".zip")
                            and row["archive"].endswith("/" + key[0] + "_" + key[1] + ".zip")):
                        raise ValueError("Conflicting ATLAS archive paths for " + str(key))
                else:
                    by_key[key] = dict(row)

            rewritten = []
            relocations = []
            pending = []
            for row in by_key.values():
                dataset, kind = row["dataset"], row["kind"]
                if _dataset_id(dataset) != dataset or _kind(kind) != kind:
                    raise ValueError("Invalid legacy ATLAS ledger identifier")
                old_dir = workspace.trajectory_path(f"atlas-{dataset}-{kind}") / "source"
                new_dir = root / f"{dataset}-{kind}"
                filename = f"{dataset}_{kind}.zip"
                old_rel = (old_dir / filename).relative_to(workspace.path).as_posix()
                new_rel = (new_dir / filename).relative_to(workspace.path).as_posix()
                if row["archive"] not in (old_rel, new_rel):
                    raise ValueError("Unexpected archive location in ledger: " + row["archive"])
                if old_dir.is_symlink() or new_dir.is_symlink():
                    raise ValueError("ATLAS source directory may not be a symbolic link")
                if old_dir.exists() and new_dir.exists():
                    raise FileExistsError("Old and new ATLAS source directories both exist")
                if (old_dir.exists() and not old_dir.is_dir()) or (
                        new_dir.exists() and not new_dir.is_dir()):
                    raise ValueError("ATLAS source must be a directory")
                location = old_dir if old_dir.exists() else new_dir
                archive = location / filename
                if row["status"] == "complete" and not archive.is_file():
                    raise FileNotFoundError("Completed ATLAS archive missing: " + str(archive))
                if archive.exists() or archive.is_symlink():
                    if archive.is_symlink() or not archive.is_file():
                        raise ValueError("ATLAS archive must be a regular file")
                    if row["bytes"] and archive.stat().st_size != int(row["bytes"]):
                        raise ValueError("ATLAS archive size mismatch")
                    if row["sha256"] and _sha256(archive) != row["sha256"]:
                        raise ValueError("ATLAS archive SHA256 mismatch")
                relocations.append((old_dir, new_dir))
                rewritten.append(dict(row, archive=new_rel))
                if old_dir.is_dir():
                    pending.append((old_dir, new_dir))
            # Validate all ledger rows before mutating disk. After a crash,
            # entries already moved simply skip the rename.
            for source, destination in pending:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError("ATLAS migration destination exists: " + str(destination))
                source.rename(destination)
                moved += 1
            _relink_manifests(workspace, relocations)
            if rewritten or legacy_ledger.exists() or modern_ledger.exists():
                _write_rows(modern_ledger, rewritten)
            # Publish the new ledger first. A crash before unlink leaves two
            # reconcilable ledgers and never deletes the last valid copy.
            if legacy_ledger.exists() and modern_ledger.exists():
                legacy_ledger.unlink()
    return moved
