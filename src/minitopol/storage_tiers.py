"""Verified internal/external storage for immutable MiniTopol datasets.

The scientific catalog remains in the internal Workspace. Only explicitly
offloaded logical datasets have an external primary; internal copies of those
datasets are expendable and have a separate, startup-fixed space allowance.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import threading
import time
from uuid import uuid4

from filelock import FileLock, Timeout

MIB = 1024 ** 2
EXTERNAL_NAME = "MiniTopol Storage"


def logical_path(value: str) -> str:
    """Accept only managed dataset roots, never arbitrary filesystem paths."""
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("A logical path must use forward slashes")
    path = PurePosixPath(value)
    parts = path.parts
    if (path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/"))
            or not (len(parts) == 2 and parts[0] == "trajectories"
                    or len(parts) == 3 and parts[0] == "descriptors"
                    or len(parts) >= 3 and parts[0] == "sources")):
        raise ValueError("Expected trajectories/ID, descriptors/NAME/ID, or sources/GROUP/ID")
    return path.as_posix()


def _tree_digest(path: Path):
    """SHA-256 over relative names and file contents, also returning byte count."""
    if not path.exists() or path.is_symlink():
        raise FileNotFoundError(f"Storage object missing or unsafe: {path}")
    hasher = hashlib.sha256()
    files = sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.as_posix()) if path.is_dir() else [path]
    total = 0
    for file in files:
        if file.is_symlink():
            raise ValueError(f"Symlinks are not supported in managed datasets: {file}")
        relative = file.relative_to(path).as_posix() if path.is_dir() else ""
        hasher.update(relative.encode("utf-8") + b"\0")
        size = file.stat().st_size
        total += size
        hasher.update(size.to_bytes(8, "big"))
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * MIB), b""):
                hasher.update(chunk)
    if path.is_dir():
        for entry in path.rglob("*"):
            if entry.is_symlink():
                raise ValueError(f"Symlinks are not supported in managed datasets: {entry}")
    return hasher.hexdigest(), total


def _copy_verified(source: Path, destination: Path, expected=None):
    """Never publish a partial copy; preserve existing distinct files."""
    digest, size = _tree_digest(source)
    if expected is not None and digest != expected:
        raise ValueError(f"Source checksum changed: {source}")
    if destination.exists():
        found, _ = _tree_digest(destination)
        if found != digest:
            raise FileExistsError(f"Conflicting existing data: {destination}")
        return digest, size, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (f".{destination.name}.partial-{uuid4().hex}")
    try:
        if source.is_dir():
            shutil.copytree(source, temporary)
        else:
            shutil.copy2(source, temporary)
        verified, actual_size = _tree_digest(temporary)
        if verified != digest or actual_size != size:
            raise IOError(f"Checksum mismatch while copying {source}")
        # FileLock at a higher level protects managed destinations.
        os.replace(temporary, destination)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        elif temporary.exists():
            temporary.unlink()
    return digest, size, True


class TieredStorage:
    """Internal authoritative catalog plus removable external datasets."""

    def __init__(self, workspace):
        self.root = Path(workspace).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Registered MiniTopol Storage is missing: {self.root}")
        self.lock_root = self.root / "locks" / "storage-tiers"
        self.lock_root.mkdir(parents=True, exist_ok=True)
        (self.root / "staging").mkdir(exist_ok=True)
        self.database = self.root / "storage_index.sqlite3"
        self._mutex = threading.RLock()
        # Only a Provider restart takes a fresh free-space snapshot.
        self.available_at_start = shutil.disk_usage(self.root).free
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS tier_settings (
                name TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS tier_objects (
                relative TEXT PRIMARY KEY, primary_tier TEXT NOT NULL,
                sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                last_used REAL NOT NULL)""")
            for name, value in (("enabled", False), ("cache_percent", 10.0),
                                ("write_policy", "longevity"), ("prefetch_batches", 1),
                                ("external_root", None),
                                ("external_identity", None)):
                db.execute("INSERT OR IGNORE INTO tier_settings VALUES (?, ?)",
                           (name, json.dumps(value)))
        config = self.settings()
        if config["external_root"] is not None:
            self._external_root(config)  # Never recreate a missing registered disk.

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.database, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def settings(self):
        with self._db() as db:
            return {name: json.loads(value) for name, value in
                    db.execute("SELECT name, value FROM tier_settings")}

    def _external_root(self, config=None):
        config = config or self.settings()
        value = config.get("external_root")
        if value is None:
            raise ValueError("No external storage is configured")
        root = Path(value)
        marker = root / ".minitopol-storage-identity"
        if not root.is_dir() or not marker.is_file():
            raise FileNotFoundError(f"Registered external MiniTopol Storage is unavailable: {root}")
        if marker.read_text(encoding="utf-8").strip() != config["external_identity"]:
            raise RuntimeError("External storage identity differs; refusing another device")
        return root

    def configure(self, *, enabled=None, external_root=None, cache_percent=None,
                  write_policy=None, prefetch_batches=None):
        with self._mutex:
            current = self.settings()
            changes = {}
            if external_root is not None:
                root = Path(external_root).expanduser().resolve()
                if root.name != EXTERNAL_NAME:
                    root = root / EXTERNAL_NAME
                previous = current["external_root"]
                if previous is not None and str(root) != previous:
                    with self._db() as db:
                        if db.execute("SELECT 1 FROM tier_objects LIMIT 1").fetchone():
                            raise ValueError("Cannot change external device while offloaded objects exist")
                marker = root / ".minitopol-storage-identity"
                if root.exists() and not root.is_dir():
                    raise NotADirectoryError(root)
                if not root.exists():
                    root.mkdir(parents=True)
                if marker.exists():
                    identity = marker.read_text(encoding="utf-8").strip()
                    if not identity:
                        raise ValueError("Empty external storage identity")
                else:
                    if any(root.iterdir()):
                        raise ValueError("Unidentified external storage is not empty")
                    identity = uuid4().hex
                    marker.write_text(identity, encoding="utf-8")
                if previous == str(root) and current["external_identity"] != identity:
                    raise RuntimeError("External storage identity changed")
                changes.update(external_root=str(root), external_identity=identity)
            if enabled is not None:
                if type(enabled) is not bool:
                    raise ValueError("enabled must be boolean")
                changes["enabled"] = enabled
            if cache_percent is not None:
                if (isinstance(cache_percent, bool) or not isinstance(cache_percent, (int, float))
                        or not 0 < cache_percent <= 100):
                    raise ValueError("cache_percent must be in (0, 100]")
                changes["cache_percent"] = float(cache_percent)
            if write_policy is not None:
                if write_policy not in ("longevity", "balanced", "throughput"):
                    raise ValueError("Invalid write_policy")
                changes["write_policy"] = write_policy
            if prefetch_batches is not None:
                if type(prefetch_batches) is not int or not 1 <= prefetch_batches <= 64:
                    raise ValueError("prefetch_batches must be between 1 and 64")
                changes["prefetch_batches"] = prefetch_batches
            merged = {**current, **changes}
            if merged["enabled"] and merged["external_root"] is None:
                raise ValueError("Configure external storage before enabling multi-tier batches")
            if merged["external_root"] is not None:
                self._external_root(merged)
            with self._db() as db:
                for key, value in changes.items():
                    db.execute("INSERT OR REPLACE INTO tier_settings VALUES (?, ?)",
                               (key, json.dumps(value)))
            return self.status()

    def _lock(self, relative):
        key = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        return FileLock(str(self.lock_root / (key + ".lock")))

    def _safe_path(self, root, relative):
        path = Path(root)
        for component in PurePosixPath(relative).parts:
            path = path / component
            if path.is_symlink():
                raise ValueError(f"Symlink in managed storage path: {path}")
        return path

    def _object(self, relative):
        with self._db() as db:
            row = db.execute(
                "SELECT primary_tier, sha256, size_bytes, last_used FROM tier_objects WHERE relative=?",
                (relative,)).fetchone()
        return row

    def _cache_rows(self):
        with self._db() as db:
            rows = db.execute(
                "SELECT relative, size_bytes, last_used FROM tier_objects "
                "WHERE primary_tier='external' ORDER BY last_used ASC").fetchall()
        return [(name, size, last) for name, size, last in rows
                if self._safe_path(self.root, name).exists()]

    def _cap(self):
        return int(self.available_at_start * self.settings()["cache_percent"] / 100.0)

    def status(self):
        config = self.settings()
        rows = self._cache_rows()
        with self._db() as db:
            objects = db.execute("SELECT COUNT(*) FROM tier_objects").fetchone()[0]
        return {**config, "available_at_start_bytes": self.available_at_start,
                "temporary_cache_limit_bytes": self._cap(),
                "temporary_cache_bytes": sum(size for _, size, _ in rows),
                "managed_external_objects": objects}

    def _prune(self, *, required=0, exempt=(), all_cache=False):
        cap = self._cap()
        rows = self._cache_rows()
        used = sum(size for _, size, _ in rows)
        removed = []
        for relative, size, _ in rows:
            if not all_cache and used + required <= cap:
                break
            if relative in exempt:
                continue
            lock = self._lock(relative)
            try:
                with lock.acquire(timeout=0):
                    # Recheck the DB: a concurrent operation may have promoted it.
                    record = self._object(relative)
                    if not record or record[0] != "external":
                        continue
                    path = self._safe_path(self.root, relative)
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
                    used -= size
                    removed.append(relative)
            except Timeout:
                continue  # A Provider request still has the dataset pinned.
        if used + required > cap and not all_cache:
            raise OSError("Temporary storage limit cannot accommodate this batch")
        return removed

    def prune(self, *, all_cache=False):
        with self._mutex:
            removed = self._prune(all_cache=all_cache)
            return {"removed": removed, **self.status()}

    def offload(self, relative):
        relative = logical_path(relative)
        config = self.settings()
        external = self._external_root(config)
        with self._lock(relative):
            internal_path = self._safe_path(self.root, relative)
            external_path = self._safe_path(external, relative)
            if not internal_path.exists():
                raise FileNotFoundError(f"Cannot offload missing internal data: {relative}")
            digest, size, copied = _copy_verified(internal_path, external_path)
            with self._db() as db:
                db.execute("INSERT OR REPLACE INTO tier_objects VALUES (?, 'external', ?, ?, ?)",
                           (relative, digest, size, time.time()))
        # An offload never deletes its source before publishing a verified primary.
        if config["enabled"]:
            self._prune()
        return {"relative": relative, "sha256": digest, "bytes": size,
                "external_copy_created": copied}

    def _ensure_locked(self, relative, exempt=()):
        record = self._object(relative)
        internal = self._safe_path(self.root, relative)
        if not record or record[0] != "external":
            if not internal.exists():
                raise FileNotFoundError(f"Dataset unavailable: {relative}")
            return False
        if internal.exists():
            # Reuse already verified local copies without redundant SSD writes.
            return False
        external = self._safe_path(self._external_root(), relative)
        size = record[2]
        if size > self._cap():
            raise OSError(f"Dataset {relative} exceeds the configured temporary cache limit")
        self._prune(required=size, exempt=tuple(exempt) + (relative,))
        _copy_verified(external, internal, expected=record[1])
        with self._db() as db:
            db.execute("UPDATE tier_objects SET last_used=? WHERE relative=?",
                       (time.time(), relative))
        return True

    def ensure(self, relative):
        relative = logical_path(relative)
        with self._lock(relative):
            return {"relative": relative, "transferred": self._ensure_locked(relative)}

    def request_paths(self, request, workspace):
        if not isinstance(request, dict):
            return []
        tid = request.get("trajectory")
        if not isinstance(tid, str):
            return []
        if "/" in tid or "\\" in tid or tid in ("", ".", ".."):
            return []  # Let the existing Provider request validator reject it.
        names = request.get("descriptors", [])
        if names == "all" or names == ["all"]:
            with workspace.catalog() as db:
                names = [row[0] for row in db.execute(
                    "SELECT DISTINCT descriptor FROM results WHERE trajectory=?", [tid]).fetchall()]
        elif not isinstance(names, list):
            names = []
        paths = ["trajectories/" + tid]
        for name in names:
            if isinstance(name, str) and "/" not in name and "\\" not in name:
                paths.append(f"descriptors/{name}/{tid}")
        return sorted(set(paths))

    @contextmanager
    def hold_request(self, request, workspace):
        paths = [logical_path(p) for p in self.request_paths(request, workspace)]
        with ExitStack() as stack:
            # Lock across the whole Provider read so eviction cannot remove
            # a source file while Zarr/DuckDB is reading it.
            for name in paths:
                stack.enter_context(self._lock(name))
            for name in paths:
                if self._object(name) is not None:
                    self._ensure_locked(name, exempt=paths)
            yield

    def prefetch(self, request, workspace):
        if not self.settings()["enabled"]:
            raise ValueError("Multi-tier batches are disabled")
        targets = self.request_paths(request, workspace)
        return [self.ensure(name) for name in targets if self._object(name) is not None]


    def ingest_sources(self, trajectory):
        """Explicitly adopt converted raw topology/XTC into managed storage.

        Source fingerprints in the immutable trajectory identity are retained.
        Manifest absolute source paths are repointed only after verified copies.
        Originals are never deleted.
        """
        from .storage import atomic_json, fingerprint, read_json, identifier

        identifier(trajectory)
        relative = f"trajectories/{trajectory}"
        self.ensure(relative)
        with self._lock(relative):
            manifest_path = self.root / relative / "manifest.json"
            manifest = read_json(manifest_path)
            source_identity = manifest.get("options", {}).get("source", {})
            inputs = {}
            for kind in ("tpr", "gro", "pdb", "xtc"):
                key = kind + "_path"
                if key not in manifest:
                    continue
                source = Path(manifest[key]).resolve(strict=True)
                expected = source_identity.get(kind)
                if expected != fingerprint(source):
                    raise ValueError(f"Manifest source fingerprint differs: {source}")
                inputs[key] = (kind, source)
            if not inputs or "xtc_path" not in inputs:
                raise ValueError("Converted trajectory has no verifiable raw inputs")
            destination = self.root / "sources" / "manual" / trajectory
            updated = dict(manifest)
            copied = []
            for key, (kind, source) in inputs.items():
                target = destination / (kind + source.suffix.lower())
                if source == target:
                    updated[key] = str(target)
                    continue
                _copy_verified(source, target)
                updated[key] = str(target)
                copied.append(str(target))
            if updated != manifest:
                atomic_json(manifest_path, updated)
            return {"relative": f"sources/manual/{trajectory}",
                    "files": copied, "originals_retained": True}

def migrate_workspace(old, new):
    """Explicitly copy and verify a legacy Workspace; never remove originals."""
    source, target = Path(old).resolve(), Path(new).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Legacy source and new destination must be distinct directories")
    if not (source / "integrated.duckdb").is_file():
        raise FileNotFoundError("Legacy Workspace catalog is missing")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Destination is not empty: {target}")
    digest, size = _tree_digest(source)
    _, _, copied = _copy_verified(source, target, expected=digest)
    return {"source": str(source), "destination": str(target),
            "bytes": size, "sha256": digest, "copied": copied,
            "original_retained": True}
