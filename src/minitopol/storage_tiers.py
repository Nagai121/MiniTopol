"""Verified internal/external storage for immutable MiniTopol datasets.

The scientific catalog remains in the internal Workspace. Only explicitly
offloaded logical datasets have an external primary; internal copies of those
datasets are expendable and have a separate, startup-fixed space allowance.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import sys
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
                    or len(parts) >= 3 and parts[0] == "sources")
            or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", part) is None
                   for part in parts[1:])):
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
        self.registry_marker = self.root / ".minitopol-tier-registry"
        if self.registry_marker.exists() and not self.database.is_file():
            raise FileNotFoundError("Storage placement catalog is missing; refusing a new empty catalog")
        self._mutex = threading.RLock()
        # Protect the registration transition across processes and CLI/Provider.
        # An offload must never publish on a drive that configure() just changed.
        self._state_lock = FileLock(str(self.lock_root / "storage-configuration.lock"))
        # A capacity decision and its copy must be atomic across Provider
        # processes. Per-object locks alone cannot prevent two distinct
        # concurrent requests from independently consuming the same headroom.
        self._capacity_lock = FileLock(str(self.lock_root / "capacity-admission.lock"))
        self._touch_times = {}
        # Cache verification is scoped to this process. Never trust an existing
        # internal copy on a new Provider startup without checking its digest.
        self._validated_internal = {}
        self._validated_external = {}
        # Only a Provider restart takes a fresh free-space snapshot.
        self.available_at_start = shutil.disk_usage(self.root).free
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS tier_settings (
                name TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS tier_objects (
                relative TEXT PRIMARY KEY, primary_tier TEXT NOT NULL,
                sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                last_used REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS tier_metrics (
                name TEXT PRIMARY KEY, value INTEGER NOT NULL)""")
            for name, value in (("enabled", False), ("cache_percent", 10.0),
                                ("cache_limit_bytes", None), ("cache_slack_percent", 10.0),
                                ("cache_slack_bytes", 0),
                                ("write_policy", "longevity"), ("prefetch_batches", 1),
                                ("external_root", None),
                                ("external_identity", None)):
                db.execute("INSERT OR IGNORE INTO tier_settings VALUES (?, ?)",
                           (name, json.dumps(value)))
        if not self.registry_marker.exists():
            self.registry_marker.write_text("schema=1\\n", encoding="utf-8")
        # An unplugged external archive must not prevent internal-only reads.
        # Explicit operations touching the external tier still fail closed.

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

    def _require_external_device(self, path):
        """Refuse an unmounted Linux drive path on the internal filesystem.

        A directory and a marker on the internal disk are not evidence of
        a mounted removable drive. A different st_dev is required on Linux;
        mount-point detection by name alone is insufficient (nested mounts).
        """
        if sys.platform.startswith("linux"):
            if os.stat(path).st_dev == os.stat(self.root).st_dev:
                raise RuntimeError(
                    f"External MiniTopol Storage is not on a separately mounted "
                    f"filesystem: {path}")

    def _external_root(self, config=None):
        config = config or self.settings()
        value = config.get("external_root")
        if value is None:
            raise ValueError("No external storage is configured")
        root = Path(value)
        marker = root / ".minitopol-storage-identity"
        if root == self.root or root in self.root.parents or self.root in root.parents:
            raise ValueError("External storage overlaps internal MiniTopol Storage")
        if not root.is_dir() or not marker.is_file():
            raise FileNotFoundError(f"Registered external MiniTopol Storage is unavailable: {root}")
        self._require_external_device(root)
        if marker.read_text(encoding="utf-8").strip() != config["external_identity"]:
            raise RuntimeError("External storage identity differs; refusing another device")
        return root

    def configure(self, *, enabled=None, external_root=None, cache_percent=None,
                  cache_limit_bytes=None, cache_slack_percent=None, cache_slack_bytes=None,
                  write_policy=None, prefetch_batches=None):
        with self._state_lock, self._mutex:
            current = self.settings()
            changes = {}
            if external_root is not None:
                root = Path(external_root).expanduser().resolve()
                if root == self.root or root in self.root.parents or self.root in root.parents:
                    raise ValueError(
                        "External storage must not overlap the internal MiniTopol Storage")
                if root.name != EXTERNAL_NAME:
                    if not root.is_dir():
                        raise FileNotFoundError(f"External drive root is not mounted: {root}")
                    root = root / EXTERNAL_NAME
                elif not root.parent.is_dir():
                    raise FileNotFoundError(f"External drive root is unavailable: {root.parent}")
                if root == self.root or root in self.root.parents or self.root in root.parents:
                    raise ValueError(
                        "External storage must not overlap the internal MiniTopol Storage")
                # Check the actual backing filesystem before creating a marker,
                # folder, or copy. In particular /media/usb may exist on the
                # internal disk while the USB drive is disconnected.
                self._require_external_device(root if root.is_dir() else root.parent)
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
            if cache_percent is not None and cache_limit_bytes is not None:
                raise ValueError("Specify either cache_percent or cache_limit_bytes, not both")
            if cache_percent is not None:
                if (isinstance(cache_percent, bool) or not isinstance(cache_percent, (int, float))
                        or not math.isfinite(cache_percent) or not 0 < cache_percent <= 100):
                    raise ValueError("cache_percent must be finite and in (0, 100]")
                changes["cache_percent"] = float(cache_percent)
                changes["cache_limit_bytes"] = None  # Switch back to relative mode.
            if cache_limit_bytes is not None:
                if type(cache_limit_bytes) is not int or cache_limit_bytes < 0:
                    raise ValueError("cache_limit_bytes must be a nonnegative integer")
                changes["cache_limit_bytes"] = cache_limit_bytes
            if cache_slack_percent is not None:
                if (isinstance(cache_slack_percent, bool) or
                        not isinstance(cache_slack_percent, (int, float)) or
                        not math.isfinite(cache_slack_percent) or cache_slack_percent < 0):
                    raise ValueError("cache_slack_percent must be finite and nonnegative")
                changes["cache_slack_percent"] = float(cache_slack_percent)
            if cache_slack_bytes is not None:
                if type(cache_slack_bytes) is not int or cache_slack_bytes < 0:
                    raise ValueError("cache_slack_bytes must be a nonnegative integer")
                changes["cache_slack_bytes"] = cache_slack_bytes
            if write_policy is not None:
                if write_policy not in ("longevity", "balanced", "throughput"):
                    raise ValueError("Invalid write_policy")
                changes["write_policy"] = write_policy
                if prefetch_batches is None:
                    changes["prefetch_batches"] = {
                        "longevity": 1, "balanced": 2, "throughput": 4
                    }[write_policy]
            if prefetch_batches is not None:
                if type(prefetch_batches) is not int or not 1 <= prefetch_batches <= 64:
                    raise ValueError("prefetch_batches must be between 1 and 64")
                changes["prefetch_batches"] = prefetch_batches
            merged = {**current, **changes}
            if merged["enabled"] and merged["external_root"] is None:
                raise ValueError("Configure external storage before enabling multi-tier batches")
            if merged["external_root"] is not None and (
                    external_root is not None or enabled is True):
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

    def _cap(self, config=None):
        config = config or self.settings()
        fixed = config.get("cache_limit_bytes")
        if fixed is not None:
            return fixed
        return int(self.available_at_start * config["cache_percent"] / 100.0)

    def _soft_cap(self, config=None):
        config = config or self.settings()
        target = self._cap(config)
        # This is operational headroom for capacity fluctuations and atomic
        # transfer peaks, not an allocation guarantee. Physical free space is
        # checked separately before every actual copy.
        return target + int(target * config["cache_slack_percent"] / 100.0) + config["cache_slack_bytes"]

    def status(self):
        config = self.settings()
        rows = self._cache_rows()
        with self._db() as db:
            objects = db.execute("SELECT COUNT(*) FROM tier_objects").fetchone()[0]
            metrics = dict(db.execute("SELECT name, value FROM tier_metrics"))
        external_available = False
        if config["external_root"] is not None:
            try:
                self._external_root(config)
                external_available = True
            except (FileNotFoundError, RuntimeError, ValueError):
                pass
        return {**config, **metrics, "external_available": external_available,
                "available_at_start_bytes": self.available_at_start,
                "available_now_bytes": shutil.disk_usage(self.root).free,
                "temporary_cache_mode": "absolute" if config["cache_limit_bytes"] is not None else "relative",
                "temporary_cache_limit_bytes": self._cap(config),
                "temporary_cache_soft_limit_bytes": self._soft_cap(config),
                "temporary_cache_bytes": sum(size for _, size, _ in rows),
                "managed_external_objects": objects}

    def _prune(self, *, required=0, exempt=(), all_cache=False):
        cap = self._soft_cap()
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
                    if path.exists() and _tree_digest(path)[0] != record[1]:
                        # Locally modified data must never be discarded as a cache.
                        continue
                    # The external primary may have disappeared or become
                    # corrupt since offload. Never delete the last valid copy
                    # based solely on an old catalog entry.
                    external = self._safe_path(self._external_root(), relative)
                    actual, _ = _tree_digest(external)
                    if actual != record[1]:
                        raise IOError(f"Cannot evict: external primary checksum differs: {relative}")
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
                    self._validated_internal.pop(relative, None)
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

    @staticmethod
    def _add_metric(db, name, amount):
        db.execute("INSERT INTO tier_metrics VALUES (?, ?) "
                   "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
                   (name, amount))

    def offload(self, relative):
        relative = logical_path(relative)
        with self._state_lock, self._lock(relative):
            config = self.settings()
            external = self._external_root(config)
            internal_path = self._safe_path(self.root, relative)
            external_path = self._safe_path(external, relative)
            if not internal_path.exists():
                raise FileNotFoundError(f"Cannot offload missing internal data: {relative}")
            digest, size, copied = _copy_verified(internal_path, external_path)
            self._validated_internal[relative] = digest
            self._validated_external[relative] = digest
            with self._db() as db:
                db.execute("INSERT OR REPLACE INTO tier_objects VALUES (?, 'external', ?, ?, ?)",
                           (relative, digest, size, time.time()))
                if copied:
                    self._add_metric(db, "external_copy_operations", 1)
                    self._add_metric(db, "external_written_bytes", size)
        # An offload never deletes its source before publishing a verified primary.
        if config["enabled"]:
            self._prune()
        return {"relative": relative, "sha256": digest, "bytes": size,
                "external_copy_created": copied}

    def _touch(self, relative):
        # Persist cache recency at most once per two minutes per process:
        # reliable enough for eviction without a DB write on every get().
        now = time.monotonic()
        if now - self._touch_times.get(relative, 0) < 120:
            return
        with self._db() as db:
            db.execute("UPDATE tier_objects SET last_used=? WHERE relative=?",
                       (time.time(), relative))
        self._touch_times[relative] = now

    def _direct_external(self, relative, record):
        """Verify one immutable external primary before read-only direct access."""
        external = self._safe_path(self._external_root(), relative)
        if self._validated_external.get(relative) != record[1]:
            digest, size = _tree_digest(external)
            if digest != record[1] or size != record[2]:
                raise IOError(f"External primary differs from its published checksum: {relative}")
            self._validated_external[relative] = digest
        return external

    def _ensure_locked(self, relative, exempt=()):
        record = self._object(relative)
        internal = self._safe_path(self.root, relative)
        if not record or record[0] != "external":
            if not internal.exists():
                raise FileNotFoundError(f"Dataset unavailable: {relative}")
            return False
        if internal.exists():
            # A file's existence is not proof of validity. Verify once per
            # Provider lifetime; repeated batch requests do not hash it again.
            if self._validated_internal.get(relative) != record[1]:
                actual, _ = _tree_digest(internal)
                if actual != record[1]:
                    raise ValueError(
                        f"Internal cache differs from its verified primary: {relative}")
                self._validated_internal[relative] = actual
            self._touch(relative)
            return False
        external = self._safe_path(self._external_root(), relative)
        size = record[2]
        if size > self._soft_cap():
            # A logical object larger than the configured soft allowance is
            # read directly. No entire-object staging or artificial hard error.
            self._direct_external(relative, record)
            return "direct"
        with self._capacity_lock:
            # Serialize admission and publication of distinct staging copies.
            # A concurrent transfer may have consumed capacity since the
            # initial lookup; recompute used space while holding this lock.
            try:
                self._prune(required=size, exempt=tuple(exempt) + (relative,))
            except OSError as exc:
                if "Temporary storage limit cannot accommodate" not in str(exc):
                    raise
                self._direct_external(relative, record)
                return "direct"
            # The startup snapshot governs the *target*, but physical free
            # space may have drifted. Reserve some transfer working room.
            if shutil.disk_usage(self.root).free < size + 8 * MIB:
                self._direct_external(relative, record)
                return "direct"
            try:
                _copy_verified(external, internal, expected=record[1])
            except OSError as exc:
                if exc.errno != errno.ENOSPC:
                    raise
                self._direct_external(relative, record)
                return "direct"
            with self._db() as db:
                db.execute("UPDATE tier_objects SET last_used=? WHERE relative=?",
                           (time.time(), relative))
                self._add_metric(db, "internal_copy_operations", 1)
                self._add_metric(db, "internal_written_bytes", size)
        self._touch_times[relative] = time.monotonic()
        self._validated_internal[relative] = record[1]
        return True

    def ensure(self, relative):
        relative = logical_path(relative)
        with self._lock(relative):
            outcome = self._ensure_locked(relative)
            return {"relative": relative, "transferred": outcome is True,
                    "direct_read": outcome == "direct"}

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
            if paths:
                placeholders = ",".join("?" for _ in paths)
                with self._db() as db:
                    managed = {row[0] for row in db.execute(
                        "SELECT relative FROM tier_objects WHERE primary_tier='external' "
                        f"AND relative IN ({placeholders})", paths)}
                redirects = {}
                for name in paths:
                    if name in managed:
                        outcome = self._ensure_locked(name, exempt=paths)
                        if outcome == "direct":
                            redirects[name] = self._direct_external(name, self._object(name))
                with workspace.storage_read_scope(redirects):
                    yield
            else:
                yield

    def prefetch(self, request, workspace):
        if not self.settings()["enabled"]:
            raise ValueError("Multi-tier batches are disabled")
        # Pin the full logical request while staging. Otherwise staging the
        # second descriptor could evict the first within the same batch.
        with self.hold_request(request, workspace):
            return {"prepared": self.request_paths(request, workspace)}


    def ingest_sources(self, trajectory):
        """Explicitly adopt converted raw topology/XTC into managed storage.

        Source fingerprints in the immutable trajectory identity are retained.
        Manifest absolute source paths are repointed only after verified copies.
        Originals are never deleted.
        """
        from .storage import atomic_json, fingerprint, read_json, identifier

        identifier(trajectory)
        relative = f"trajectories/{trajectory}"
        with self._lock(relative):
            if self._object(relative) is not None:
                raise ValueError(
                    "Raw source ingestion changes the manifest; ingest before "
                    "offloading this trajectory, not while its primary is external")
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
    if target.exists():
        target.rmdir()  # An empty destination can be populated atomically.
    _, _, copied = _copy_verified(source, target, expected=digest)
    record = Path.home() / ".minitopol" / "storage-location.json"
    if record.is_file():
        settings = json.loads(record.read_text(encoding="utf-8"))
        if settings.get("path") == str(source):
            temporary = record.with_suffix(".tmp")
            temporary.write_text(json.dumps({"path": str(target)}), encoding="utf-8")
            os.replace(temporary, record)
    return {"source": str(source), "destination": str(target),
            "bytes": size, "sha256": digest, "copied": copied,
            "original_retained": True}
