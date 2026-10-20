"""Small, explicit client for the ATLAS protein molecular-dynamics REST API.

Downloads are archives, not automatically imported trajectories. In particular,
ATLAS usually supplies PDB/TPR + XTC, while the existing MiniTopol converter
accepts GRO + XTC; this module never invents a GRO topology.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen
import zipfile
import stat
import zlib

from .ledger import DownloadLedger

BASE_URL = "https://www.dsimb.inserm.fr/ATLAS/api"
DATASET = re.compile(r"[A-Za-z0-9]{4}_[A-Za-z0-9]")
KINDS = frozenset(("analysis", "protein", "total"))
EXTRACT_SUFFIXES = frozenset((".gro", ".xtc", ".pdb", ".tpr"))


def _dataset_id(dataset: str) -> str:
    if not isinstance(dataset, str) or not DATASET.fullmatch(dataset):
        raise ValueError("ATLAS ID must be a PDB code and one chain, e.g. 1k5n_A")
    return dataset.lower()[:4] + "_" + dataset[5:].upper()


def _kind(value: str) -> str:
    if value not in KINDS:
        raise ValueError("format must be analysis, protein or total")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def _extract_inputs(archive: Path, target: Path) -> list[str]:
    """Extract only trajectory-related files, never trusting ZIP member paths."""
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        members = {}
        for info in zf.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in EXTRACT_SUFFIXES:
                continue
            # Do not follow or create symbolic links supplied in a ZIP.
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise ValueError("ATLAS archive contains a symbolic link")
            name = Path(info.filename.replace("\\", "/")).name
            if not name or name in members:
                raise ValueError("Ambiguous trajectory filenames in ATLAS archive")
            members[name] = info
        if not any(name.lower().endswith(".xtc") for name in members):
            raise ValueError("ATLAS archive contains no XTC trajectory")
        for name, info in members.items():
            dst = target / name
            if dst.exists():
                if dst.is_file() and dst.stat().st_size == info.file_size and _crc32(dst) == info.CRC:
                    extracted.append(str(dst))
                    continue
                raise FileExistsError(f"Existing extracted file differs: {dst}")
            fd, tmpname = tempfile.mkstemp(prefix=".atlas-extract-", dir=target)
            try:
                with os.fdopen(fd, "wb") as output, zf.open(info) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if Path(tmpname).stat().st_size != info.file_size:
                    raise IOError(f"Invalid extracted size for {name}")
                os.replace(tmpname, dst)
            finally:
                Path(tmpname).unlink(missing_ok=True)
            extracted.append(str(dst))
    return extracted


class AtlasClient:
    """ATLAS metadata and ZIP acquisition, using an injectable URL opener."""

    def __init__(self, *, timeout: float = 60.0, opener=None):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self._opener = opener or urlopen

    def _request(self, url: str, *, accept: str):
        req = Request(url, headers={
            "Accept": accept,
            "User-Agent": "MiniTopol/0.1 (ATLAS data client)",
        })
        return self._opener(req, timeout=self.timeout)

    def metadata(self, dataset: str) -> dict:
        import json
        dataset = _dataset_id(dataset)
        url = f"{BASE_URL}/ATLAS/metadata/{dataset}"
        with self._request(url, accept="application/json") as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError("ATLAS returned unexpected metadata JSON")
        return result

    def download(self, workspace, dataset: str, kind: str = "analysis", *,
                 extract: bool = True) -> dict:
        """Fetch into trajectories/atlas-<dataset>-<kind>/source, never overwriting.

        The ledger records downloaded archives; extraction is independently
        repeatable. Large archives are streamed to a temporary file, checked for
        complete size (where provided) and valid ZIP directory, then published.
        """
        dataset, kind = _dataset_id(dataset), _kind(kind)
        url = f"{BASE_URL}/ATLAS/{kind}/{dataset}"
        trajectory_id = f"atlas-{dataset}-{kind}"
        directory = workspace.trajectory_path(trajectory_id) / "source"
        archive = directory / f"{dataset}_{kind}.zip"
        ledger = DownloadLedger(workspace)
        with workspace.lock(f"atlas-download-{trajectory_id}"):
            directory.mkdir(parents=True, exist_ok=True)
            old = ledger.get(dataset, kind)
            if archive.exists():
                checksum = _sha256(archive)
                if not (archive.is_file() and zipfile.is_zipfile(archive)
                        and old and old["status"] == "complete"
                        and old["sha256"] == checksum
                        and int(old["bytes"]) == archive.stat().st_size):
                    raise FileExistsError("Existing ATLAS archive is unverified; will not overwrite it")
                status = "skipped"
            else:
                record = {
                    "dataset": dataset, "kind": kind, "url": url,
                    "archive": archive.relative_to(workspace.path).as_posix(),
                    "downloaded_at": "", "bytes": "", "sha256": "",
                    "status": "downloading", "error": "",
                }
                ledger.put(record)
                tmpname = None
                try:
                    fd, tmpname = tempfile.mkstemp(prefix=".atlas-download-", dir=directory)
                    h = hashlib.sha256()
                    size = 0
                    with os.fdopen(fd, "wb") as output, self._request(url, accept="application/zip") as response:
                        length = response.headers.get("Content-Length")
                        expected = int(length) if length is not None else None
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(block)
                            h.update(block)
                            size += len(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if expected is not None and size != expected:
                        raise IOError(f"Incomplete ATLAS download: expected {expected} bytes, got {size}")
                    if not zipfile.is_zipfile(tmpname):
                        raise ValueError("ATLAS response is not a ZIP archive")
                    with zipfile.ZipFile(tmpname) as zf:
                        if not zf.infolist():
                            raise ValueError("ATLAS returned an empty ZIP archive")
                    os.replace(tmpname, archive)
                    record.update(
                        downloaded_at=datetime.now(timezone.utc).isoformat(),
                        bytes=str(size), sha256=h.hexdigest(), status="complete",
                    )
                    ledger.put(record)
                    status = "downloaded"
                except Exception as exc:
                    # The ZIP may already have been atomically published if the
                    # ledger write failed. Preserve it; never silently overwrite.
                    record.update(status="failed", error=str(exc)[:500])
                    try:
                        ledger.put(record)
                    except Exception:
                        pass
                    raise
                finally:
                    if tmpname is not None:
                        Path(tmpname).unlink(missing_ok=True)
            extracted = _extract_inputs(archive, directory) if extract else []
            return {
                "dataset": dataset, "kind": kind, "status": status,
                "trajectory_id": trajectory_id,
                "archive": str(archive), "sha256": _sha256(archive),
                "files": extracted,
            }
