"""Small, explicit client for the ATLAS protein molecular-dynamics REST API.

Downloads are archives, not automatically imported trajectories.
The checker/importer performs subsequent validated GRO/PDB + XTC conversion;
this download module never invents a GRO topology.
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
            if dst.is_symlink():
                raise FileExistsError(f"Extracted path must not be a symlink: {dst}")
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


def _verify_archive(archive: Path) -> None:
    """Validate the ZIP payload and required XTC before publishing it.

    Checking the central directory alone is insufficient: the file contents
    may still be truncated or fail their CRC. Stream decompression in testzip
    keeps peak memory independent of archive size.
    """
    if not zipfile.is_zipfile(archive):
        raise ValueError("ATLAS response is not a ZIP archive")
    with zipfile.ZipFile(archive) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if not members:
            raise ValueError("ATLAS archive contains no files")
        if not any(Path(info.filename.replace("\\", "/")).suffix.lower() == ".xtc"
                   for info in members):
            raise ValueError("ATLAS archive contains no XTC trajectory")
        # Verify all members, not just the entries that will be extracted.
        broken = zf.testzip()
        if broken is not None:
            raise zipfile.BadZipFile(f"CRC failure in ATLAS ZIP member: {broken}")


def _remove_interrupted_temporaries(directory: Path) -> None:
    """Remove only our unpublished temporary files, under the download lock."""
    for prefix in (".atlas-download-", ".atlas-extract-"):
        for entry in directory.glob(prefix + "*"):
            if entry.is_file() or entry.is_symlink():
                entry.unlink()


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
        """Download and verify an ATLAS ZIP, optionally extracting input files.

        "complete" in the TSV means the requested download/extraction finished.
        Failures after ZIP publication preserve the verified archive, allowing
        retry without a second network transfer. No unverified user file is
        overwritten, and abandoned temporary files are removed on retry.
        """
        dataset, kind = _dataset_id(dataset), _kind(kind)
        url = f"{BASE_URL}/ATLAS/{kind}/{dataset}"
        trajectory_id = f"atlas-{dataset}-{kind}"
        directory = workspace.trajectory_path(trajectory_id) / "source"
        archive = directory / f"{dataset}_{kind}.zip"
        relative = archive.relative_to(workspace.path).as_posix()
        ledger = DownloadLedger(workspace)
        with workspace.lock(f"atlas-download-{trajectory_id}"):
            directory.mkdir(parents=True, exist_ok=True)
            _remove_interrupted_temporaries(directory)
            old = ledger.get(dataset, kind)
            if old is None and (archive.exists() or archive.is_symlink()):
                raise FileExistsError("Untracked ATLAS archive will not be adopted or overwritten")
            if old and (old["url"] != url or old["archive"] != relative):
                raise ValueError("ATLAS ledger entry does not match the requested archive")
            record = dict(old) if old else {
                "dataset": dataset, "kind": kind, "url": url,
                "archive": relative, "downloaded_at": "", "bytes": "",
                "sha256": "", "status": "downloading", "error": "",
            }
            tmpname = None
            fresh = False
            try:
                if archive.exists() or archive.is_symlink():
                    if not archive.is_file() or archive.is_symlink() or not old:
                        raise FileExistsError("Untracked or non-regular ATLAS archive will not be overwritten")
                    size = archive.stat().st_size
                    checksum = _sha256(archive)
                    if (old["sha256"] and old["sha256"] != checksum) or (
                        old["bytes"] and int(old["bytes"]) != size
                    ):
                        raise FileExistsError("Existing ATLAS archive differs from its recorded checksum/size")
                    # Includes a one-time revalidation path for archives made by
                    # older MiniTopol versions that did not test member CRCs.
                    _verify_archive(archive)
                    if not record["downloaded_at"]:
                        record["downloaded_at"] = datetime.now(timezone.utc).isoformat()
                    record.update(bytes=str(size), sha256=checksum)
                else:
                    record.update(status="downloading", error="")
                    ledger.put(record)
                    fd, tmpname = tempfile.mkstemp(prefix=".atlas-download-", dir=directory)
                    checksum_state = hashlib.sha256()
                    size = 0
                    with os.fdopen(fd, "wb") as output, self._request(
                        url, accept="application/zip"
                    ) as response:
                        length = response.headers.get("Content-Length")
                        expected = int(length) if length is not None else None
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(block)
                            checksum_state.update(block)
                            size += len(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if expected is not None and size != expected:
                        raise IOError(f"Incomplete ATLAS download: expected {expected} bytes, got {size}")
                    _verify_archive(Path(tmpname))
                    os.replace(tmpname, archive)
                    tmpname = None
                    checksum = checksum_state.hexdigest()
                    record.update(
                        downloaded_at=datetime.now(timezone.utc).isoformat(),
                        bytes=str(size), sha256=checksum,
                    )
                    fresh = True
                # Only mark complete after requested extraction has succeeded.
                extracted = _extract_inputs(archive, directory) if extract else []
                record.update(status="complete", error="")
                ledger.put(record)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
                try:
                    ledger.put(record)
                except Exception:
                    # Preserve the original error and any already-published ZIP.
                    pass
                raise
            finally:
                if tmpname is not None:
                    Path(tmpname).unlink(missing_ok=True)
            return {
                "dataset": dataset, "kind": kind,
                "status": "downloaded" if fresh else "skipped",
                "trajectory_id": trajectory_id,
                "archive": str(archive), "sha256": checksum,
                "files": extracted,
            }
