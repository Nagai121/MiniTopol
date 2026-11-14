"""ATLAS raw-source locations: durable placement recorded in downloads.tsv.

The nine-field ledger uses an "external:" prefix in its archive field for
offloaded source ZIPs; converted trajectories and DuckDB stay on the SSD.
"""
from __future__ import annotations

from ..windows_paths import filesystem_path

_EXTERNAL_PREFIX = "external:"


def _source_rel(dataset: str, kind: str) -> str:
    return f"sources/ATLAS/{dataset}-{kind}"


def _external_source_root(workspace):
    from ..storage_tiers import TieredStorage
    manager = TieredStorage(workspace.path)
    return manager._external_root()


def _stored_archive(workspace, dataset, kind):
    from .ledger import DownloadLedger
    return DownloadLedger(workspace).get(dataset, kind)


def source_directory(workspace, dataset: str, kind: str):
    """Resolve raw storage; recorded tier always outranks current defaults.

    Existing untracked local/legacy sources remain readable after enabling an
    external default. A registered external source never silently falls back to
    the local disk when the drive is disconnected.
    """
    rel = _source_rel(dataset, kind)
    modern = workspace.path / rel
    legacy = workspace.trajectory_path(f"atlas-{dataset}-{kind}") / "source"
    record = _stored_archive(workspace, dataset, kind)
    if record is not None:
        archive = record["archive"]
        expected_file = f"{dataset}_{kind}.zip"
        if archive == _EXTERNAL_PREFIX + rel + "/" + expected_file:
            return filesystem_path(_external_source_root(workspace) / rel)
        if archive == rel + "/" + expected_file:
            return filesystem_path(modern)
        legacy_ref = (legacy / expected_file).relative_to(workspace.path).as_posix()
        if archive == legacy_ref:
            return filesystem_path(legacy)
        raise ValueError("Unexpected ATLAS ledger archive location: " + archive)

    if modern.exists() or modern.is_symlink():
        return filesystem_path(modern)
    if legacy.exists() or legacy.is_symlink():
        return filesystem_path(legacy)

    from ..storage_tiers import TieredStorage
    manager = TieredStorage(workspace.path)
    if manager.settings().get("atlas_source_tier", "internal") == "external":
        return filesystem_path(manager._external_root() / rel)
    return filesystem_path(modern)


def archive_reference(workspace, dataset, kind, directory):
    """Encode physical placement without adding columns to downloads.tsv."""
    rel = _source_rel(dataset, kind)
    expected_file = f"{dataset}_{kind}.zip"
    modern = filesystem_path(workspace.path / rel)
    legacy_logical = workspace.trajectory_path(
        f"atlas-{dataset}-{kind}") / "source"
    legacy = filesystem_path(legacy_logical)
    if directory == modern:
        return rel + "/" + expected_file
    if directory == legacy:
        return (legacy_logical / expected_file).relative_to(workspace.path).as_posix()
    if directory == filesystem_path(_external_source_root(workspace) / rel):
        return _EXTERNAL_PREFIX + rel + "/" + expected_file
    raise ValueError("ATLAS directory does not belong to configured storage")
