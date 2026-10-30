"""Canonical ATLAS source storage, with read-only legacy-path compatibility."""

def source_directory(workspace, dataset: str, kind: str):
    modern = workspace.path / "sources" / "ATLAS" / f"{dataset}-{kind}"
    legacy = workspace.trajectory_path(f"atlas-{dataset}-{kind}") / "source"
    if modern.exists() or modern.is_symlink():
        return modern
    if legacy.exists() or legacy.is_symlink():
        return legacy
    return modern
