"""Local stdio MCP façade. Scientific calculation remains in MiniTopol core."""

from mcp.server.fastmcp import FastMCP

from ..atlas import AtlasClient, DownloadLedger, check_atlas, import_atlas
from ..bond_store import BondStore
from ..converter import convert
from ..registry import Registry
from ..workspace import Workspace

mcp = FastMCP("MiniTopol")


@mcp.tool()
def run_trajectory_pipelines(jobs: list[dict], workers: int = 6,
                             native_threads: int = 2, gpu_workers: int = 2,
                             rss_limit_mb: float = 4096) -> dict:
    """Synchronously convert and analyze independent jobs in separate Workspaces.

    Each job explicitly supplies workspace, gro, xtc, solvent_resnames and
    analysis.coordinate_mode. Optional annotation and TPR atom mapping are
    never inferred. Only the largest gpu_workers jobs may initialize CUDA.
    """
    from ..batch import run_pipelines
    return run_pipelines(jobs, workers=workers, native_threads=native_threads,
                         gpu_workers=gpu_workers, rss_limit_mb=rss_limit_mb)


@mcp.tool()
def atlas_metadata(dataset: str) -> dict:
    """Get ATLAS metadata for a PDB-chain ID such as 1k5n_A."""
    return AtlasClient().metadata(dataset)


@mcp.tool()
def atlas_download(dataset: str, kind: str = "analysis", extract: bool = True) -> dict:
    """Download an ATLAS archive into the MiniTopol Workspace.

    kind: analysis (1000-frame protein), protein (10000-frame protein),
    total (10000-frame full system). Source PDB/TPR/XTC files are not
    automatically converted on download; use atlas_import for validated conversion.
    """
    return AtlasClient().download(Workspace(), dataset, kind, extract=extract)



@mcp.tool()
def atlas_check(dataset: str, kind: str = "analysis") -> list[dict]:
    """Inspect extracted ATLAS replicas and optional TPR correspondence."""
    plans = check_atlas(Workspace(), dataset, kind)
    # Large atom-index mappings are consumed internally, never dumped over MCP.
    return [{k: v for k, v in plan.items() if k != "source_to_tpr"}
            for plan in plans]


@mcp.tool()
def atlas_import(dataset: str, kind: str = "analysis",
                 solvent_resnames: list[str] | None = None) -> dict:
    """Convert each validated ATLAS replica to A and independently add TPR bonds."""
    solvents = ("SOL", "WAT", "HOH") if solvent_resnames is None else solvent_resnames
    return import_atlas(Workspace(), dataset, kind, solvent_resnames=solvents)


@mcp.tool()
def get_bond_graph(trajectory: str, limit: int = 1000) -> dict:
    """Return verified TPR-derived static A-point connectivity (at most 1000 edges)."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be in 1..1000")
    workspace = Workspace()
    table = BondStore(workspace).get(trajectory)
    return {"trajectory": trajectory, "total_edges": table.num_rows,
            "edges": table.slice(0, limit).to_pylist()}


@mcp.tool()
def atlas_downloads() -> list[dict]:
    """Read the tiny TSV ledger of ATLAS downloads."""
    return DownloadLedger(Workspace()).list()


@mcp.tool()
def list_trajectories() -> list[dict]:
    """List completed, converted MiniTopol trajectories (not raw ATLAS ZIPs)."""
    return [
        {"id": row[0], "frames": row[1], "atoms": row[2]}
        for row in Workspace().list_trajectories()
    ]


@mcp.tool()
def list_representations() -> list[str]:
    """List available registered A/C molecular representations."""
    workspace = Workspace()
    return Registry(workspace).representations()


@mcp.tool()
def convert_trajectory(gro: str, xtc: str, solvent_resnames: list[str],
                       trajectory_id: str | None = None) -> str:
    """Convert local GRO+XTC into the existing MiniTopol Registry."""
    return convert(Workspace(), gro, xtc, solvent_resnames=solvent_resnames,
                   trajectory_id=trajectory_id)


@mcp.tool()
def annotate_trajectory(trajectory: str, chains: list[dict]) -> str:
    """Register verified chain/segment continuity; never infer chemical bonds."""
    workspace = Workspace()
    return Registry(workspace).annotate(trajectory, chains=chains)


@mcp.tool()
def calculate_descriptors(trajectory: str, coordinate_mode: str,
                          initial_whole: bool = False,
                          half_box_steps: bool = False,
                          memory_mb: float = 128.0,
                          descriptors: list[str] | None = None) -> dict:
    """Run existing A descriptor calculators synchronously.

    This can be long-running. It does not create a background job.
    coordinate_mode: nonperiodic, continuous, or nojump.
    """
    from ..descriptors import analyze
    return analyze(Workspace(), trajectory, descriptors=descriptors,
                   coordinate_mode=coordinate_mode, initial_whole=initial_whole,
                   half_box_steps=half_box_steps, memory_mb=memory_mb)


@mcp.tool()
def list_descriptor_results(trajectory: str | None = None,
                            descriptor: str | None = None,
                            limit: int = 100) -> list[dict]:
    """List completed descriptor result metadata without loading large Parquet data."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be in 1..1000")
    workspace = Workspace()
    clauses = []
    values = []
    if trajectory is not None:
        clauses.append("trajectory = ?")
        values.append(trajectory)
    if descriptor is not None:
        clauses.append("descriptor = ?")
        values.append(descriptor)
    sql = "SELECT id, trajectory, representation, descriptor, path, rows FROM results"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id LIMIT ?"
    with workspace.catalog() as con:
        records = con.execute(sql, [*values, limit]).fetchall()
    return [
        {"id": a, "trajectory": b, "representation": c,
         "descriptor": d, "path": e, "rows": f}
        for a, b, c, d, e, f in records
    ]



@mcp.tool()
def register_representation(name: str, trajectory: str, version: str,
                            groups: list[dict],
                            descriptor_definitions: dict) -> str:
    """Register an explicit C mapping and its predeclared descriptor definitions."""
    workspace = Workspace()
    return Registry(workspace).register(name, trajectory=trajectory,
                                        version=version, groups=groups,
                                        descriptor_definitions=descriptor_definitions)


@mcp.tool()
def get_representation_definition(name: str) -> dict:
    """Get registered A/C definition; does not generate a C trajectory."""
    workspace = Workspace()
    return Registry(workspace).definition(name)


@mcp.tool()
def query_descriptor_rows(result_id: str, limit: int = 100) -> list[dict]:
    """Return at most 1000 rows of an existing result without materializing all rows.

    The result ID must come from list_descriptor_results. Source Parquet is
    filtered with DuckDB LIMIT before materialization.
    """
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be in 1..1000")
    workspace = Workspace()
    with workspace.catalog() as con:
        parent = con.execute("SELECT path FROM results WHERE id = ?", [result_id]).fetchone()
        if parent is None:
            raise LookupError("Unknown completed descriptor result ID")
        marker = workspace.path.joinpath(parent[0]).with_suffix('.json')
        from ..storage import read_json
        if read_json(marker).get('schema') == 5:
            from ..descriptors.virtual_rama_store import read_rows
            return read_rows(workspace,parent[0],limit,connection=con)
        if read_json(marker).get('schema') in (4,6):
            from ..descriptors.count_store import read_rows
            return read_rows(workspace, parent[0], limit)
        parts = con.execute(
            "SELECT path FROM result_parts WHERE result_id = ? ORDER BY part_index",
            [result_id],
        ).fetchall()
        files = [str(workspace.path / row[0]) for row in parts] if parts else [
            str(workspace.path / parent[0])
        ]
        relation = con.read_parquet(files, union_by_name=True, hive_partitioning=False)
        return relation.limit(limit).to_arrow_table().to_pylist()

@mcp.tool()
def get_hydrogen_bond_info(trajectory: str) -> dict:
    """Verify and summarize independently stored 3-category H-bond events."""
    from ..hydrogen_bonds import hydrogen_bond_info
    result = hydrogen_bond_info(Workspace(), trajectory)
    if result is None:
        raise LookupError("No completed hydrogen-bond dataset")
    return result


@mcp.tool()
def query_hydrogen_bond_events(trajectory: str, category: str,
                               start: int = 0, stop: int | None = None,
                               limit: int = 100) -> list[dict]:
    """Read verified completed H-bond events (max 1000) for one category."""
    from ..hydrogen_bonds import hydrogen_bond_events
    return hydrogen_bond_events(Workspace(), trajectory, category, start=start, stop=stop,
                                limit=limit)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
