"""Local stdio MCP façade. Scientific calculation remains in MiniTopol core."""

from mcp.server.fastmcp import FastMCP

from ..atlas import AtlasClient, DownloadLedger
from ..converter import convert
from ..registry import Registry
from ..workspace import Workspace

mcp = FastMCP("MiniTopol")


@mcp.tool()
def atlas_metadata(dataset: str) -> dict:
    """Get ATLAS metadata for a PDB-chain ID such as 1k5n_A."""
    return AtlasClient().metadata(dataset)


@mcp.tool()
def atlas_download(dataset: str, kind: str = "analysis", extract: bool = True) -> dict:
    """Download an ATLAS archive into the MiniTopol Workspace.

    kind: analysis (1000-frame protein), protein (10000-frame protein),
    total (10000-frame full system). Source PDB/TPR/XTC files are not
    automatically converted to GRO; an appropriate GRO is required to run
    MiniTopol's existing GRO+XTC converter.
    """
    return AtlasClient().download(Workspace(), dataset, kind, extract=extract)


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
        parts = con.execute(
            "SELECT path FROM result_parts WHERE result_id = ? ORDER BY part_index",
            [result_id],
        ).fetchall()
        files = [str(workspace.path / row[0]) for row in parts] if parts else [
            str(workspace.path / parent[0])
        ]
        relation = con.read_parquet(files, union_by_name=True, hive_partitioning=False)
        return relation.limit(limit).to_arrow_table().to_pylist()

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
