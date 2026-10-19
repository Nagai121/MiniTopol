"""GRO + XTC to an unaligned, unwrapped-independent heavy-atom base A."""

from pathlib import Path
from contextlib import contextmanager

import MDAnalysis as mda
import numpy as np
import pyarrow as pa
import zarr

from .storage import atomic_json, atomic_parquet, digest, fingerprint, read_json


def convert(workspace, gro, xtc, *, solvent_resnames, trajectory_id=None,
            atom_elements=None, chunk_frames=None):
    """Convert all frames, resuming only at committed Zarr chunk boundaries.

    Solvent residue names are explicit because GRO does not identify solvent.
    Optional atom_elements supplies one element for every *source* atom.
    Otherwise MDAnalysis guesses elements from names; this provenance is saved.
    No chemical bonds are inferred. Original selected coordinates and all box
    vectors are retained in Å; times are retained in ps.
    """
    if chunk_frames is not None and (type(chunk_frames) is not int or chunk_frames <= 0):
        raise ValueError("chunk_frames must be a positive integer")
    if isinstance(solvent_resnames, str):
        raise TypeError("solvent_resnames must be a sequence of residue names")
    solvents = sorted(set(solvent_resnames))
    source = {"gro": fingerprint(gro), "xtc": fingerprint(xtc)}
    with _universe(gro, xtc) as universe:
        if atom_elements is None:
            universe.guess_TopologyAttrs(to_guess=["elements"])
            elements = np.asarray(universe.atoms.elements, dtype=str)
            element_origin = "MDAnalysis name-based guess"
        else:
            elements = np.asarray(atom_elements, dtype=str)
            element_origin = "caller supplied"
        elements = np.array([value.strip().capitalize() for value in elements])
        if elements.shape != (len(universe.atoms),) or np.any(elements == ""):
            raise ValueError("A nonempty element is required for every source atom")
        selected = np.flatnonzero(
            ~np.isin(elements, ["H", "D", "T"]) & ~np.isin(universe.atoms.resnames, solvents)
        )
        if not len(selected):
            raise ValueError("No non-solvent heavy atoms remain")
        if chunk_frames is None:
            # ~32 MiB of coordinate payload per write, capped for small systems.
            chunk_frames = max(1, min(1024, (32 * 1024 * 1024) // (len(selected) * 3 * 4)))
        options = {"source": source, "solvent_resnames": solvents,
                   "elements_sha256": digest(elements.tolist()), "chunk_frames": chunk_frames, "format": 1}
        identity = digest(options)
        trajectory_id = trajectory_id or identity[:32]
        path = workspace.trajectory_path(trajectory_id)
        with workspace.lock("trajectory-" + trajectory_id):
            path.mkdir(exist_ok=True)
            manifest_path = path / "manifest.json"
            frames, atoms = len(universe.trajectory), len(selected)
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                if manifest["identity"] != identity:
                    raise ValueError("Trajectory ID already belongs to different sources/options")
            else:
                manifest = {"schema": 1, "identity": identity, "options": options,
                            "gro_path": str(Path(gro).resolve()), "xtc_path": str(Path(xtc).resolve()),
                            "element_origin": element_origin, "frames": frames, "atoms": atoms,
                            "committed_frames": 0, "status": "converting",
                            "coordinate_unit": "angstrom", "time_unit": "ps"}
                atomic_json(manifest_path, manifest)
            group = zarr.open_group(str(path / "A.zarr"), mode="a", zarr_format=3)
            specs = {"positions": ((frames, atoms, 3), (chunk_frames, atoms, 3), "f4"),
                     "time": ((frames,), (chunk_frames,), "f8"),
                     "box": ((frames, 3, 3), (chunk_frames, 3, 3), "f4")}
            for name, (shape, chunks, dtype) in specs.items():
                if name not in group:
                    if manifest["committed_frames"]:
                        raise IOError("Committed trajectory array is missing")
                    group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
                elif group[name].shape != shape or group[name].chunks != chunks:
                    raise IOError("Trajectory array schema does not match manifest")
            group.attrs.update({"representation": "A", "coordinate_unit": "angstrom", "time_unit": "ps"})
            ag = universe.atoms[selected]
            atomic_parquet(path / "atoms.parquet", pa.table({
                "point": np.arange(atoms), "source_atom": selected,
                "name": ag.names.tolist(), "element": elements[selected].tolist(),
                "residue_index": ag.resindices, "residue_id": ag.resids,
                "residue_name": ag.resnames.tolist(), "segment": ag.segids.tolist(),
            }))
            for start in range(manifest["committed_frames"], frames, chunk_frames):
                stop = min(start + chunk_frames, frames)
                positions = np.empty((stop - start, atoms, 3), dtype=np.float32)
                time = np.empty(stop - start, dtype=np.float64)
                boxes = np.empty((stop - start, 3, 3), dtype=np.float32)
                for index, ts in enumerate(universe.trajectory[start:stop]):
                    positions[index] = ts.positions[selected]
                    time[index] = ts.time
                    boxes[index] = np.nan if ts.triclinic_dimensions is None else ts.triclinic_dimensions
                if not np.isfinite(positions).all() or not np.isfinite(time).all():
                    raise ValueError("Nonfinite source coordinates or times")
                group["positions"][start:stop] = positions
                group["time"][start:stop] = time
                group["box"][start:stop] = boxes
                # The manifest is the commit marker. A restart overwrites an
                # interrupted, uncommitted chunk without touching earlier chunks.
                manifest["committed_frames"] = stop
                atomic_json(manifest_path, manifest)
            manifest["status"] = "complete"
            atomic_json(manifest_path, manifest)
            with workspace.catalog() as con:
                con.execute("INSERT INTO trajectories VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                            [trajectory_id, path.relative_to(workspace.path).as_posix(), frames, atoms, identity])
        return trajectory_id


@contextmanager
def _universe(gro, xtc):
    universe = mda.Universe(str(gro), str(xtc))
    try:
        yield universe
    finally:
        universe.trajectory.close()
