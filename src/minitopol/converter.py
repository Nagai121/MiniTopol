"""GRO/PDB + XTC to an unaligned, unwrapped-independent heavy-atom base A."""

from pathlib import Path
from contextlib import contextmanager

import MDAnalysis as mda
import numpy as np
import pyarrow as pa
import zarr

from .storage import atomic_json, atomic_parquet, digest, fingerprint, read_json


def position_chunk_frames(logical,atoms):
    if type(logical) is not int or logical<1 or type(atoms) is not int or atoms<1:raise ValueError('Positive integer AGX logical/atom axes required')
    cap=min(256,logical,(8*1024**2)//(atoms*12))
    if cap<1:raise ValueError('One atom frame exceeds AGX8MiB working limit')
    return max(d for d in range(1,cap+1) if logical%d==0)


def convert(workspace, gro, xtc, *, solvent_resnames, trajectory_id=None,
            atom_elements=None, chunk_frames=None, hydrogen_bonds=None,
            hbond_report=None):
    """Convert all frames, resuming only at committed Zarr chunk boundaries.

    Solvent residue names are explicit for both GRO and PDB sources.
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
    source_format = Path(gro).suffix.lower().lstrip(".")
    if source_format not in ("gro", "pdb"):
        raise ValueError("A source topology must be GRO or PDB")
    source = {source_format: fingerprint(gro), "xtc": fingerprint(xtc)}
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
                   "elements_sha256": digest(elements.tolist()), "chunk_frames": chunk_frames, "format": 3, "position_codec": "AGX3-modular-commit-divisor256"}
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
                            source_format + "_path": str(Path(gro).resolve()), "xtc_path": str(Path(xtc).resolve()),
                            "element_origin": element_origin, "frames": frames, "atoms": atoms,
                            "committed_frames": 0, "status": "converting",
                            "coordinate_unit": "angstrom", "time_unit": "ps"}
                manifest['coordinate_accuracy']={'status': 'experimental_approximate', 'family': 'exact_A', 'precision': 32, 'unit': 'angstrom', 'axis_error_bound': 0, 'coordinate_plane': 'originalbits', 'SASA': 'mixed:relative-float32-RN-GPU-points;float64-CPU-fallback-near0.01angstrom', 'descriptor_plane': 'SASAapprox;allothersoriginal', 'scientific_acceptance': False, 'near_coincident_policy': 'Any distinct atom pair distance<=0.01angstrom triggers originalCPU wholeframe; originaloverflow/coincident/otherguards maintained'}
                atomic_json(manifest_path, manifest)
            group = zarr.open_group(str(path / "A.zarr"), mode="a", zarr_format=3)
            specs = {"positions": ((frames, atoms, 3), (chunk_frames, atoms, 3), "f4"),
                     "time": ((frames,), (chunk_frames,), "f8"),
                     "box": ((frames, 3, 3), (chunk_frames, 3, 3), "f4")}
            for name, (shape, chunks, dtype) in specs.items():
                if name not in group:
                    if manifest["committed_frames"]:
                        raise IOError("Committed trajectory array is missing")
                    if name == "positions":
                        from .agx_zarr import AGXCodec
                        physical=position_chunk_frames(chunk_frames,atoms)
                        group.create_array(name,shape=shape,chunks=(physical,atoms,3),dtype=dtype,compressors=[AGXCodec(atoms=atoms)])
                    else:
                        group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
                elif group[name].shape != shape or group[name].chunks != ((position_chunk_frames(chunk_frames,atoms),atoms,3) if name=="positions" else chunks):
                    raise IOError("Trajectory array schema does not match manifest")
            group.attrs.update({"representation": "A", "coordinate_unit": "angstrom", "time_unit": "ps"})
            ag = universe.atoms[selected]
            atomic_parquet(path / "atoms.parquet", pa.table({
                "point": np.arange(atoms), "source_atom": selected,
                "name": ag.names.tolist(), "element": elements[selected].tolist(),
                "residue_index": ag.resindices, "residue_id": ag.resids,
                "residue_name": ag.resnames.tolist(), "segment": ag.segids.tolist(),
            }))
            hb_writer, hb_error, hb_identity = None, None, None
            if hydrogen_bonds is not None:
                try:
                    from .hydrogen_bonds import HydrogenBondStore
                    hb_writer = HydrogenBondStore(
                        workspace, trajectory_id, source_identity=identity,
                        frames=frames, tpr=hydrogen_bonds["tpr"],
                        source_to_tpr=hydrogen_bonds["source_to_tpr"],
                        elements=elements, names=universe.atoms.names,
                        resnames=universe.atoms.resnames,
                        solvent_resnames=solvents, chunk_frames=chunk_frames,
                        periodic=hydrogen_bonds.get("periodic", True))
                    hb_identity = hb_writer.identity
                except Exception as error:
                    hb_error = f"{type(error).__name__}: {error}"
            # The single full-atom XTC walk feeds A and HB independently.
            # Already committed A chunks are never rewritten for HB backfill.
            for start in range(0, frames, chunk_frames):
                stop = min(start + chunk_frames, frames)
                need_a = start >= manifest["committed_frames"]
                need_hb = hb_writer is not None and not hb_writer.has(start, stop)
                if not need_a and not need_hb:
                    continue
                if need_a:
                    positions = np.empty((stop - start, atoms, 3), dtype=np.float32)
                    time = np.empty(stop - start, dtype=np.float64)
                    boxes = np.empty((stop - start, 3, 3), dtype=np.float32)
                def source_frames():
                    for index, ts in enumerate(universe.trajectory[start:stop]):
                        box = (ts.triclinic_dimensions.copy()
                               if ts.triclinic_dimensions is not None
                               else np.full((3, 3), np.nan))
                        if need_a:
                            positions[index] = ts.positions[selected]
                            time[index] = ts.time
                            boxes[index] = box
                        yield ts.positions, box
                if need_hb:
                    try:
                        hb_writer.append(start, stop, source_frames())
                    except Exception as error:
                        hb_error = f"{type(error).__name__}: {error}"
                        hb_writer.close()
                        hb_writer = None
                        # Complete the A chunk even when HB processing stopped
                        # before consuming its entire source-frame iterator.
                        if need_a:
                            for _ in source_frames():
                                pass
                elif need_a:
                    for _ in source_frames():
                        pass
                if need_a:
                    if not np.isfinite(positions).all() or not np.isfinite(time).all():
                        raise ValueError("Nonfinite source coordinates or times")
                    group["positions"][start:stop] = positions
                    group["time"][start:stop] = time
                    group["box"][start:stop] = boxes
                    manifest["committed_frames"] = stop
                    atomic_json(manifest_path, manifest)
            manifest["status"] = "complete"
            atomic_json(manifest_path, manifest)
            with workspace.catalog() as con:
                con.execute("INSERT INTO trajectories VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                            [trajectory_id, path.relative_to(workspace.path).as_posix(), frames, atoms, identity])
            if hb_writer is not None:
                try:
                    hb_writer.finish()
                except Exception as error:
                    hb_error = f"{type(error).__name__}: {error}"
                finally:
                    hb_writer.close()
            if hydrogen_bonds is not None:
                if hbond_report is not None:
                    hbond_report.update(
                        status="failed" if hb_error else "complete",
                        identity=hb_identity, error=hb_error)
                elif hb_error is not None:
                    raise RuntimeError("A conversion completed, hydrogen bonds failed: " + hb_error)
        from .system_metadata import publish_conditions
        publish_conditions(workspace, trajectory_id)
        return trajectory_id


@contextmanager
def _universe(gro, xtc):
    universe = mda.Universe(str(gro), str(xtc))
    try:
        yield universe
    finally:
        universe.trajectory.close()
