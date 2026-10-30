"""Independent, resumable hydrogen-bond event layer for a full-atom XTC.

Chemical connectivity and molecular membership come from a validated TPR.
Geometry comes only from source XTC frames, never from heavy-atom A.
"""
from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_angles, capped_distance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .coordinates import box_dimensions
from .descriptors.bond_reconstruction import _covalent_edges_from_tpr
from .storage import atomic_json, atomic_parquet, canonical, digest, fingerprint, read_json

CATEGORIES = ("intramolecular", "intermolecular", "solvent_molecular")
SCHEMA = pa.schema([
    ("frame", pa.int64()), ("donor", pa.int64()), ("hydrogen", pa.int64()),
    ("acceptor", pa.int64()), ("distance_angstrom", pa.float64()),
    ("angle_degree", pa.float64()),
])


def _validate_mapping(mapping, source_count, tpr_count):
    if (not isinstance(mapping, (list, tuple)) or len(mapping) != source_count
            or any(type(x) is not int or x < 0 or x >= tpr_count for x in mapping)
            or len(set(mapping)) != len(mapping)
            or any(a >= b for a, b in zip(mapping, mapping[1:]))):
        raise ValueError("A complete, unique, order-preserving source-to-TPR mapping is required")
    return np.asarray(mapping, dtype=np.int64)


class HydrogenBondStore:
    """Independent three-category published dataset; no A/result mutation."""

    def __init__(self, workspace, trajectory, *, source_identity, frames, tpr,
                 source_to_tpr, elements, names, resnames, solvent_resnames,
                 chunk_frames, periodic=True, distance_cutoff=3.0,
                 angle_cutoff=150.0, acceptor_charge_max=-0.5):
        if type(periodic) is not bool:
            raise ValueError("periodic must be boolean")
        if (not np.isfinite([distance_cutoff, angle_cutoff, acceptor_charge_max]).all()
                or distance_cutoff <= 0 or not 0 <= angle_cutoff <= 180):
            raise ValueError("Invalid hydrogen-bond geometry or charge selection")
        if type(chunk_frames) is not int or chunk_frames < 1 or type(frames) is not int or frames < 1:
            raise ValueError("Invalid hydrogen-bond frame boundaries")
        self.workspace, self.trajectory = workspace, trajectory
        self.frames, self.chunk_frames = frames, chunk_frames
        self.directory = workspace.trajectory_path(trajectory) / "hydrogen_bonds"
        self.complete_path = self.directory / "manifest.json"
        self.progress_path = self.directory / "progress.json"
        self.lock = workspace.lock("trajectory-hydrogen-bonds-" + trajectory)
        self.lock.acquire()
        try:
            elements = np.asarray(elements, dtype=str)
            names = np.asarray(names, dtype=str)
            resnames = np.asarray(resnames, dtype=str)
            if any(len(array) != len(elements) for array in (names, resnames)):
                raise ValueError("Source atom metadata axes differ")
            tpr_universe = mda.Universe(str(tpr))
            try:
                tpr_count = len(tpr_universe.atoms)
                charges = np.asarray(tpr_universe.atoms.charges, dtype=np.float64).copy()
            finally:
                if getattr(tpr_universe, "trajectory", None) is not None:
                    tpr_universe.trajectory.close()
            if not np.isfinite(charges).all() or len(charges) != tpr_count:
                raise ValueError("TPR does not provide finite charges for all atoms")
            mapping = _validate_mapping(source_to_tpr, len(elements), tpr_count)
            count, edges, molecule_ids = _covalent_edges_from_tpr(
                tpr, include_settle=True, molecule_ids=True)
            if count != tpr_count or len(molecule_ids) != count:
                raise ValueError("TPR chemical topology does not match atom metadata")
            molecule_ids = np.asarray(molecule_ids, dtype=np.int64)
            inverse = {int(tpr_atom): src for src, tpr_atom in enumerate(mapping)}
            solvent_names = sorted(set(solvent_resnames))
            is_solvent = np.isin(resnames, solvent_names)
            # A molecule is classified by its entire mapped membership; a
            # mixed solvent/non-solvent molecule is ambiguous, never guessed.
            molecule_solvent = {}
            for source, tpr_atom in enumerate(mapping):
                mol = int(molecule_ids[tpr_atom])
                flag = bool(is_solvent[source])
                if mol in molecule_solvent and molecule_solvent[mol] != flag:
                    raise ValueError("Solvent labels disagree inside one TPR molecule")
                molecule_solvent[mol] = flag
            donor_h = set()
            for a, b in edges:
                if a not in inverse or b not in inverse:
                    continue
                i, j = inverse[a], inverse[b]
                x, y = elements[i].capitalize(), elements[j].capitalize()
                if x in ("H", "D", "T") and y in ("N", "O", "S"):
                    donor_h.add((j, i))
                elif y in ("H", "D", "T") and x in ("N", "O", "S"):
                    donor_h.add((i, j))
            donor_h = sorted(donor_h)
            self.donors = np.asarray([inverse[d] for d, _ in donor_h], dtype=np.int64)
            self.hydrogens = np.asarray([inverse[h] for _, h in donor_h], dtype=np.int64)
            self.acceptors = np.asarray([
                src for src, tpr_atom in enumerate(mapping)
                if elements[src].capitalize() in ("N", "O", "S")
                and charges[tpr_atom] <= acceptor_charge_max
            ], dtype=np.int64)
            if not np.any(np.isin(elements.astype(str), ("H", "D", "T"))):
                raise ValueError("No explicit hydrogen coordinates in source XTC topology")
            if not len(self.donors):
                raise ValueError("No chemically defined donor-hydrogen pairs in source")
            self.molecules = molecule_ids[mapping]
            self.solvent = is_solvent
            self.periodic = periodic
            self.distance_cutoff = float(distance_cutoff)
            self.angle_cutoff = float(angle_cutoff)
            self.category_status = {
                "intramolecular": "complete",
                "intermolecular": "complete",
                "solvent_molecular": "complete" if is_solvent.any() else "unavailable",
            }
            self.info = {
                "schema": 1, "definition_version": "HB-v1", "source_identity": source_identity, "frames": frames,
                "chunk_frames": chunk_frames, "tpr_sha256": fingerprint(tpr)["sha256"],
                "mapping_sha256": digest(list(source_to_tpr)),
                "source_elements_sha256": digest(elements.tolist()),
                "solvent_resnames": solvent_names, "periodic": periodic,
                "distance_cutoff_angstrom": self.distance_cutoff,
                "angle_cutoff_degree": self.angle_cutoff,
                "acceptor_rule": "TPR charge <= threshold; element N/O/S",
                "acceptor_charge_max": float(acceptor_charge_max),
                "donor_rule": "typed TPR D-H (including SETTLE), D element N/O/S",
                "categories": self.category_status,
            }
            self.identity = digest(self.info)
            self.directory.mkdir(parents=True, exist_ok=True)
            self.atom_path = self.directory / "atoms.parquet"
            # Source atom indices, including solvent and H, are the stable IDs
            # used in events. Map back to A via a_point when available.
            a_point = 0
            rows = []
            for source, tpr_atom in enumerate(mapping):
                heavy = elements[source].capitalize() not in ("H", "D", "T")
                point = a_point if heavy and not is_solvent[source] else None
                if point is not None:
                    a_point += 1
                rows.append({
                    "source_atom": source, "tpr_atom": int(tpr_atom),
                    "a_point": point, "name": str(names[source]),
                    "element": str(elements[source]), "residue_name": str(resnames[source]),
                    "molecule": int(self.molecules[source]),
                    "is_solvent": bool(is_solvent[source]),
                })
            atoms = pa.Table.from_pylist(rows)
            if self.atom_path.exists():
                if not pq.read_table(self.atom_path).equals(atoms):
                    raise ValueError("Hydrogen-bond source atom mapping differs from existing data")
            else:
                atomic_parquet(self.atom_path, atoms)
            self.info["atoms_sha256"] = fingerprint(self.atom_path)["sha256"]
            self.identity = digest(self.info)
            if self.complete_path.exists():
                self.state = read_json(self.complete_path)
                self._verify_state(self.state, completed=True)
                self.complete = True
                self._register()
            elif self.progress_path.exists():
                self.state = read_json(self.progress_path)
                self._verify_state(self.state, completed=False)
                self.complete = False
            else:
                self.state = {"schema": 1, "info": self.info, "identity": self.identity,
                              "committed_frames": 0, "parts": []}
                atomic_json(self.progress_path, self.state)
                self.complete = False
        except BaseException:
            self.lock.release()
            raise

    def _verify_state(self, state, *, completed):
        if (state.get("schema") != 1 or state.get("identity") != self.identity
                or state.get("info") != self.info
                or type(state.get("committed_frames")) is not int
                or not 0 <= state["committed_frames"] <= self.frames):
            raise IOError("Hydrogen-bond identity or progress mismatch")
        if completed and state["committed_frames"] != self.frames:
            raise IOError("Incomplete hydrogen-bond manifest")
        position = 0
        for index, part in enumerate(state["parts"]):
            end = min(position + self.chunk_frames, self.frames)
            if part.get("start") != position or part.get("stop") != end:
                raise IOError("Hydrogen-bond part boundary mismatch")
            for category in CATEGORIES:
                item = part["categories"][category]
                relative = f"{category}/part-{index:08d}.parquet"
                if item["path"] != relative:
                    raise IOError("Hydrogen-bond part path mismatch")
                path = self.directory / relative
                if not path.is_file() or fingerprint(path)["sha256"] != item["sha256"]:
                    raise IOError("Hydrogen-bond part checksum mismatch")
                table = pq.read_table(path)
                if not table.schema.equals(SCHEMA) or len(table) != item["rows"]:
                    raise IOError("Hydrogen-bond part schema/row mismatch")
                if len(table):
                    frames = table["frame"].to_numpy()
                    if frames.min() < position or frames.max() >= end:
                        raise IOError("Hydrogen-bond event outside partition")
            position = end
        if position != state["committed_frames"]:
            raise IOError("Hydrogen-bond progress does not match verified parts")

    def has(self, start, stop):
        if start < self.state["committed_frames"]:
            if stop <= self.state["committed_frames"]:
                return True
            raise ValueError("Hydrogen-bond chunk crosses committed boundary")
        if start != self.state["committed_frames"] or stop != min(start + self.chunk_frames, self.frames):
            raise ValueError("Hydrogen-bond chunks must follow the original conversion layout")
        return False

    def append(self, start, stop, frames):
        """Consume one source frame at a time; never retain all solvent coordinates."""
        if self.complete or self.has(start, stop):
            return
        from .hb_column_batch import ColumnBatch
        records = ColumnBatch(SCHEMA, CATEGORIES)
        observed = 0
        for offset, (xyz, box) in enumerate(frames):
            if offset >= stop-start or len(xyz) != len(self.molecules):
                raise ValueError("Hydrogen-bond source frame axes mismatch")
            observed += 1
            dims = box_dimensions(box) if self.periodic else None
            if not len(self.acceptors):
                continue
            matches, distances = capped_distance(
                xyz[self.donors], xyz[self.acceptors],
                max_cutoff=self.distance_cutoff, min_cutoff=0.0,
                box=dims, return_distances=True)
            if not len(matches):
                continue
            d_indices = matches[:, 0]
            a_indices = matches[:, 1]
            d = self.donors[d_indices]
            h = self.hydrogens[d_indices]
            a = self.acceptors[a_indices]
            # D-H-A is undefined if D=A, or if H and A are the same atom.
            keep = (d != a) & (h != a)
            if not np.any(keep):
                continue
            d, h, a = d[keep], h[keep], a[keep]
            distances = distances[keep]
            angles = np.degrees(calc_angles(xyz[d], xyz[h], xyz[a], box=dims))
            records.append(start + offset, d, h, a, distances, angles,
                           self.solvent, self.molecules, self.angle_cutoff)
        if observed != stop-start:
            raise ValueError("Hydrogen-bond source frames are incomplete")
        index = len(self.state["parts"])
        part = {"start": start, "stop": stop, "categories": {}}
        for category in CATEGORIES:
            table = records.table(category)
            relative = f"{category}/part-{index:08d}.parquet"
            path = self.directory / relative
            if path.exists():
                if not pq.read_table(path).equals(table):
                    raise IOError("Uncommitted hydrogen-bond part differs; refusing overwrite")
            else:
                from .hb_parquet import atomic_events
                atomic_events(path, table)
            part["categories"][category] = {
                "path": relative, "sha256": fingerprint(path)["sha256"], "rows": len(table)}
        self.state["parts"].append(part)
        self.state["committed_frames"] = stop
        atomic_json(self.progress_path, self.state)

    def _register(self):
        with self.workspace.catalog() as con:
            old = con.execute("SELECT info FROM hydrogen_bond_sets WHERE trajectory=?",
                              [self.trajectory]).fetchone()
            value = canonical({"identity": self.identity, "info": self.info})
            if old and old[0] != value:
                raise ValueError("Different hydrogen-bond dataset already registered")
            con.execute("INSERT INTO hydrogen_bond_sets VALUES (?, ?) ON CONFLICT DO NOTHING",
                        [self.trajectory, value])

    def finish(self):
        if self.complete:
            return
        if self.state["committed_frames"] != self.frames:
            raise ValueError("Hydrogen-bond frames are incomplete")
        self._verify_state(self.state, completed=True)
        atomic_json(self.complete_path, self.state)
        self._register()
        self.progress_path.unlink(missing_ok=True)
        self.complete = True

    def close(self):
        self.lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def hydrogen_bond_info(workspace, trajectory):
    """Read and verify an independently completed hydrogen-bond dataset."""
    directory = workspace.trajectory_path(trajectory) / "hydrogen_bonds"
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    with workspace.lock("trajectory-hydrogen-bonds-" + trajectory):
        record = read_json(path)
        from .registry import Registry
        manifest = Registry(workspace).manifest(trajectory)
        if record["info"]["source_identity"] != manifest["identity"]:
            raise IOError("Hydrogen-bond/A provenance mismatch")
        atoms = directory / "atoms.parquet"
        if not atoms.is_file() or fingerprint(atoms)["sha256"] != record["info"]["atoms_sha256"]:
            raise IOError("Hydrogen-bond source mapping missing or corrupt")
        for part in record["parts"]:
            for category in CATEGORIES:
                info = part["categories"][category]
                name = directory / info["path"]
                if not name.is_file() or fingerprint(name)["sha256"] != info["sha256"]:
                    raise IOError("Hydrogen-bond part missing or corrupt")
                if pq.read_metadata(name).num_rows != info["rows"]:
                    raise IOError("Hydrogen-bond rows differ")
        return {
            "identity": record["identity"], "frames": record["committed_frames"],
            "categories": record["info"]["categories"],
            "rows": {category: sum(part["categories"][category]["rows"]
                                    for part in record["parts"]) for category in CATEGORIES},
        }


def hydrogen_bond_events(workspace, trajectory, category, *,
                         start=0, stop=None, limit=100):
    """Return a bounded logical selection from a completed, verified category."""
    if category not in CATEGORIES:
        raise ValueError("Unknown hydrogen-bond category")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be 1..1000")
    info = hydrogen_bond_info(workspace, trajectory)
    if info is None:
        raise LookupError("No completed hydrogen-bond dataset")
    if info["categories"][category] != "complete":
        raise LookupError("This category was unavailable in the source trajectory")
    last = info["frames"] if stop is None else stop
    if type(start) is not int or type(last) is not int or not 0 <= start <= last <= info["frames"]:
        raise ValueError("Invalid hydrogen-bond frame selection")
    directory = workspace.trajectory_path(trajectory) / "hydrogen_bonds"
    record = read_json(directory / "manifest.json")
    paths = [str(directory / p["categories"][category]["path"])
             for p in record["parts"] if p["start"] < last and p["stop"] > start]
    if not paths:
        return []
    with workspace.catalog() as con:
        relation = con.read_parquet(paths, union_by_name=True, hive_partitioning=False)
        return relation.filter(f"frame >= {start} AND frame < {last}").limit(limit).to_arrow_table().to_pylist()


def recover_hydrogen_bonds(workspace):
    """Re-register verified complete HB sidecars after interrupted DB publication."""
    recovered = 0
    for marker in (workspace.path / "trajectories").glob("*/hydrogen_bonds/manifest.json"):
        trajectory = marker.parent.parent.name
        if hydrogen_bond_info(workspace, trajectory) is None:
            continue
        record = read_json(marker)
        value = canonical({"identity": record["identity"], "info": record["info"]})
        with workspace.catalog() as con:
            old = con.execute("SELECT info FROM hydrogen_bond_sets WHERE trajectory=?",
                              [trajectory]).fetchone()
            if old and old[0] != value:
                raise ValueError("Conflicting hydrogen-bond catalog record")
            con.execute("INSERT INTO hydrogen_bond_sets VALUES (?, ?) ON CONFLICT DO NOTHING",
                        [trajectory, value])
        recovered += 1
    return recovered


def reconstruct_hydrogen_bonds(workspace, trajectory, gro, xtc, tpr,
                               source_to_tpr, solvent_resnames, *,
                               periodic=True, chunk_frames=None, atom_elements=None):
    """Backfill from original full-atom inputs, not from heavy-atom A."""
    from .registry import Registry
    manifest = Registry(workspace).manifest(trajectory)
    gro_id = ("gro" if Path(gro).suffix.lower() == ".gro" else "pdb")
    if (fingerprint(gro) != manifest["options"]["source"][gro_id]
            or fingerprint(xtc) != manifest["options"]["source"]["xtc"]):
        raise ValueError("Backfill source differs from the original A inputs")
    universe = mda.Universe(str(gro), str(xtc))
    try:
        if atom_elements is None:
            universe.guess_TopologyAttrs(to_guess=["elements"])
            elements = np.asarray(universe.atoms.elements, dtype=str)
        else:
            elements = np.asarray(atom_elements, dtype=str)
        elements = np.asarray([e.strip().capitalize() for e in elements])
        if (len(elements) != len(universe.atoms)
                or digest(elements.tolist()) != manifest["options"]["elements_sha256"]):
            raise ValueError("Backfill elements differ from original A input")
        frames = len(universe.trajectory)
        if frames != manifest["frames"]:
            raise ValueError("Backfill frame count differs from A")
        size = chunk_frames or manifest["options"]["chunk_frames"]
        if size != manifest["options"]["chunk_frames"]:
            raise ValueError("Backfill requires original conversion chunk length")
        writer = HydrogenBondStore(
            workspace, trajectory, source_identity=manifest["identity"], frames=frames,
            tpr=tpr, source_to_tpr=source_to_tpr, elements=elements,
            names=universe.atoms.names, resnames=universe.atoms.resnames,
            solvent_resnames=solvent_resnames, chunk_frames=size, periodic=periodic)
        try:
            for start in range(0, frames, size):
                stop = min(start+size, frames)
                if writer.has(start, stop):
                    continue
                def source_frames():
                    for index in range(start, stop):
                        ts = universe.trajectory[index]
                        box = (ts.triclinic_dimensions.copy()
                               if ts.triclinic_dimensions is not None
                               else np.full((3, 3), np.nan))
                        yield ts.positions.copy(), box
                writer.append(start, stop, source_frames())
            writer.finish()
        finally:
            writer.close()
    finally:
        universe.trajectory.close()
    return hydrogen_bond_info(workspace, trajectory)
