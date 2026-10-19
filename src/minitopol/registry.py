"""Uniform chunk access to A and explicitly registered coarse representations."""

from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import zarr

from .storage import canonical, digest, identifier, read_json


@dataclass(frozen=True)
class FrameBlock:
    start: int
    stop: int
    positions: np.ndarray
    time: np.ndarray
    box: np.ndarray


class Registry:
    def __init__(self, workspace):
        self.workspace = workspace

    def representations(self):
        with self.workspace.catalog() as con:
            return ["A"] + [row[0] for row in con.execute("SELECT id FROM representations ORDER BY id").fetchall()]

    def register(self, name, *, trajectory, version, groups, descriptor_definitions):
        """Register an explicit weighted mapping without generating trajectories.

        groups = [{"name": "bead", "atoms": [0, 1], "weights": [0.5, 0.5]}]
        Indices refer to A, weights must sum to 1. No mass/geometric convention
        is chosen implicitly. Definitions are supplied by the caller.
        """
        identifier(name)
        if name == "A" or not name.startswith("C"):
            raise ValueError("Coarse representation names must start with C; A is reserved")
        if not isinstance(version, str) or not version.strip() or not groups:
            raise ValueError("A nonempty version and at least one point group are required")
        if not isinstance(descriptor_definitions, dict) or not descriptor_definitions:
            raise ValueError("Explicit descriptor definitions are required")
        topology = self.atoms(trajectory)
        topology_identity = self._topology_identity(topology)
        names = set()
        for group in groups:
            if not isinstance(group.get("name"), str) or not group["name"] or group["name"] in names:
                raise ValueError("Point group names must be nonempty and unique")
            names.add(group["name"])
            atoms, weights = group["atoms"], group["weights"]
            if not atoms or len(atoms) != len(weights) or len(set(atoms)) != len(atoms):
                raise ValueError("Each point requires unique atoms and one weight per atom")
            if any(type(x) is not int or x < 0 for x in atoms):
                raise ValueError("A atom indices must be nonnegative integers")
            if max(atoms) >= len(topology):
                raise ValueError("Representation references atoms absent from this A topology")
            if not np.isfinite(weights).all() or not np.isclose(sum(weights), 1, rtol=0, atol=1e-12):
                raise ValueError("Explicit finite weights must sum to one")
        from .descriptors.definitions import DescriptorDefinition
        for key, value in descriptor_definitions.items():
            descriptor = DescriptorDefinition(**value)
            if descriptor.name != key or descriptor.representation != name:
                raise ValueError("Descriptor definitions must match their name and C representation")
        definition = canonical({"schema": 1, "version": version, "groups": groups,
                                "topology_identity": topology_identity,
                                "descriptor_definitions": descriptor_definitions})
        with self.workspace.catalog() as con:
            previous = con.execute("SELECT definition FROM representations WHERE id = ?", [name]).fetchone()
            if previous and previous[0] != definition:
                raise ValueError("Registered definitions are immutable; use a new representation ID")
            con.execute("INSERT INTO representations VALUES (?, ?) ON CONFLICT DO NOTHING", [name, definition])
        return name

    @staticmethod
    def _topology_identity(topology):
        return digest(topology.select(["point", "name", "element", "residue_index", "residue_name", "segment"]).to_pydict())

    def definition(self, name):
        import json
        if name == "A":
            return {"version": "1", "kind": "non-solvent heavy atoms, original coordinates"}
        with self.workspace.catalog() as con:
            row = con.execute("SELECT definition FROM representations WHERE id = ?", [name]).fetchone()
        if not row:
            raise KeyError(name)
        return json.loads(row[0])

    def manifest(self, trajectory):
        result = read_json(self.workspace.trajectory_path(trajectory) / "manifest.json")
        if result["status"] != "complete":
            raise ValueError("Trajectory conversion has not completed")
        return result

    def atoms(self, trajectory):
        self.manifest(trajectory)
        return pq.read_table(self.workspace.trajectory_path(trajectory) / "atoms.parquet")

    def validate_representation(self, trajectory, representation):
        definition = self.definition(representation)
        if representation != "A" and definition["topology_identity"] != self._topology_identity(self.atoms(trajectory)):
            raise ValueError("Representation was registered for a different A topology")
        return definition

    def blocks(self, trajectory, representation="A", *, start=0, stop=None, chunk_frames=None):
        manifest = self.manifest(trajectory)
        stop = manifest["frames"] if stop is None else stop
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= manifest["frames"]:
            raise ValueError("Frame range must lie within the trajectory")
        if chunk_frames is None:
            chunk_frames = manifest["options"]["chunk_frames"]
        if type(chunk_frames) is not int or chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        definition = None if representation == "A" else self.validate_representation(trajectory, representation)
        groups = None if definition is None else definition["groups"]
        if groups and any(max(group["atoms"]) >= manifest["atoms"] for group in groups):
            raise ValueError("Representation references atoms absent from this A topology")
        source = zarr.open_group(str(self.workspace.trajectory_path(trajectory) / "A.zarr"), mode="r")
        for offset in range(start, stop, chunk_frames):
            end = min(offset + chunk_frames, stop)
            positions = source["positions"][offset:end]
            if groups:
                positions = np.stack([
                    np.einsum("faj,a->fj", positions[:, group["atoms"]], group["weights"])
                    for group in groups
                ], axis=1)
            yield FrameBlock(offset, end, positions, source["time"][offset:end], source["box"][offset:end])
