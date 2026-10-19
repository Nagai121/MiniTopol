"""Bounded coordinate preparation and shared A metadata; no source-file reads."""

from importlib.resources import files
import json

import numpy as np
import zarr

from ..coordinates import fit_positions
from ..registry import Registry
from ..storage import atomic_json, digest, read_json


PROTEIN_NAMES = frozenset("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL HID HIE HIP HSD HSE HSP ASH GLH CYX CYM LYN MSE".split())


class AnalysisContext:
    def __init__(self, workspace, trajectory, *, coordinate_mode, initial_whole=False,
                 half_box_steps=False, memory_mb=128, chunk_frames=None, pair_block=32,
                 fft_block=None):
        if not isinstance(memory_mb, (int, float)) or not np.isfinite(memory_mb) or memory_mb <= 0:
            raise ValueError("memory_mb must be positive and finite")
        for value in (pair_block, *([] if fft_block is None else [fft_block])):
            if type(value) is not int or value <= 0:
                raise ValueError("Pair and FFT block sizes must be positive integers")
        if chunk_frames is not None and (type(chunk_frames) is not int or chunk_frames <= 0):
            raise ValueError("chunk_frames must be a positive integer")
        if coordinate_mode not in ("nonperiodic", "continuous", "nojump"):
            raise ValueError("Choose an explicit coordinate_mode: nonperiodic, continuous, nojump")
        if type(initial_whole) is not bool or type(half_box_steps) is not bool:
            raise ValueError("PBC assumptions must be explicitly boolean")
        self.workspace, self.trajectory = workspace, trajectory
        self.registry = Registry(workspace)
        self.manifest = self.registry.manifest(trajectory)
        self.n, self.m = self.manifest["frames"], self.manifest["atoms"]
        self.budget = int(memory_mb * 1024**2)
        max_frames = self.budget // (self.m * 3 * 8 * 8)
        if max_frames < 1:
            raise ValueError("Memory budget is smaller than the working space for one complete frame")
        self.chunk = min(chunk_frames or self.manifest["options"]["chunk_frames"], max_frames)
        self.pair_block = min(pair_block, max(1, int(np.sqrt(self.budget / (self.chunk * 3 * 8 * 8)))))
        self.fft_block = min(fft_block or self.n, max(1, self.budget // (8 * 64)))
        self.mode, self.initial_whole, self.half_box_steps = coordinate_mode, initial_whole, half_box_steps
        self.atoms = self.registry.atoms(trajectory).to_pydict()
        self.residues = sorted(set(self.atoms["residue_index"]))
        self.res_atoms = {r: [] for r in self.residues}
        for i, r in enumerate(self.atoms["residue_index"]):
            self.res_atoms[r].append(i)
        self.annotation, self.res_annotation = self.registry.annotations(trajectory)
        self.properties = json.loads(files("minitopol").joinpath("data/atomic-properties.json").read_text())
        self.parameters = {"coordinate_mode": coordinate_mode, "initial_whole": bool(initial_whole),
                           "half_box_steps": bool(half_box_steps), "annotation": self.annotation,
                           "atomic_properties": digest(self.properties), "scope": "all A heavy atoms; all frames",
                           "ca_policy": "explicit indices or unique CA/C in a recognized protein residue",
                           "protein_residue_names": sorted(PROTEIN_NAMES), "alignment": "C-alpha equal-weight proper Kabsch to frame 0",
                           "precision": "float64", "definition_set": "A-1"}
        self.layout = {"chunk_frames": self.chunk, "pair_block": self.pair_block, "fft_block": self.fft_block}
        key = digest([trajectory, self.manifest["identity"], self.parameters, self.layout])
        self.cache_path = workspace.path / "work" / key
        self.cache_lock = workspace.lock("analysis-cache-" + key)
        self.cache = None
        self._stats = None

    def ca_indices(self):
        explicit = self.annotation["alpha_carbons"]
        if explicit is not None:
            return np.asarray(explicit, dtype=np.int64)
        ca = []
        for residue, points in self.res_atoms.items():
            candidates = [i for i in points if self.atoms["name"][i] == "CA" and self.atoms["element"][i] == "C"]
            if self.atoms["residue_name"][points[0]] not in PROTEIN_NAMES:
                if candidates:
                    raise ValueError("Ambiguous C-alpha role in a nonstandard residue; annotate alpha_carbons explicitly")
                continue
            if len(candidates) > 1:
                raise ValueError("Duplicate C-alpha atoms; explicit atom-role annotation is required")
            ca.extend(candidates)
        return np.asarray(ca, dtype=np.int64)

    def atomic_values(self, kind):
        overrides = self.annotation[kind]
        values = []
        for i, element in enumerate(self.atoms["element"]):
            value = overrides.get(str(i), self.properties[kind].get(element))
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Undefined {kind} for A point {i}, element {element}; supply an explicit override")
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    def coordinate_blocks(self):
        return self.registry.coordinate_blocks(self.trajectory, mode=self.mode,
                                              initial_whole=self.initial_whole,
                                              half_box_steps=self.half_box_steps, chunk_frames=self.chunk)

    def prepare_aligned(self):
        if self.cache is not None:
            return
        ca = self.ca_indices()
        if len(ca) < 3:
            raise ValueError("At least three non-collinear C-alpha atoms are required for aligned descriptors")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        marker = self.cache_path / "state"
        state = read_json(marker) if marker.exists() else {"committed": 0, "layout": self.layout}
        if state["layout"] != self.layout:
            raise ValueError("Coordinate cache layout mismatch")
        cache = zarr.open_group(str(self.cache_path / "aligned.zarr"), mode="a", zarr_format=3)
        for name, shape, chunks in (("positions", (self.n, self.m, 3), (self.chunk, min(32, self.m), 3)),
                                    ("time", (self.n,), (self.chunk,))):
            if name not in cache:
                if state["committed"]:
                    raise IOError("Committed coordinate cache data is missing")
                cache.create_array(name, shape=shape, chunks=chunks, dtype="f8")
            elif cache[name].shape != shape:
                raise IOError("Coordinate cache shape mismatch")
        if state["committed"] < self.n:
            reference = None
            for block in self.coordinate_blocks():
                if reference is None:
                    reference = block.positions[0].copy()
                    fit_positions(reference, reference, ca)
                if block.stop <= state["committed"]:
                    continue
                if block.start != state["committed"]:
                    raise IOError("Coordinate cache commit boundary mismatch")
                aligned = np.empty_like(block.positions)
                for index, frame in enumerate(block.positions):
                    aligned[index] = fit_positions(frame, reference, ca)
                cache["positions"][block.start:block.stop] = aligned
                cache["time"][block.start:block.stop] = block.time
                state["committed"] = block.stop
                atomic_json(marker, state)
        self.cache = cache

    def read_aligned(self, start, stop, indices=None):
        self.prepare_aligned()
        if indices is None:
            return self.cache["positions"][start:stop]
        return self.cache["positions"].oindex[start:stop, np.asarray(indices, dtype=np.int64), :]

    def aligned_blocks(self):
        self.prepare_aligned()
        for start in range(0, self.n, self.chunk):
            stop = min(start + self.chunk, self.n)
            yield start, stop, self.read_aligned(start, stop), self.cache["time"][start:stop]

    def statistics(self):
        if self._stats is None:
            mean = np.zeros((self.m, 3))
            m2 = np.zeros_like(mean)
            count = 0
            for _, _, positions, _ in self.aligned_blocks():
                for frame in positions:
                    count += 1
                    delta = frame - mean
                    mean += delta / count
                    m2 += delta * (frame - mean)
            self._stats = mean, np.maximum(m2, 0) / self.n
        return self._stats

    def timestep(self):
        self.prepare_aligned()
        if self.n == 1:
            return 0.0
        first, last = float(self.cache["time"][0]), float(self.cache["time"][-1])
        step = (last - first) / (self.n - 1)
        tolerance = max(1e-6, 4 * np.finfo(np.float32).eps * max(abs(first), abs(last)))
        previous = None
        if step <= 0:
            raise ValueError("Time differences require increasing, regularly sampled timestamps")
        for start in range(0, self.n, self.chunk):
            time = self.cache["time"][start:min(start + self.chunk, self.n)]
            if (previous is not None and time[0] <= previous) or np.any(np.diff(time) <= 0):
                raise ValueError("Duplicate or reversed timestamps are unsupported for time-lag descriptors")
            if np.any(np.abs(time - (first + np.arange(start, start + len(time)) * step)) > tolerance):
                raise ValueError("Irregular timestamps require a separately defined lag aggregation; no resampling was performed")
            previous = time[-1]
        return step
