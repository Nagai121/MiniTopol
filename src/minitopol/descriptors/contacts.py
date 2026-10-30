import numpy as np
import pyarrow as pa
from MDAnalysis.lib.distances import distance_array

from ..coordinates import box_dimensions


SCHEMA = pa.schema([("level", pa.string()), ("frame", pa.int64()), ("time_ps", pa.float64()),
                    ("residue_i", pa.int64()), ("residue_j", pa.int64()), ("distance_angstrom", pa.float64()),
                    ("contact_frames", pa.int64()), ("frames", pa.int64()), ("value", pa.float64())])


def eligible(ctx, r, s):
    if r == s:
        return False
    a, b = ctx.res_annotation.get(r, {}), ctx.res_annotation.get(s, {})
    return not (a.get("run") is not None and a.get("run") == b.get("run") and
                abs(a["sequence_position"] - b["sequence_position"]) <= 2)


def minimum_residue_distance(ctx, xyz, a, b, box):
    best = np.inf
    for i in range(0, len(a), 128):
        for j in range(0, len(b), 128):
            best = min(best, float(distance_array(xyz[a[i:i+128]], xyz[b[j:j+128]], box=box).min()))
    return best


def calculate(ctx, writer):
    if ctx.mode != 'nonperiodic':
        from .periodic_contacts import calculate as periodic, UseOriginal
        try:
            return periodic(ctx, writer)
        except UseOriginal:
            if any(str(key).startswith('state-compact-') for key in writer.by_key):
                raise ValueError('Resume periodic spatial parts only with their original supported inputs and memory layout')
            pass
    if ctx.mode == 'nonperiodic':
        try:
            from .spatial_contacts import calculate as spatial, UseOriginal
        except ImportError:
            pass
        else:
            try:
                return spatial(ctx, writer)
            except UseOriginal:
                pass
    return _calculate_original(ctx, writer)


def _calculate_original(ctx, writer):
    from .context import PROTEIN_NAMES
    protein = {r for r, points in ctx.res_atoms.items() if ctx.atoms["residue_name"][points[0]] in PROTEIN_NAMES}
    protein.update(ctx.atoms["residue_index"][i] for i in (ctx.annotation["alpha_carbons"] or []))
    if any(ctx.res_annotation.get(r, {}).get("run") is None for r in protein):
        raise ValueError("Contact neighbor exclusions require explicit continuous-segment annotations for protein residues")
    for left in range(0, len(ctx.residues), ctx.pair_block):
        a = ctx.residues[left:left+ctx.pair_block]
        for right in range(left, len(ctx.residues), ctx.pair_block):
            b = ctx.residues[right:right+ctx.pair_block]
            key = f"frequency-{left}-{right}"
            if writer.has(key):
                continue
            pairs = [(r, s) for r in a for s in b if r < s and eligible(ctx, r, s)]
            if not pairs:
                writer.append(key, pa.Table.from_pylist([], schema=SCHEMA))
                continue
            counts = {pair: 0 for pair in pairs}
            try:
                from .contact_reduction import prepare, frame_minima
            except ImportError:
                plan = None
            else:
                plan = prepare(ctx.res_atoms, pairs, len(ctx.atoms['residue_index']))
            for block in ctx.registry.blocks(ctx.trajectory, chunk_frames=ctx.chunk):
                rows, part = [], 0
                for offset, xyz in enumerate(block.positions):
                    box = box_dimensions(block.box[offset]) if ctx.mode != "nonperiodic" else None
                    minima = frame_minima(xyz, box, plan, ctx.budget) if plan is not None else None
                    for pair_index, (r, s) in enumerate(pairs):
                        distance = (minimum_residue_distance(ctx, xyz, ctx.res_atoms[r], ctx.res_atoms[s], box)
                                    if minima is None else float(minima[pair_index]))
                        if distance <= 4.5:
                            counts[r, s] += 1
                            rows.append({"level": "state", "frame": block.start + offset, "time_ps": block.time[offset],
                                         "residue_i": r, "residue_j": s, "distance_angstrom": distance,
                                         "contact_frames": None, "frames": ctx.n, "value": 1.})
                            if len(rows) >= 4096:
                                writer.append(f"state-{left}-{right}-{block.start}-{part}", pa.Table.from_pylist(rows, schema=SCHEMA))
                                rows, part = [], part + 1
                writer.append(f"state-{left}-{right}-{block.start}-{part}", pa.Table.from_pylist(rows, schema=SCHEMA))
            rows = [{"level": "frequency", "frame": None, "time_ps": None, "residue_i": r, "residue_j": s,
                     "distance_angstrom": None, "contact_frames": count, "frames": ctx.n,
                     "value": count / ctx.n} for (r, s), count in counts.items() if count]
            writer.append(key, pa.Table.from_pylist(rows, schema=SCHEMA))
