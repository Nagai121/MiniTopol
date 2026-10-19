import numpy as np
import pyarrow as pa

from .kernels import backbone_angles, backbone_quads


SCHEMA = pa.schema([("frame", pa.int64()), ("time_ps", pa.float64()), ("residue", pa.int64()),
                    ("angle", pa.string()), ("value", pa.float64()), ("reason", pa.string())])


def calculate(ctx, writer):
    quads, reasons = backbone_quads(ctx)
    for block in ctx.registry.blocks(ctx.trajectory, chunk_frames=ctx.chunk):
        for offset, xyz in enumerate(block.positions):
            frame = block.start + offset
            if writer.has(frame):
                continue
            values, why = backbone_angles(xyz, block.box[offset], quads, reasons, periodic=ctx.mode != "nonperiodic")
            rows = [{"frame": frame, "time_ps": block.time[offset], "residue": r, "angle": angle,
                     "value": float(values[angle][i]) if np.isfinite(values[angle][i]) else None,
                     "reason": why[angle][i]} for i, r in enumerate(ctx.residues) for angle in ("phi", "psi")]
            writer.append(frame, pa.Table.from_pylist(rows, schema=SCHEMA))
