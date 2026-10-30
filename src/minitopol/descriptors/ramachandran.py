import numpy as np
import pyarrow as pa

from .kernels import backbone_frames


SCHEMA = pa.schema([("frame", pa.int64()), ("time_ps", pa.float64()), ("residue", pa.int64()),
                    ("angle", pa.string()), ("value", pa.float64()), ("reason", pa.string())])


def calculate(ctx, writer):
    for frame, time, values, why in backbone_frames(ctx, writer):
        rows = [{"frame": frame, "time_ps": time, "residue": r, "angle": angle,
                 "value": float(values[angle][i]) if np.isfinite(values[angle][i]) else None,
                 "reason": why[angle][i]} for i, r in enumerate(ctx.residues) for angle in ("phi", "psi")]
        writer.append(frame, pa.Table.from_pylist(rows, schema=SCHEMA))
