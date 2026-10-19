import pyarrow as pa

from .kernels import backbone_frames, secondary_labels


SCHEMA = pa.schema([("frame", pa.int64()), ("time_ps", pa.float64()), ("residue", pa.int64()),
                    ("value", pa.string()), ("reason", pa.string())])


def calculate(ctx, writer):
    for frame, time, angles, why in backbone_frames(ctx, writer):
        labels = secondary_labels(angles["phi"], angles["psi"])
        rows = [{"frame": frame, "time_ps": time, "residue": r,
                 "value": labels[i], "reason": (why["phi"][i] or why["psi"][i]) if labels[i] is None else None}
                for i, r in enumerate(ctx.residues)]
        writer.append(frame, pa.Table.from_pylist(rows, schema=SCHEMA))
