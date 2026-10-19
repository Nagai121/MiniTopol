import pyarrow as pa

from .kernels import backbone_angles, backbone_quads, secondary_labels


SCHEMA = pa.schema([("frame", pa.int64()), ("time_ps", pa.float64()), ("residue", pa.int64()),
                    ("value", pa.string()), ("reason", pa.string())])


def calculate(ctx, writer):
    quads, reasons = backbone_quads(ctx)
    for block in ctx.registry.blocks(ctx.trajectory, chunk_frames=ctx.chunk):
        for offset, xyz in enumerate(block.positions):
            frame = block.start + offset
            if writer.has(frame):
                continue
            angles, why = backbone_angles(xyz, block.box[offset], quads, reasons, periodic=ctx.mode != "nonperiodic")
            labels = secondary_labels(angles["phi"], angles["psi"])
            rows = [{"frame": frame, "time_ps": block.time[offset], "residue": r,
                     "value": labels[i], "reason": (why["phi"][i] or why["psi"][i]) if labels[i] is None else None}
                    for i, r in enumerate(ctx.residues)]
            writer.append(frame, pa.Table.from_pylist(rows, schema=SCHEMA))
