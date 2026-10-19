import pyarrow as pa

from .kernels import sasa_atoms


SCHEMA = pa.schema([("frame", pa.int64()), ("time_ps", pa.float64()), ("level", pa.string()),
                    ("point", pa.int64()), ("residue", pa.int64()), ("value", pa.float64())])


def calculate(ctx, writer):
    radii = ctx.atomic_values("radii")
    for block in ctx.coordinate_blocks():
        for index, positions in enumerate(block.positions):
            frame = block.start + index
            if writer.has(f"total-{frame}"):
                continue
            values = sasa_atoms(positions, radii)
            rows, part = [], 0
            for point, value in enumerate(values):
                rows.append({"frame": frame, "time_ps": block.time[index], "level": "atom", "point": point,
                             "residue": ctx.atoms["residue_index"][point], "value": value})
                if len(rows) == 4096:
                    writer.append(f"atom-{frame}-{part}", pa.Table.from_pylist(rows, schema=SCHEMA))
                    rows, part = [], part + 1
            if rows:
                writer.append(f"atom-{frame}-{part}", pa.Table.from_pylist(rows, schema=SCHEMA))
            rows = [{"frame": frame, "time_ps": block.time[index], "level": "residue", "point": None,
                     "residue": r, "value": values[points].sum()} for r, points in ctx.res_atoms.items()]
            for offset in range(0, len(rows), 4096):
                writer.append(f"residue-{frame}-{offset}", pa.Table.from_pylist(rows[offset:offset+4096], schema=SCHEMA))
            writer.append(f"total-{frame}", pa.Table.from_pylist([
                {"frame": frame, "time_ps": block.time[index], "level": "total", "point": None,
                 "residue": None, "value": values.sum()}], schema=SCHEMA))
