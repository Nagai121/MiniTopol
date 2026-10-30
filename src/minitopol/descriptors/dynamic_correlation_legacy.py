import numpy as np
import pyarrow as pa


SCHEMA = pa.schema([("residue_i", pa.int64()), ("residue_j", pa.int64()),
                    ("frames", pa.int64()), ("value", pa.float64()), ("reason", pa.string())])


def calculate(ctx, writer):
    ca = ctx.ca_indices()
    mean, variance = ctx.statistics()
    written = False
    for left in range(0, len(ca), ctx.pair_block):
        a = ca[left:left+ctx.pair_block]
        for right in range(left, len(ca), ctx.pair_block):
            key = f"{left}-{right}"
            if writer.has(key):
                written = True
                continue
            b = ca[right:right+ctx.pair_block]
            products = np.zeros((len(a), len(b)))
            for start in range(0, ctx.n, ctx.chunk):
                stop = min(start + ctx.chunk, ctx.n)
                x = ctx.read_aligned(start, stop, a) - mean[a]
                y = x if right == left else ctx.read_aligned(start, stop, b) - mean[b]
                products += np.einsum("tic,tjc->ij", x, y)
            rows = []
            for i, point_i in enumerate(a):
                for j, point_j in enumerate(b):
                    if left + i >= right + j:
                        continue
                    denominator = np.sqrt(variance[point_i].sum() * variance[point_j].sum())
                    value = products[i, j] / ctx.n / denominator if denominator > 0 else None
                    if value is not None:
                        if abs(value) > 1 + 1e-10:
                            raise ArithmeticError("Dynamic correlation violates its normalization bound")
                        value = float(np.clip(value, -1, 1))
                    r, s = sorted([ctx.atoms["residue_index"][point_i], ctx.atoms["residue_index"][point_j]])
                    rows.append({"residue_i": r, "residue_j": s, "frames": ctx.n, "value": value,
                                 "reason": "zero_variance" if value is None else None})
            writer.append(key, pa.Table.from_pylist(rows, schema=SCHEMA))
            written = True
    if not written:
        writer.append("empty", pa.Table.from_pylist([], schema=SCHEMA))
