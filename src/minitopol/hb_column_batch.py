"""One publication chunk of HB events as typed columns, preserving tuple order."""
import numpy as np
import pyarrow as pa

class ColumnBatch:
    def __init__(self, schema, categories):
        self.schema = schema
        self.columns = {name: [[] for _ in schema] for name in categories}

    def append(self, frame, donor, hydrogen, acceptor, distances, angles, solvent, molecules, cutoff):
        valid = np.isfinite(angles) & (angles >= cutoff)
        donor, hydrogen, acceptor = donor[valid], hydrogen[valid], acceptor[valid]
        distances, angles = distances[valid], angles[valid]
        ds, aa = solvent[donor], solvent[acceptor]
        same = molecules[donor] == molecules[acceptor]
        categories = {
            'solvent_molecular': ds != aa,
            'intramolecular': ~ds & ~aa & same,
            'intermolecular': ~ds & ~aa & ~same,
        }
        for category, mask in categories.items():
            count = int(np.count_nonzero(mask))
            if not count:
                continue
            values = (np.full(count, frame, dtype='i8'), donor[mask], hydrogen[mask],
                      acceptor[mask], distances[mask], angles[mask])
            for parts, value in zip(self.columns[category], values):
                parts.append(value)

    def table(self, category):
        parts = self.columns[category]
        if not parts[0]:
            return pa.Table.from_batches([], schema=self.schema)
        columns = [np.concatenate(items) for items in parts]
        # Python's previous sorted(tuple6) order, including distance/angle ties.
        order = np.lexsort(tuple(reversed(columns)))
        return pa.Table.from_arrays([pa.array(values[order], type=field.type)
                                     for values, field in zip(columns, self.schema)], schema=self.schema)
