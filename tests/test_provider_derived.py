import pyarrow as pa

from minitopol.provider.compact_rama import CompactRama, pack
from minitopol.provider.derived_store import DerivedStore


def test_compact_ramachandran_keeps_columns_nulls_and_metadata(tmp_path):
    table = pa.table({
        "frame": [0, 0, 1, 1, 2, 2],
        "time_ps": [0., 0., 10., 10., 20., 20.],
        "residue": [7] * 6,
        "angle": ["phi", "psi"] * 3,
        "value": [None, 0.5, None, -0.25, 1., 2.],
        "reason": ["missing", "", "missing", "", "", ""],
        "trajectory": ["test"] * 6,
    }).replace_schema_metadata({b"minitopol": b"fixture"})
    store = DerivedStore(tmp_path)
    version = ("fixture",)
    store.publish("rama-compact", version, [pack(table)])
    loaded, = store.tables("rama-compact", version)
    import numpy as np
    selected = CompactRama(loaded).select(np.array([0, 2], dtype=np.int64))
    assert selected.equals(table.take([0, 1, 4, 5]), check_metadata=True)
