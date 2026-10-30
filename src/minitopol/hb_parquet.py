"""Standard typed Parquet encoding for unchanged HB-v1 event columns."""
import pyarrow.parquet as pq
from .storage import _publish_parquet, atomic_parquet

def atomic_events(path, table):
    from .hydrogen_bonds import SCHEMA
    if not table.schema.equals(SCHEMA, check_metadata=True):
        return atomic_parquet(path, table)
    encodings = {field.name: ('DELTA_BINARY_PACKED' if i < 4 else 'BYTE_STREAM_SPLIT')
                 for i, field in enumerate(SCHEMA)}
    def write(value, name):
        pq.write_table(value, name, compression='zstd', compression_level=3,
                       use_dictionary=False, column_encoding=encodings)
    _publish_parquet(path, table, write)
