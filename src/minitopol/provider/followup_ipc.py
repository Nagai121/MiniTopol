"""Defer IPC encoding until the cache has allocated the final segment."""
import pyarrow as pa


class SegmentIPC:
    def __init__(self, table):
        self.schema = table.schema
        self.batches = table.to_batches()
        # The serialized schema is followed by the eight-byte stream end marker.
        self.size = self.schema.serialize().size + 8 + sum(
            pa.ipc.get_record_batch_size(batch) for batch in self.batches)

    def __len__(self):
        return self.size

    def write(self, segment):
        with pa.FixedSizeBufferWriter(pa.py_buffer(segment.buf)) as sink:
            with pa.ipc.new_stream(sink, self.schema) as stream:
                for batch in self.batches:
                    stream.write_batch(batch)
            if sink.tell() != self.size:
                raise IOError('Arrow IPC stream size differs from estimate')
