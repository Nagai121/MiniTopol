"""Bounded rebuildable Zstd scratch sharing one serial codec across files.

Only used when the original per-file native-context reservation does not fit.
All encoded and raw checks remain mandatory before result publication.
"""
import hashlib
import struct
import threading
from contextlib import contextmanager
import pyarrow as pa
from .contact_scratch import _HashingOutput

MAGIC = b'MTCFS001'
HEADER = struct.Struct('<II')
RAW_BATCH = 65536
MAX_RAW = 2 * 1024**2
MAX_ENCODED = MAX_RAW + 65536
_codec = pa.Codec('zstd', compression_level=3)
_lock = threading.Lock()


def reserve_bytes(streams):
    # Includes shared native contexts, current raw/encoded/decode/read buffers,
    # Python bookkeeping, and all per-file buffered sinks. Not an RSS limit.
    return 24 * 1024**2 + streams * (16384 + RAW_BATCH)


class FramedScratch:
    def __init__(self, path):
        self.sink = _HashingOutput(path)
        self.closed = False
        self.pending = bytearray()
        try:
            self.sink.write(MAGIC)
        except BaseException:
            self.sink.close()
            raise

    def _flush(self):
        if not self.pending:
            return
        raw = bytes(self.pending)
        self.pending.clear()
        with _lock:
            encoded = _codec.compress(raw)
        if not 0 < len(encoded) <= MAX_ENCODED:
            raise IOError('Contact scratch encoded frame exceeds bound')
        self.sink.write(HEADER.pack(len(raw), len(encoded)))
        self.sink.write(encoded)

    def write(self, data):
        if self.closed:
            raise ValueError('Contact scratch is closed')
        data = memoryview(data).cast('B')
        start = 0
        while start < len(data):
            take = min(RAW_BATCH - len(self.pending), len(data) - start)
            self.pending.extend(data[start:start + take]); start += take
            if len(self.pending) == RAW_BATCH:
                self._flush()
        return len(data)

    def close(self):
        if not self.closed:
            try:
                self._flush()
            finally:
                try:
                    self.sink.close()
                finally:
                    self.closed = True
        self.compressed_bytes = self.sink.position
        self.compressed_sha256 = self.sink.hasher.digest()


class _Reader:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = b''
        self.offset = 0
        if stream.read(len(MAGIC)) != MAGIC:
            raise IOError('Contact scratch magic mismatch')

    def read(self, count):
        if not isinstance(count, int) or not 0 <= count <= MAX_RAW:
            raise ValueError('Contact scratch read exceeds bound')
        result = bytearray()
        while len(result) < count:
            if self.offset == len(self.buffer):
                self.buffer = b''
                self.offset = 0
                header = self.stream.read(HEADER.size)
                if not header:
                    break
                if len(header) != HEADER.size:
                    raise IOError('Contact scratch truncated header')
                raw_size, encoded_size = HEADER.unpack(header)
                if not 0 < raw_size <= MAX_RAW or not 0 < encoded_size <= MAX_ENCODED:
                    raise IOError('Contact scratch invalid frame lengths')
                encoded = self.stream.read(encoded_size)
                if len(encoded) != encoded_size:
                    raise IOError('Contact scratch truncated frame')
                with _lock:
                    decoded = _codec.decompress(encoded, raw_size)
                if len(decoded) != raw_size:
                    raise IOError('Contact scratch decoded length mismatch')
                self.buffer = decoded
            take = min(count - len(result), len(self.buffer) - self.offset)
            result.extend(memoryview(self.buffer)[self.offset:self.offset + take])
            self.offset += take
        return bytes(result)


@contextmanager
def open_reader(path):
    with path.open('rb', buffering=16384) as stream:
        reader = _Reader(stream)
        try:
            yield reader
        finally:
            reader.buffer = b''


def verify(path, writer, raw_bytes, raw_digest, itemsize):
    if raw_bytes % itemsize or path.stat().st_size != writer.compressed_bytes:
        raise IOError('Contact scratch length mismatch')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').digest()
    if actual != writer.compressed_sha256:
        raise IOError('Contact scratch encoded checksum mismatch')
    digest = hashlib.sha256()
    count = 0
    with open_reader(path) as stream:
        while True:
            block = stream.read(1024**2)
            if not block:
                break
            digest.update(block)
            count += len(block)
    if count != raw_bytes or digest.digest() != raw_digest:
        raise IOError('Contact scratch raw checksum mismatch')
