"""Rebuildable lossless Zstd scratch with compressed and raw byte checks."""
import hashlib,io
from contextlib import contextmanager
import numpy as np
import pyarrow as pa

class _HashingOutput(io.RawIOBase):
    def __init__(self,path):
        super().__init__();self.raw=path.open('wb',buffering=16384);self.hasher=hashlib.sha256();self.position=0
    def writable(self):return True
    def tell(self):return self.position
    def write(self,data):
        count=self.raw.write(data)
        if count!=len(data):raise IOError('Short compressed contact scratch write')
        self.hasher.update(data);self.position+=count;return count
    def flush(self):
        if not self.raw.closed:self.raw.flush()
    def close(self):
        if not self.closed:
            try:super().close()
            finally:self.raw.close()

class CompressedScratch:
    def __init__(self,path):
        self.sink=_HashingOutput(path)
        try:self.stream=pa.CompressedOutputStream(self.sink,'zstd')
        except BaseException:self.sink.close();raise
    def write(self,data):
        if self.stream is None:raise ValueError('Contact scratch is closed')
        count=self.stream.write(data)
        if count!=len(data):raise IOError('Short raw contact scratch write')
        return count
    def close(self):
        if self.stream is not None:
            try:self.stream.close()
            finally:
                self.stream=None;self.sink.close()
        self.compressed_bytes=self.sink.position
        self.compressed_sha256=self.sink.hasher.digest()

@contextmanager
def open_reader(path):
    # Explicitly own both handles; a closed decoder must not leave a Windows
    # file handle alive until garbage collection.
    with pa.OSFile(str(path),'rb') as raw:
        with pa.CompressedInputStream(raw,'zstd') as stream:
            yield stream

def verify(path,writer,raw_bytes,raw_digest,itemsize):
    if raw_bytes%itemsize or path.stat().st_size!=writer.compressed_bytes:
        raise IOError('Spatial contact scratch length mismatch')
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').digest()
    if actual!=writer.compressed_sha256:
        raise IOError('Spatial contact scratch checksum mismatch')
    digest=hashlib.sha256();count=0
    with open_reader(path) as stream:
        while True:
            data=stream.read(1024*1024)
            if not data:break
            digest.update(data);count+=len(data)
    if count!=raw_bytes or digest.digest()!=raw_digest:
        raise IOError('Spatial contact scratch checksum mismatch')

def read_records(stream,count,dtype):
    wanted=count*dtype.itemsize;parts=[];received=0
    while received<wanted:
        data=stream.read(wanted-received)
        if not data:break
        parts.append(data);received+=len(data)
    if received%dtype.itemsize:raise IOError('Spatial contact scratch length mismatch')
    raw=parts[0] if len(parts)==1 else b''.join(parts)
    return np.frombuffer(raw,dtype=dtype)
