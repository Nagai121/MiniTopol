"""Immutable, checksum-verified derived cache kept outside published results."""
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp

import pyarrow as pa


def digest(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


class DerivedStore:
    def __init__(self, path):
        self.path = Path(path)

    def location(self, kind, version):
        encoded = json.dumps(['v1', kind, version], separators=(',', ':')).encode()
        return self.path / kind / sha256(encoded).hexdigest()

    def publish(self, kind, version, tables):
        dest = self.location(kind, version)
        if dest.exists():
            self.check(kind, version)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(mkdtemp(prefix='.building-', dir=dest.parent))
        payload = staging / 'data.arrow'
        try:
            tables = iter(tables)
            first = next(tables)
            with pa.OSFile(str(payload), 'wb') as sink, pa.ipc.new_file(sink, first.schema) as writer:
                writer.write_table(first)
                for table in tables:
                    writer.write_table(table)
            with payload.open('r+b') as stream:
                os.fsync(stream.fileno())
            with (staging/'manifest.json').open('w', encoding='utf-8') as marker:
                json.dump({'file':payload.name,'bytes':payload.stat().st_size,
                           'sha256':digest(payload)}, marker)
                marker.flush()
                os.fsync(marker.fileno())
            try:
                staging.rename(dest)
            except OSError:
                if not dest.is_dir():
                    raise
                self.check(kind, version)
            return dest
        finally:
            if staging.exists():
                if staging.resolve().parent != dest.parent.resolve() or not staging.name.startswith('.building-'):
                    raise RuntimeError('Unexpected temporary cache location')
                shutil.rmtree(staging)

    def check(self, kind, version):
        folder = self.location(kind, version)
        marker = json.loads((folder/'manifest.json').read_bytes())
        name = marker['file']
        if name != 'data.arrow':
            raise IOError('Unknown derived cache format')
        path = folder/name
        if path.stat().st_size != marker['bytes'] or digest(path) != marker['sha256']:
            raise IOError('Derived cache checksum differs')
        return path

    def tables(self, kind, version):
        path = self.check(kind, version)
        with pa.memory_map(str(path), 'r') as source:
            reader = pa.ipc.open_file(source)
            return tuple(pa.Table.from_batches([reader.get_batch(i)], schema=reader.schema)
                         for i in range(reader.num_record_batches)) or (reader.read_all(),)
