from collections import OrderedDict
from hashlib import sha256
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from . import experiments, runtime
from ..descriptors import count_store, virtual_rama_layout
from ..storage import canonical, read_json
from ..descriptors.results import with_result_metadata
from .. import direct_counts, sasa_counts


BaseProvider = runtime.DataProvider


class QueryProvider(experiments.ExperimentalProvider):
    features = frozenset()
    coord_mib = 512
    part_mib = 1536

    def __init__(self, workspace, budget):
        super().__init__(workspace, budget)
        self.decoded_budget = self.coord_mib * 1024**2
        self.part_budget = self.part_mib * 1024**2 if 'parts' in self.features else 0
        self.cache.budget = budget - self.decoded_budget - self.part_budget
        if self.cache.budget <= 0:
            raise ValueError('Invalid cache allocation')
        self.parts = OrderedDict()
        self.part_bytes = 0
        self.skip_coordinate_catalog = 'catalog' in self.features
        self.payload_views = 'views' in self.features

    def _get(self, request, trace=None):
        if trace is not None:
            trace.increment('descriptor_part_budget_bytes', self.part_budget)
        return super()._get(request, trace)

    def _record(self, name, value=1):
        trace = getattr(self.request_context, 'trace', None)
        if trace is not None:
            trace.increment(name, value)

    def _cached_part(self, key, loader):
        if key in self.parts:
            self.parts.move_to_end(key)
            self._record('descriptor_part_hit')
            return self.parts[key][0]
        self._record('descriptor_part_miss')
        started = time.perf_counter_ns()
        value, size = loader()
        self._record('descriptor_part_load_ns_work', time.perf_counter_ns() - started)
        if size <= self.part_budget:
            while self.part_bytes + size > self.part_budget:
                _, (_, removed_size) = self.parts.popitem(last=False)
                self.part_bytes -= removed_size
            self.parts[key] = value, size
            self.part_bytes += size
        return value

    def _paths(self, trajectory, name):
        with self.workspace.catalog() as con:
            return con.execute(
                'SELECT id, path FROM results WHERE trajectory=? AND descriptor=? ORDER BY id',
                [trajectory, name]).fetchall()

    @staticmethod
    def _file_version(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def _rama_tables(self, result_id, relative):
        path = self.workspace.resolve_path(relative)
        marker = path.with_suffix('.json')
        # Results are immutable after publication. Include the completion marker
        # and manifest identity so a new publication cannot reuse an old table.
        detail = getattr(self, '_detail', None) if 'detail' in self.features else None
        started = time.perf_counter_ns() if detail else 0
        info = read_json(marker)
        identity = self.registry.manifest(info['info']['trajectory'])['identity']
        source = self.workspace.trajectory_path(info['info']['trajectory'])
        dependencies = [source / name for name in ('manifest.json', 'atoms.parquet', 'annotations.parquet', 'bonds.json', 'bonds.parquet')]
        dependencies.extend(sorted(item for item in (source / 'A.zarr').rglob('*') if item.is_file()))
        dependencies.extend(path.parent / part['file'] for part in info['parts'])
        dependencies.append((path.parent / info['parts'][0]['file']).parent / 'recipe-axes')
        version = tuple((str(item), self._file_version(item) if item.exists() else None)
                        for item in dependencies)
        key = ('rama', result_id, relative, identity, self._file_version(marker), version)
        if detail:
            detail('rama_revision', started)

        def load():
            started = time.perf_counter_ns() if detail else 0
            def materialize():
                return virtual_rama_layout.logical_tables(self.workspace, relative)

            read = getattr(self, '_derived_rama', None)
            tables = read(key, materialize) if read else tuple(materialize())
            if detail:
                detail('rama_reconstruct', started)
            size = sum(table.get_total_buffer_size() for table in tables)
            if 'rama-index' in self.features:
                tables = self._prepare_rama_tables(tables)
                size += sum(index.nbytes for index in tables)
            return tables, size

        return self._cached_part(key, load)

    def _sasa_tables(self, result_id, relative, frames):
        path = self.workspace.resolve_path(relative)
        marker = path.with_suffix('.json')
        detail = getattr(self, '_detail', None) if 'detail' in self.features else None
        completed = read_json(marker)
        if completed.get('schema') not in (4, 6) or any(
                part.get('hash_kind') != 'native-counts-v1' for part in completed['parts']):
            for table in count_store.logical_tables(
                    self.workspace, relative, start=min(frames), stop=max(frames)+1):
                yield table.filter(pc.is_in(table['frame'], value_set=pa.array(frames, type=pa.int64())))
            return
        frame_list = np.sort(np.asarray(frames, dtype=np.int64))
        for part in completed['parts']:
            first, last = (int(value) for value in part['key'].split('-')[1:])
            lo, hi = np.searchsorted(frame_list, [first, last])
            if lo == hi:
                continue
            source = path.parent / part['file']
            axes = count_store.axes_path(source)
            key = ('sasa', result_id, str(source), self._file_version(marker),
                   self._file_version(source), self._file_version(axes))

            def load():
                raw, raw_axes = source.read_bytes(), axes.read_bytes()
                if (sha256(raw).hexdigest() != part['sha256']
                        or sha256(raw_axes).hexdigest() != part['axes_sha256']):
                    raise IOError('SASA part or axes checksum differs')
                meta = json.loads(raw_axes)
                if meta['scientific_metadata'] != canonical(completed['info']):
                    raise IOError('SASA scientific metadata differs')
                counts, indices, times = direct_counts.extract(raw)
                context = sasa_counts.context(meta)
                if (counts.shape != (len(indices), context.m)
                        or len(indices) * (context.m + len(context.res_atoms) + 1) != part['rows']
                        or count_store.native_hash(counts, indices, times, meta) != part['payload_hash']):
                    raise IOError('SASA scientific payload differs')
                size = counts.nbytes + indices.nbytes + times.nbytes + len(raw_axes)
                return (counts, indices, times, meta), size

            started = time.perf_counter_ns() if detail else 0
            counts, indices, times, meta = self._cached_part(key, load)
            if detail:
                detail('sasa_part_cache', started)
                started = time.perf_counter_ns()
            chosen = np.flatnonzero(np.isin(indices, frame_list[lo:hi], assume_unique=True))
            factors = np.frombuffer(bytes.fromhex(meta['factors_f8_hex']), dtype='<f8')
            values = counts[chosen].astype('f8') / 960 * factors
            context = sasa_counts.context(meta)
            if detail:
                detail('sasa_select_counts', started)
            expand = getattr(self, '_expand_sasa', sasa_counts.expanded_table)
            table = expand(context, indices[chosen], times[chosen], values)
            yield with_result_metadata(table.replace_schema_metadata(None), completed['info'])

    def _measured_descriptor(self, trajectory, name, frames):
        sparse = len(frames) != max(frames) - min(frames) + 1
        if 'parts' in self.features and sparse and name in ('ramachandran', 'sasa'):
            paths = self._paths(trajectory, name)
            if not paths:
                raise LookupError('No completed descriptor')
            if name == 'ramachandran' and any(
                    not marker.is_file() or read_json(marker).get('schema') != 5
                    for marker in ((self.workspace.resolve_path(relative)).with_suffix('.json') for _, relative in paths)):
                return BaseProvider._descriptor(self, trajectory, name, frames)
            selected = pa.array(frames, type=pa.int64())
            tables = []
            for result_id, relative in paths:
                if name == 'sasa':
                    tables.extend(self._sasa_tables(result_id, relative, frames))
                else:
                    tables.extend(table.filter(pc.is_in(table['frame'], value_set=selected))
                                  for table in self._rama_tables(result_id, relative))
            if tables:
                return pa.concat_tables(tables)
        if 'adaptive' in self.features and name in ('sasa', 'ramachandran',
                                                    'secondary_structure', 'electrostatic_score'):
            return experiments.ExperimentalProvider._measured_descriptor(
                self, trajectory, name, frames)
        return BaseProvider._descriptor(self, trajectory, name, frames)
