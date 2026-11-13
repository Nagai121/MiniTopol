"""Opt-in memory, encoded-response, indexing and bounded pipeline experiments."""
from dataclasses import dataclass
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
try:
    from numba import njit
except ImportError:
    def njit(**_):
        return lambda function: function

from . import experiments, query_experiments, runtime
from .query_scheduling import ScheduledProvider
from ..storage import read_json

BASE_FEATURES = 'parts,adaptive,current,coord-prefetch,global,next,catalog'


@dataclass(frozen=True)
class EncodedTable:
    data: bytes | memoryview
    num_rows: int
    schema: pa.Schema


@njit(cache=True, nogil=True)
def gather_rows(starts, stops):
    count = 0
    for i in range(len(starts)):
        count += stops[i] - starts[i]
    indices = np.empty(count, dtype=np.int64)
    position = 0
    for i in range(len(starts)):
        for row in range(starts[i], stops[i]):
            indices[position] = row
            position += 1
    return indices


class AdvancedProvider(ScheduledProvider):
    workers = 2

    def __init__(self, workspace, budget):
        super().__init__(workspace, budget)
        self.total_budget = budget
        self.local = threading.local()
        self.pressure_events = 0
        self.config = {
            'total_ram': experiments.psutil.virtual_memory().total,
            'available_ram': experiments.psutil.virtual_memory().available,
            'retained_budget': budget, 'coordinate_budget': self.decoded_budget,
            'part_budget': self.part_budget, 'shared_budget': self.cache.budget,
            'mode': os.environ.get('MINITOPOL_EXPERIMENT_MEMORY_MODE', 'fixed'),
            'features': sorted(self.features), 'workers': self.workers}
        if 'native' in self.features:
            from . import advanced_native
            self._expand_sasa = lambda ctx, frames, times, values: advanced_native.build(
                self, ctx, frames, times, values)
            gather_rows(np.array([0], dtype=np.int64), np.array([1], dtype=np.int64))

    def _get(self, request, trace=None):
        # Maintain the cache lock ordering used by the request builder.
        with self.cache.lock:
            available = experiments.psutil.virtual_memory().available
            if available < 6 * 1024**3:
                target = available // 5
                with self.part_lock:
                    if self.part_bytes + self.decoded_used + self.cache.used > target:
                        self.parts.clear()
                        self.part_bytes = 0
                        self.decoded_chunks.clear()
                        self.decoded_used = 0
                        self.cache._evict(max(0, self.cache.budget - target))
                        self.pressure_events += 1
            result = super()._get(request, trace=trace)
            result['memory_configuration'] = self.config
            if trace is not None:
                trace.increment('memory_pressure_events', self.pressure_events)
            return result

    def _encoding_key(self, trajectory, name):
        paths = self._paths(trajectory, name)
        return ('encoded', trajectory, name, tuple(
            (result_id, relative, self._file_version((self.workspace.resolve_path(relative)).with_suffix('.json')))
            for result_id, relative in paths))

    def _descriptor(self, trajectory, name, frames):
        if 'encoded' in self.features and name in runtime.GLOBAL_DESCRIPTORS:
            key = self._encoding_key(trajectory, name)
            with self.part_lock:
                if key in self.parts:
                    self.parts.move_to_end(key)
                    table = self.parts[key][0]
                    self._record('encoded_response_hit')
                    self._record('encoded_response_reused_bytes', len(table.data))
                    return table
            if name in ('time_correlation', 'dynamic_correlation') and sum(
                    read_json((self.workspace.resolve_path(relative)).with_suffix('.json')).get('rows', 0)
                    for _, relative, _ in key[-1]) >= 8_000_000:
                encoded = self._encode_large_global(trajectory, name)
                self._cached_part(key, lambda: (encoded, len(encoded.data)))
                self._record('encoded_response_miss')
                self._record('streamed_global_descriptor')
                return encoded
            table = super()._descriptor(trajectory, name, frames)
            # Encode under the existing serialization stage, after query returns.
            self.local.encoding = (name, key)
            return table
        pending = getattr(self.local, 'pipeline', {})
        if name in pending:
            table, elapsed = pending[name].result()
            self._record('pipeline_' + name + '_ns_work', elapsed)
            return table
        return super()._descriptor(trajectory, name, frames)

    def _encode_large_global(self, trajectory, name):
        with self.workspace.catalog() as con:
            paths = [str(self.workspace.resolve_path(row[0])) for row in con.execute(
                "SELECT COALESCE(p.path,r.path) FROM results r LEFT JOIN result_parts p ON p.result_id=r.id "
                "WHERE r.trajectory=? AND r.descriptor=? ORDER BY r.id,p.part_index",
                [trajectory, name]).fetchall()]
            relation = con.read_parquet(paths, union_by_name=True, hive_partitioning=False)
            sink = pa.BufferOutputStream()
            options = pa.ipc.IpcWriteOptions(compression='lz4')
            read_batches = getattr(relation, 'to_arrow_reader', None) or relation.fetch_record_batch
            with read_batches(65_536) as batches:
                schema = batches.schema
                rows = 0
                with pa.ipc.new_stream(sink, schema, options=options) as writer:
                    for batch in batches:
                        writer.write_batch(batch)
                        rows += batch.num_rows
            return EncodedTable(memoryview(sink.getvalue()).cast('B'), rows, schema)

    def _encode_descriptor(self, trajectory, name, table):
        if isinstance(table, EncodedTable):
            return table.data
        data = super()._encode_descriptor(trajectory, name, table)
        pending = getattr(self.local, 'encoding', None)
        if pending is not None and pending[0] == name:
            self.local.encoding = None
            encoded = EncodedTable(bytes(data), table.num_rows, table.schema)
            self._cached_part(pending[1], lambda: (encoded, len(encoded.data)))
            self._record('encoded_response_miss')
        return data

    def _rama_select(self, table, frames):
        selected = np.sort(np.asarray(frames, dtype=np.int64))
        axis = table['frame'].to_numpy()
        starts = np.searchsorted(axis, selected, side='left')
        stops = np.searchsorted(axis, selected, side='right')
        if 'native' in self.features:
            indices = gather_rows(starts, stops)
        else:
            length = stops - starts
            offsets = np.cumsum(length) - length
            indices = np.repeat(starts - offsets, length) + np.arange(length.sum(), dtype=np.int64)
        return table.take(pa.array(indices))

    def _measured_descriptor(self, trajectory, name, frames):
        sparse = len(frames) != max(frames) - min(frames) + 1
        if ('indexed' in self.features or 'native' in self.features) and name == 'ramachandran' and sparse:
            paths = self._paths(trajectory, name)
            if any(not marker.is_file() or read_json(marker).get('schema') != 5
                   for marker in ((self.workspace.resolve_path(path)).with_suffix('.json') for _, path in paths)):
                return super()._measured_descriptor(trajectory, name, frames)
            tables = []
            for result_id, relative in paths:
                tables.extend(self._rama_select(table, frames)
                              for table in self._rama_tables(result_id, relative))
            if not tables:
                raise LookupError('No completed Ramachandran')
            return pa.concat_tables(tables)
        return super()._measured_descriptor(trajectory, name, frames)

    def _request_cache(self, key, builder, trace=None):
        if 'pipeline' not in self.features:
            return super()._request_cache(key, builder, trace=trace)
        import json
        tid, frames, _, _, descriptors, _ = json.loads(key)

        def build():
            targets = [name for name in ('sasa', 'contacts', 'electrostatic_score', 'secondary_structure')
                       if name in descriptors][:self.workers]
            if len(descriptors) < 2:
                return builder()

            def load(name):
                started = time.perf_counter_ns()
                table = self._measured_descriptor(tid, name, frames)
                return table, time.perf_counter_ns() - started

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                self.local.pipeline = {name: pool.submit(load, name) for name in targets}
                try:
                    return builder()
                finally:
                    self.local.pipeline = {}
        return super()._request_cache(key, build, trace=trace)
