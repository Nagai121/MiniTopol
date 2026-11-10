"""Opt-in, independent Provider experiments; normal Provider defaults stay intact.

Run only through an explicit private registration launch command. Scientific
calculators, saved formats and result verification contracts are unchanged.
"""
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc

from . import runtime

VARIANTS = ('baseline', 'decoded_cache', 'sparse_query', 'coordinate_catalog',
            'payload_views', 'parallel_decode')
_coordinate_trace = None
_counter_lock = threading.Lock()


def _work(name, elapsed):
    trace = _coordinate_trace
    if trace is not None:
        with _counter_lock:
            trace.increment(name, elapsed)


def install_work_counters():
    """Worker elapsed sums, not additive top-level stage times or cold disk I/O."""
    from zarr.storage import LocalStore
    from ..agx_zarr import AGXNumcodec, AGX2Numcodec, AGX3Numcodec
    original_get = LocalStore.get

    @functools.wraps(original_get)
    async def get(store, key, *args, **kwargs):
        started = time.perf_counter_ns()
        try:
            return await original_get(store, key, *args, **kwargs)
        finally:
            if '/c/' in key:
                _work('storage_chunk_get_ns_work', time.perf_counter_ns() - started)

    LocalStore.get = get
    for codec in (AGXNumcodec, AGX2Numcodec, AGX3Numcodec):
        original = codec.decode

        def wrap(decode):
            @functools.wraps(decode)
            def measured(self, *args, **kwargs):
                started = time.perf_counter_ns()
                try:
                    return decode(self, *args, **kwargs)
                finally:
                    _work('agx_decode_ns_work', time.perf_counter_ns() - started)
                    _work('agx_decode_calls', 1)
            return measured

        codec.decode = wrap(original)


class ExperimentalProvider(runtime.DataProvider):
    variant = 'baseline'

    def __init__(self, workspace, budget):
        super().__init__(workspace, budget)
        if self.variant == 'decoded_cache':
            self.decoded_budget = min(512 * 1024**2, budget // 5)
            self.cache.budget = budget - self.decoded_budget
        self.skip_coordinate_catalog = self.variant == 'coordinate_catalog'
        self.payload_views = self.variant == 'payload_views'
        self.request_context = threading.local()

    def _get(self, request, trace=None):
        if trace is not None:
            trace.increment('decoded_cache_budget_bytes', self.decoded_budget)
            trace.increment('shared_cache_budget_bytes', self.cache.budget)
        self.request_context.trace = trace
        try:
            return super()._get(request, trace=trace)
        finally:
            self.request_context.trace = None

    def _array(self, name, array, unit):
        if not self.payload_views:
            return super()._array(name, array, unit)
        array = np.ascontiguousarray(array)
        return name, memoryview(array).cast('B'), {
            'format': 'ndarray', 'dtype': array.dtype.str,
            'shape': list(array.shape), 'unit': unit}

    def _load_coordinates(self, *args, trace=None, **kwargs):
        global _coordinate_trace
        _coordinate_trace = trace
        try:
            if self.variant == 'parallel_decode':
                return self._parallel_coordinates(*args, trace=trace, **kwargs)
            return super()._load_coordinates(*args, trace=trace, **kwargs)
        finally:
            _coordinate_trace = None

    def _parallel_coordinates(self, trajectory, frames, selected, frame_count, manifest, trace=None):
        import zarr
        with runtime._stage(trace, 'output_array_allocate'):
            output = (np.empty((len(frames), len(selected), 3), dtype='<f4'),
                      np.empty(len(frames), dtype='<f8'),
                      np.empty((len(frames), 3, 3), dtype='<f4'))
        with runtime._stage(trace, 'zarr_open_metadata'):
            source = zarr.open_group(str(self.workspace.trajectory_path(trajectory) / 'A.zarr'), mode='r')
            arrays = tuple(source[name] for name in ('positions', 'time', 'box'))
            width = int(arrays[0].chunks[0])
        with runtime._stage(trace, 'chunk_grouping'):
            groups = {}
            for destination, frame in enumerate(frames):
                groups.setdefault(frame // width, []).append((destination, frame))
        if psutil.virtual_memory().available < self.cache.budget // 4:
            self.decoded_chunks.clear()
            self.decoded_used = 0
        cached, missing, keys = {}, [], {}
        for group in sorted(groups):
            first, last = group * width, min((group + 1) * width, frame_count)
            key = (trajectory, manifest['identity'], first, last)
            keys[group] = key
            if key in self.decoded_chunks:
                cached[group] = self.decoded_chunks[key]
                self.decoded_chunks.move_to_end(key)
            else:
                missing.append((group, first, last))
            if trace is not None:
                trace.increment('decoded_chunk_hit' if group in cached else 'decoded_chunk_miss')

        def decode(item):
            group, first, last = item
            return group, tuple(array[first:last] for array in arrays)

        # Four in-flight chunks at most. Consume in source order to preserve
        # deterministic extraction and LRU behavior, with bounded intermediates.
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {}
            next_item = iter(missing)
            for _ in range(4):
                item = next(next_item, None)
                if item is not None:
                    pending[item[0]] = pool.submit(decode, item)
            for group, rows in sorted(groups.items()):
                if group in cached:
                    decoded = cached[group]
                else:
                    with runtime._stage(trace, 'zarr_fetch_and_decode'):
                        _, decoded = pending.pop(group).result()
                    item = next(next_item, None)
                    if item is not None:
                        pending[item[0]] = pool.submit(decode, item)
                    size = sum(array.nbytes for array in decoded)
                    if trace is not None:
                        trace.increment('decoded_source_bytes', size)
                    if size <= self.decoded_budget:
                        while self.decoded_used + size > self.decoded_budget:
                            _, old = self.decoded_chunks.popitem(last=False)
                            self.decoded_used -= sum(array.nbytes for array in old)
                        self.decoded_chunks[keys[group]] = decoded
                        self.decoded_used += size
                with runtime._stage(trace, 'coordinate_extract_copy'):
                    destinations = np.fromiter((i for i, _ in rows), dtype=np.int64)
                    local = np.fromiter((f - group * width for _, f in rows), dtype=np.int64)
                    output[0][destinations] = decoded[0][local][:, selected, :]
                    output[1][destinations] = decoded[1][local]
                    output[2][destinations] = decoded[2][local]
        return output

    def _descriptor(self, trajectory, name, frames):
        started = time.perf_counter_ns()
        result = self._measured_descriptor(trajectory, name, frames)
        trace = getattr(self.request_context, 'trace', None)
        if trace is not None:
            trace.increment('descriptor_query_' + name + '_ns_work', time.perf_counter_ns() - started)
            trace.increment('descriptor_rows_' + name, result.num_rows)
        return result

    def _measured_descriptor(self, trajectory, name, frames):
        if (self.variant != 'sparse_query' or name in runtime.GLOBAL_DESCRIPTORS
                or len(frames) == max(frames) - min(frames) + 1):
            return super()._descriptor(trajectory, name, frames)
        values = pa.array(frames, type=pa.int64())
        if name in ('sasa', 'ramachandran'):
            if name == 'sasa':
                from ..descriptors.count_store import logical_tables
            else:
                from ..descriptors.virtual_rama_layout import logical_tables
            with self.workspace.catalog() as con:
                paths = con.execute('SELECT path FROM results WHERE trajectory=? AND descriptor=? ORDER BY id',
                                    [trajectory, name]).fetchall()
            if not paths:
                raise LookupError('No completed descriptor')
            parts = []
            for (relative,) in paths:
                for table in logical_tables(self.workspace, relative, start=min(frames), stop=max(frames)+1):
                    parts.append(table.filter(pc.is_in(table['frame'], value_set=values)))
            return pa.concat_tables(parts)
        with self.workspace.catalog() as con:
            paths = [str(self.workspace.path / row[0]) for row in con.execute(
                'SELECT COALESCE(p.path,r.path) FROM results r '
                'LEFT JOIN result_parts p ON p.result_id=r.id '
                'WHERE r.trajectory=? AND r.descriptor=? ORDER BY r.id,p.part_index',
                [trajectory, name]).fetchall()]
            if not paths:
                raise LookupError('No completed descriptor')
            relation = con.read_parquet(paths, union_by_name=True, hive_partitioning=False)
            predicate = 'frame IN (' + ','.join(str(int(f)) for f in frames) + ')'
            if name == 'contacts':
                predicate = 'frame IS NULL OR (' + predicate + ')'
            return relation.filter(predicate).to_arrow_table()
