"""Opt-in query scheduling on the Provider's existing HTTP and lease protocol."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

from . import query_experiments, runtime
from .query_global_cache import GlobalBufferCache


class ScheduledProvider(query_experiments.QueryProvider):
    def __init__(self, workspace, budget):
        super().__init__(workspace, budget)
        if 'global' in self.features:
            self.cache = GlobalBufferCache(self.cache.budget, self._revision)
        self.scheduler = ThreadPoolExecutor(max_workers=1, thread_name_prefix='query-prefetch')
        self.schedule_lock = threading.Lock()
        self.part_lock = threading.Lock()
        self.pending = None
        self.prefetch_bytes = min(256 * 1024**2, self.cache.budget // 8)
        self.request_cache = self.cache.request
        self.cache.request = self._request_cache
        close = self.cache.close

        def close_with_scheduler():
            self.scheduler.shutdown(wait=True, cancel_futures=True)
            close()
        self.cache.close = close_with_scheduler

    def _load_coordinates(self, *args, trace=None, **kwargs):
        frames = args[1]
        scan = 'scan-bypass' in self.features and len(frames) == max(frames) - min(frames) + 1
        budget = self.decoded_budget
        if scan:
            self.decoded_budget = 0
        try:
            if 'coord-prefetch' in self.features:
                query_experiments.experiments._coordinate_trace = trace
                try:
                    return self._parallel_coordinates(*args, trace=trace, **kwargs)
                finally:
                    query_experiments.experiments._coordinate_trace = None
            return super()._load_coordinates(*args, trace=trace, **kwargs)
        finally:
            self.decoded_budget = budget

    def _cached_part(self, key, loader):
        # Cached immutable arrays may be returned to the request and worker.
        with self.part_lock:
            return super()._cached_part(key, loader)

    @staticmethod
    def _target(descriptors):
        return next((name for name in ('ramachandran', 'sasa') if name in descriptors), None)

    def _lookup_target(self, tid, descriptors):
        if descriptors == 'all' or descriptors == ['all']:
            with self.workspace.catalog() as con:
                descriptors = [name for (name,) in con.execute(
                    'SELECT DISTINCT descriptor FROM results WHERE trajectory=?', [tid]).fetchall()]
        return self._target(descriptors)

    def _revision(self, tid, name):
        return tuple(result_id for result_id, _ in self._paths(tid, name))

    def _submit(self, tid, name, frames, revision):
        overlays = self.workspace.storage_read_overrides()
        def load():
            started = time.perf_counter_ns()
            with self.workspace.storage_read_scope(overlays):
                table = self._measured_descriptor(tid, name, frames)
            return table, time.perf_counter_ns() - started, revision

        return self.scheduler.submit(load)

    def _request_cache(self, key, builder, trace=None):
        tid, frames, _, coordinates, descriptors, _ = json.loads(key)
        name = self._target(descriptors)
        if name is None or not ({'current', 'next'} & self.features):
            return self.request_cache(key, builder, trace=trace)

        def build():
            signature = (tid, name, tuple(frames), self._revision(tid, name))
            with self.schedule_lock:
                future = None
                if self.pending is not None:
                    previous, pending = self.pending
                    self.pending = None
                    if previous == signature:
                        future = pending
                        if trace is not None:
                            trace.increment('lookahead_match')
                            trace.increment('lookahead_ready', int(future.done()))
                    else:
                        pending.cancel()
                        if trace is not None:
                            trace.increment('lookahead_unused')
                if future is None and 'current' in self.features and (coordinates or len(descriptors) > 1):
                    future = self._submit(tid, name, frames, signature[-1])
                    if trace is not None:
                        trace.increment('current_prefetch_submitted')
            self.request_context.prefetch = (name, future, signature[-1])
            try:
                return builder()
            finally:
                self.request_context.prefetch = None
                if future is not None:
                    future.cancel()

        return self.request_cache(key, build, trace=trace)

    def _descriptor(self, trajectory, name, frames):
        scheduled = getattr(self.request_context, 'prefetch', None)
        if scheduled is not None and scheduled[0] == name and scheduled[1] is not None:
            _, future, revision = scheduled
            table, elapsed, _ = future.result()
            if self._revision(trajectory, name) == revision:
                self._record('descriptor_query_' + name + '_ns_work', elapsed)
                self._record('descriptor_rows_' + name, table.num_rows)
                self._record('prefetch_result_bytes', table.get_total_buffer_size())
                return table
        return super()._descriptor(trajectory, name, frames)

    def get(self, request):
        if not isinstance(request, dict) or 'next_frames' not in request:
            return super().get(request)
        if 'next' not in self.features:
            raise ValueError('Next-frame hints are disabled')
        request = dict(request)
        next_frames = request.pop('next_frames')
        tid = request.get('trajectory')
        if not isinstance(tid, str):
            raise ValueError('trajectory must be a string')
        manifest = self.registry.manifest(tid)
        self._check_indices(next_frames, manifest['frames'], 'next_frames')
        if len(next_frames) > 1024:
            raise ValueError('Next-frame hint exceeds the experimental bound')
        reply = super().get(request)
        try:
            name = self._lookup_target(tid, reply['descriptors'])
            if name is not None:
                # The bounded lookahead targets a single frame-scoped table.
                # Use the previously produced table's width as a conservative bound.
                part = next(p for p in reply['buffers'] if p['name'] == 'descriptor:' + name)
                estimate = part['bytes'] * len(next_frames) / len(reply['frames'])
                if estimate <= self.prefetch_bytes:
                    revision = self._revision(tid, name)
                    signature = tid, name, tuple(next_frames), revision
                    with self.schedule_lock:
                        if self.pending is None:
                            self.pending = signature, self._submit(tid, name, next_frames, revision)
                            reply['prefetch_accepted'] = True
        except BaseException:
            self.cache.release(reply['lease'])
            raise
        return reply
