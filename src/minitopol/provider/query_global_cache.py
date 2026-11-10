"""Share immutable global descriptor segments between distinct response leases."""
import json

from . import runtime
from .followup_ipc import SegmentIPC


class GlobalBufferCache(runtime.Cache):
    def __init__(self, budget, revision):
        super().__init__(budget)
        self.revision = revision
        self.global_blocks = {}

    def _drop(self, block):
        block['refs'] -= 1
        if not block['refs']:
            block['segment'].close()
            block['segment'].unlink()
            self.used -= block['size']
            if block['identity'] is not None:
                self.global_blocks.pop(block['identity'], None)

    def _evict(self, needed):
        for key, entry in list(self.entries.items()):
            if self.used + needed <= self.budget:
                break
            if not entry['leases']:
                for block in entry['blocks']:
                    self._drop(block)
                del self.entries[key]

    def request(self, key, builder, trace=None):
        started = runtime.time.perf_counter_ns() if trace is not None else None
        with self.lock:
            if trace is not None:
                trace.nanoseconds['shared_cache_lock_wait'] += runtime.time.perf_counter_ns() - started
                trace.increment('shared_cache_hit' if key in self.entries else 'shared_cache_miss')
            if key in self.entries:
                self.hits += 1
            else:
                self.misses += 1
                tid, _, _, _, descriptors, _ = json.loads(key)
                revisions = {name: self.revision(tid, name) for name in descriptors
                             if name in runtime.GLOBAL_DESCRIPTORS}
                # Pin existing shared blocks during eviction and rollback all
                # allocations on failure before exposing a response lease.
                blocks, metadata, pending = [], [], []
                try:
                    payloads = builder()
                    if any(self.revision(tid, name) != revision for name, revision in revisions.items()):
                        raise RuntimeError('Global descriptor revision changed while reading')
                    for name, payload, spec in payloads:
                        identity = None
                        if spec.get('scope') == 'global':
                            descriptor = name.split(':', 1)[1]
                            identity = tid, descriptor, revisions[descriptor]
                        block = self.global_blocks.get(identity)
                        if block is not None:
                            block['refs'] += 1
                            blocks.append(block)
                            if trace is not None:
                                trace.increment('global_segment_hit')
                                trace.increment('global_segment_reused_bytes', block['size'])
                        else:
                            pending.append((len(blocks), name, payload, spec, identity))
                            blocks.append(None)
                    needed = sum(len(payload) for _, _, payload, _, _ in pending)
                    self._evict(needed)
                    if self.used + needed > self.budget:
                        raise MemoryError('Shared segments exceed the cache budget or are lease-pinned')
                    for index, name, payload, spec, identity in pending:
                        size = len(payload)
                        with runtime._stage(trace, 'shared_memory_allocate'):
                            segment = runtime.shared_memory.SharedMemory(create=True, size=max(1, size))
                        block = {'segment': segment, 'size': size, 'identity': identity, 'refs': 1,
                                 'metadata': {'name': name, 'shm': segment.name, 'bytes': size, **spec}}
                        blocks[index] = block
                        self.used += size
                        if identity is not None:
                            self.global_blocks[identity] = block
                        if isinstance(payload, SegmentIPC):
                            with runtime._stage(trace, 'descriptor_arrow_encode'):
                                payload.write(segment)
                        else:
                            with runtime._stage(trace, 'shared_memory_copy'):
                                segment.buf[:size] = payload
                        if trace is not None:
                            trace.increment('shared_memory_bytes', size)
                    metadata = [block['metadata'] for block in blocks]
                except BaseException:
                    for block in blocks:
                        if block is not None:
                            self._drop(block)
                    raise
                self.entries[key] = {'blocks': blocks, 'metadata': metadata, 'leases': set()}
            self.entries.move_to_end(key)
            lease = runtime.secrets.token_urlsafe(24)
            entry = self.entries[key]
            entry['leases'].add(lease)
            self.leases[lease] = key
            return {'lease': lease, 'buffers': entry['metadata'],
                    'cache_bytes': self.used, 'budget_bytes': self.budget}

    def close(self):
        with self.lock:
            for entry in self.entries.values():
                for block in entry['blocks']:
                    self._drop(block)
            self.entries.clear()
            self.leases.clear()
