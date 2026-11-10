"""Build and read verified Ramachandran caches under the existing Reader protocol."""
import os
import time

import numpy as np
import pyarrow as pa

from .compact_rama import CompactRama, pack
from .derived_store import DerivedStore
from .followup import FollowupProvider, FrameIndex
from ..descriptors.virtual_rama_store import environment, source_hashes


class PersistentProvider(FollowupProvider):
    def __init__(self, workspace, budget):
        super().__init__(workspace, budget)
        self.derived = DerivedStore(os.environ.get(
            'MINITOPOL_DERIVED_CACHE', self.workspace.path / 'work' / 'provider-derived'))
        self.build_derived = False
        self.recipe_contract = None

    def _derived_rama(self, key, materialize):
        if self.recipe_contract is None:
            self.recipe_contract = environment(), source_hashes()
        key = ('compact-layout-v2', self.recipe_contract, key)
        kind = 'rama-compact'
        if self.build_derived:
            start = time.perf_counter()
            path = self.derived.publish(kind, key, (pack(table) for table in materialize()))
            self.builds.append(dict(kind=kind, seconds=time.perf_counter()-start, folder=str(path),
                                    bytes=sum(file.stat().st_size for file in path.iterdir())))
        try:
            return self.derived.tables(kind,key)
        except (OSError, ValueError, KeyError, TypeError, pa.ArrowException):
            if self.build_derived:
                raise
            self._record('derived_cache_fallback')
            return tuple(materialize())

    def _prepare_rama_tables(self, tables):
        return tuple(CompactRama(table) if b'compact-axis' in (table.schema.metadata or {})
                     else FrameIndex(table) for table in tables)

    def _rama_select(self, table, frames):
        if isinstance(table, CompactRama):
            return table.select(np.sort(np.asarray(frames,dtype=np.int64)))
        return super()._rama_select(table, frames)

    def _array(self, name, array, unit):
        array = np.ascontiguousarray(array)
        return name, memoryview(array).cast('B'), {
            'format':'ndarray','dtype':array.dtype.str,'shape':list(array.shape),'unit':unit}
