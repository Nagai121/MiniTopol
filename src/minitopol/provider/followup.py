"""Opt-in follow-up candidates for descriptor reads and shared-memory encoding."""
import json
import os
from pathlib import Path
import threading
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import psutil

from . import advanced_experiments as previous, runtime
from .followup_ipc import SegmentIPC


class FrameIndex:
    def __init__(self, table):
        self.table = table
        column = table['frame']
        self.valid = len(table) - column.null_count
        self.valid_first = column.null_count == 0 or column[0].is_valid
        start = 0 if self.valid_first else column.null_count
        axis = column.slice(start, self.valid).to_numpy()
        if np.any(axis[1:] < axis[:-1]) or column.slice(start, self.valid).null_count:
            self.axis = None
            self.edges = None
            return
        changes = np.flatnonzero(axis[1:] != axis[:-1]) + 1
        self.edges = np.r_[0, changes, self.valid] + start
        self.axis = axis[self.edges[:-1] - start] if self.valid else axis

    @property
    def nbytes(self):
        return 0 if self.axis is None else self.axis.nbytes + self.edges.nbytes

    def select(self, selected, detail, name):
        started = time.perf_counter_ns()
        if self.axis is None:
            column = self.table['frame']
            mask = pc.or_(pc.is_in(column, value_set=pa.array(selected)), pc.is_null(column))
            result = self.table.filter(mask)
            detail(name + '_filter', started)
            return result
        positions = np.searchsorted(self.axis, selected)
        valid = positions < len(self.axis)
        positions = positions[valid]
        positions = positions[self.axis[positions] == selected[valid]]
        cuts = np.flatnonzero(np.diff(positions) != 1) + 1
        starts = self.edges[positions[np.r_[0, cuts]]] if positions.size else np.empty(0, dtype=np.int64)
        stops = self.edges[positions[np.r_[cuts - 1, len(positions) - 1]] + 1] if positions.size else starts
        nulls = len(self.table) - self.valid
        if nulls:
            if self.valid_first:
                starts, stops = np.r_[starts, self.valid], np.r_[stops, len(self.table)]
            else:
                starts, stops = np.r_[0, starts], np.r_[nulls, stops]
        detail(name + '_selection', started)
        started = time.perf_counter_ns()
        if len(starts) == 1:
            result = self.table.slice(int(starts[0]), int(stops[0] - starts[0]))
        elif len(starts):
            result = self.table.take(pa.array(previous.gather_rows(starts, stops)))
        else:
            result = self.table.slice(0, 0)
        detail(name + '_materialize', started)
        return result


class FollowupProvider(previous.AdvancedProvider):
    def __init__(self, workspace, budget):
        super().__init__(workspace, budget)
        self.diagnostics = threading.local()
        self.diagnostic_lock = threading.Lock()
        self.profile_detail = 'detail' in self.features

    def _detail(self, name, started):
        details = getattr(self.diagnostics, 'measurements', None)
        if details is not None:
            details[name] = details.get(name, 0) + time.perf_counter_ns() - started

    def _prepare_rama_tables(self, tables):
        return tuple(FrameIndex(table) for table in tables)

    def _indexed_contacts(self, trajectory, frames):
        started = time.perf_counter_ns()
        with self.workspace.catalog() as con:
            paths = [self.workspace.resolve_path(row[0]) for row in con.execute(
                "SELECT COALESCE(p.path,r.path) FROM results r LEFT JOIN result_parts p ON p.result_id=r.id "
                "WHERE r.trajectory=? AND r.descriptor='contacts' ORDER BY r.id,p.part_index",
                [trajectory]).fetchall()]
        if not paths:
            raise LookupError('No completed contacts')
        key = ('contacts-index', trajectory, tuple((str(path), self._file_version(path)) for path in paths))
        self._detail('contacts_catalog', started)
        def load():
            limit = min(self.part_budget, psutil.virtual_memory().available // 3)
            estimate = 0
            for path in paths:
                metadata = pq.read_metadata(path)
                # Parquet dictionary pages understate the size of repeated
                # strings once expanded into Arrow buffers.
                estimate += max(metadata.num_rows * 192,
                                sum(metadata.row_group(i).total_byte_size
                                    for i in range(metadata.num_row_groups)))
                if estimate > limit:
                    return None, 0
            started = time.perf_counter_ns()
            try:
                with self.workspace.catalog() as con:
                    table = con.read_parquet([str(path) for path in paths], union_by_name=True,
                                             hive_partitioning=False).to_arrow_table()
            except duckdb.OutOfMemoryException:
                return None, 0
            self._detail('contacts_scan', started)
            indexed = FrameIndex(table)
            return indexed, table.get_total_buffer_size() + indexed.nbytes

        indexed = self._cached_part(key, load)
        if indexed is None:
            self._record('contacts_scoped_query')
            return self._scoped_contacts(paths, frames)
        return indexed.select(np.sort(np.asarray(frames, dtype=np.int64)), self._detail, 'contacts')

    def _scoped_contacts(self, paths, frames):
        start, stop = min(frames), max(frames) + 1
        if len(frames) == stop - start:
            predicate = f'frame >= {start} AND frame < {stop}'
        else:
            # Frame values are validated integers before reaching the query.
            predicate = 'frame IN (' + ','.join(map(str, frames)) + ')'
        with self.workspace.catalog() as con:
            return con.read_parquet([str(path) for path in paths], union_by_name=True,
                                    hive_partitioning=False).filter(
                f'frame IS NULL OR ({predicate})').to_arrow_table()

    def _baseline_contacts(self, trajectory, frames):
        started = time.perf_counter_ns()
        with self.workspace.catalog() as con:
            paths = [str(self.workspace.resolve_path(row[0])) for row in con.execute(
                "SELECT COALESCE(p.path,r.path) FROM results r LEFT JOIN result_parts p ON p.result_id=r.id "
                "WHERE r.trajectory=? AND r.descriptor='contacts' ORDER BY r.id,p.part_index",
                [trajectory]).fetchall()]
            if not paths:
                raise LookupError('No completed contacts')
            self._detail('contacts_catalog', started)
            started = time.perf_counter_ns()
            first, last = min(frames), max(frames) + 1
            table = con.read_parquet(paths, union_by_name=True, hive_partitioning=False).filter(
                f'frame IS NULL OR (frame >= {first} AND frame < {last})').to_arrow_table()
            self._detail('contacts_scan', started)
        if len(frames) != last - first:
            started = time.perf_counter_ns()
            mask = pc.or_(pc.is_in(table['frame'], value_set=pa.array(frames, type=pa.int64())),
                          pc.is_null(table['frame']))
            table = table.filter(mask)
            self._detail('contacts_filter', started)
        return table

    def _rama_select(self, table, frames):
        if not self.profile_detail and not isinstance(table, FrameIndex):
            return super()._rama_select(table, frames)
        selected = np.sort(np.asarray(frames, dtype=np.int64))
        if isinstance(table, FrameIndex):
            return table.select(selected, self._detail, 'rama')
        started = time.perf_counter_ns()
        axis = table['frame'].to_numpy()
        starts = np.searchsorted(axis, selected, side='left')
        stops = np.searchsorted(axis, selected, side='right')
        indices = previous.gather_rows(starts, stops)
        self._detail('rama_selection', started)
        started = time.perf_counter_ns()
        result = table.take(pa.array(indices))
        self._detail('rama_materialize', started)
        return result

    def _measured_descriptor(self, trajectory, name, frames):
        timed = self.profile_detail and name in ('sasa', 'ramachandran', 'contacts')
        if timed:
            self.diagnostics.measurements = {}
        started = time.perf_counter_ns()
        try:
            if name == 'contacts':
                if 'contacts-index' in self.features:
                    return self._indexed_contacts(trajectory, frames)
                if timed:
                    return self._baseline_contacts(trajectory, frames)
            return super()._measured_descriptor(trajectory, name, frames)
        finally:
            if timed:
                details = self.diagnostics.measurements
                self.diagnostics.measurements = None
                record = dict(name=name, frames=len(frames), elapsed_ns=time.perf_counter_ns()-started, **details)
                path = Path(os.environ['MINITOPOL_PROVIDER_PROFILE_PATH']).with_name('descriptor-detail.jsonl')
                with self.diagnostic_lock, path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')

    def _encode_descriptor(self, trajectory, name, table):
        if ('direct-ipc' in self.features and isinstance(table, pa.Table)
                and name not in runtime.GLOBAL_DESCRIPTORS
                and all(pa.types.is_primitive(field.type) or pa.types.is_string(field.type)
                        or pa.types.is_large_string(field.type) for field in table.schema)):
            return SegmentIPC(table)
        return super()._encode_descriptor(trajectory, name, table)
