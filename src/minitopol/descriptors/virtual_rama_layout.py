"""Read schema-5 recipes using their published, immutable frame partitions.

Scientific kernels and the original source/runtime/dependency contract remain
in virtual_rama_store. Every reconstructed part must match its saved payload.
"""
import re

import pyarrow as pa
import pyarrow.compute as pc

from ..backbone_columns import RamaBatch
from ..storage import read_json
from ..virtual_rama import ReadOnlyWorkspace
from . import ramachandran
from .context import AnalysisContext
from .precision64 import backbone_frames64
from .results import _payload_hash, with_result_metadata
from .virtual_rama_store import verify_dataset


def logical_tables(workspace, relative, *, start=None, stop=None):
    path = workspace.resolve_path(relative)
    completed = read_json(path.with_suffix('.json'))
    verify_dataset(path, completed, workspace=workspace)
    ranges = []
    position = completed['info']['start']
    for part in completed['parts']:
        match = re.fullmatch(r'frames-(\d+)-(\d+)', part['key'])
        if match is None:
            raise IOError('Invalid virtual frame partition')
        first, last = map(int, match.groups())
        if first != position or last <= first or last > completed['info']['stop']:
            raise IOError('Virtual frame partitions are not contiguous')
        ranges.append((first, last, part))
        position = last
    if position != completed['info']['stop']:
        raise IOError('Virtual frame partitions do not cover the result')
    selected = [(first, last, part) for first, last, part in ranges
                if (start is None or last > start) and (stop is None or first < stop)]
    # Reconstruct one verified part to retain the schema for an empty query.
    if not selected:
        selected = ranges[:1]
    info = completed['info']
    parameters = info['definition']['parameters']
    layout = completed['layout']
    budget = layout.get('output_budget_bytes', 128 * 1024**2)
    if type(budget) is not int or budget <= 0:
        raise IOError('Invalid published virtual output budget')
    proxy = ReadOnlyWorkspace(
        workspace.path, workspace.path / 'locks',
        overrides=workspace.readonly_overrides(completed['info']['trajectory']))
    ctx = AnalysisContext(
        proxy, info['trajectory'], coordinate_mode=parameters['coordinate_mode'],
        initial_whole=parameters['initial_whole'],
        half_box_steps=parameters['half_box_steps'], memory_mb=budget / 1024**2,
        chunk_frames=layout['chunk_frames'])
    if ctx.annotation != parameters['annotation']:
        raise IOError('Virtual annotation changed')

    class Selection:
        def has(self, frame):
            return not any(first <= frame < last for first, last, _ in selected)

    original_blocks = ctx.registry.blocks

    def selected_blocks(trajectory, representation='A', *, start=0, stop=None,
                        chunk_frames=None):
        width = chunk_frames or ctx.chunk
        end = ctx.n if stop is None else stop
        for offset in range(start, end, width):
            last = min(offset + width, end)
            if any(first < last and final > offset for first, final, _ in selected):
                yield from original_blocks(trajectory, representation, start=offset,
                                           stop=last, chunk_frames=width)

    ctx.registry.blocks = selected_blocks
    index = 0
    builder = None
    for frame, time, angles, reasons in backbone_frames64(ctx, Selection()):
        if index >= len(selected):
            raise IOError('Unexpected reconstructed virtual frames')
        first, last, part = selected[index]
        if builder is None:
            builder = RamaBatch(ctx.residues, ramachandran.SCHEMA)
        if frame != first + len(builder.frames):
            raise IOError('Virtual reconstruction frame continuity differs')
        builder.append(frame, time, angles, reasons)
        if frame + 1 == last:
            table = builder.table()
            if (len(table) != part['rows'] or _payload_hash(table) != part['payload_hash']
                    or table['value'].null_count != part['undefined_values']):
                raise IOError('Virtual reconstruction scientific payload differs')
            yield with_result_metadata(table, info)
            builder = None
            index += 1
    if builder is not None or index != len(selected):
        raise IOError('Virtual reconstruction is incomplete')


def query(workspace, *, trajectory=None, definition_id=None, columns=None,
          start=None, stop=None):
    clauses = ["descriptor='ramachandran'"]
    args = []
    if trajectory is not None:
        clauses.append('trajectory=?')
        args.append(trajectory)
    if definition_id is not None:
        clauses.append("json_extract_string(definition,'$.name')='ramachandran'")
    with workspace.catalog() as con:
        paths = con.execute('SELECT path FROM results WHERE ' + ' AND '.join(clauses)
                            + ' ORDER BY id', args).fetchall()
    if not paths:
        raise LookupError('No virtual Rama results')
    tables = []
    for (relative,) in paths:
        for table in logical_tables(workspace, relative, start=start, stop=stop):
            if definition_id is not None and table['definition_id'][0].as_py() != definition_id:
                continue
            if start is not None:
                table = table.filter(pc.greater_equal(table['frame'], int(start)))
            if stop is not None:
                table = table.filter(pc.less(table['frame'], int(stop)))
            if columns:
                table = table.select(columns)
            tables.append(table)
    if not tables:
        raise LookupError('No virtual Rama results match the definition')
    return pa.concat_tables(tables)
