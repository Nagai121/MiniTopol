"""Lossless compact column representation of canonical Rama parts."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def pack(table):
    frame = table['frame'].to_numpy()
    width = np.searchsorted(frame, frame[0], side='right')
    if not width or len(frame) % width:
        raise ValueError('Uneven Ramachandran frame layout')
    axis = table.slice(0, width).select(['residue','angle'])
    constants = [name for name in table.column_names
                 if name not in ('frame','time_ps','residue','angle','value','reason')]
    for name in constants:
        column = table[name]
        if not column.equals(pa.chunked_array([pa.repeat(column[0],len(table))])):
            raise ValueError(f'Nonconstant Ramachandran field: {name}')
        axis = axis.append_column(name,pa.repeat(column[0],width))
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, axis.schema) as stream:
        stream.write_table(axis)
    meta = {**(table.schema.metadata or {}), b'compact-axis':sink.getvalue().to_pybytes()}
    columns = [pa.array(frame[::width]), pa.array(table['time_ps'].to_numpy()[::width])]
    for name in ('value','reason'):
        columns.append(pa.FixedSizeListArray.from_arrays(table[name].combine_chunks(), int(width)))
    return pa.Table.from_arrays(columns, names=['frame','time_ps','value','reason']).replace_schema_metadata(meta)


class CompactRama:
    def __init__(self, table):
        self.table = table
        self.frames = table['frame'].to_numpy()
        self.axis = pa.ipc.open_stream(table.schema.metadata[b'compact-axis']).read_all()

    @property
    def nbytes(self):
        return self.frames.nbytes + self.axis.get_total_buffer_size()

    def select(self, selected):
        positions = np.searchsorted(self.frames, selected)
        valid = positions < len(self.frames)
        positions = positions[valid]
        positions = positions[self.frames[positions] == selected[valid]]
        selected = self.table.take(pa.array(positions))
        width = len(self.axis)
        times = selected['time_ps'].to_numpy()
        frames = selected['frame'].to_numpy()
        tiled = pa.array(np.tile(np.arange(width,dtype=np.int64), len(frames)))
        metadata = {k:v for k,v in self.table.schema.metadata.items() if k != b'compact-axis'}
        table = pa.Table.from_arrays([
            pa.array(np.repeat(frames,width)),pa.array(np.repeat(times,width)),
            pc.take(self.axis['residue'],tiled),pc.take(self.axis['angle'],tiled),
            selected['value'].combine_chunks().flatten(),
            selected['reason'].combine_chunks().flatten()],
            names=['frame','time_ps','residue','angle','value','reason'])
        for name in self.axis.column_names[2:]:
            table = table.append_column(name,pa.repeat(self.axis[name][0],len(table)))
        return table.replace_schema_metadata(metadata)
