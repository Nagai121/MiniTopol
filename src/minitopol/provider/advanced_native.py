"""Arrow layout reuse for SASA; numeric operations match the existing reader."""
import numpy as np
import pyarrow as pa

from ..sasa_counts import expanded_table as baseline_table
from ..descriptors.sasa import SCHEMA


def layout(context, count):
    m = context.m
    residues = list(context.res_atoms)
    nr = len(residues)
    point = np.r_[np.arange(m, dtype=np.int64), np.zeros(nr + 1, dtype=np.int64)]
    residue = np.r_[np.asarray(context.atoms['residue_index'], dtype=np.int64),
                    np.asarray(residues, dtype=np.int64), np.zeros(1, dtype=np.int64)]
    return pa.table({
        'level': pa.array((['atom']*m + ['residue']*nr + ['total'])*count, type=pa.string()),
        'point': pa.array(np.tile(point,count), mask=np.tile(np.r_[np.zeros(m,dtype=bool),np.ones(nr+1,dtype=bool)],count)),
        'residue': pa.array(np.tile(residue,count), mask=np.tile(np.r_[np.zeros(m+nr,dtype=bool),np.ones(1,dtype=bool)],count))})


def build(owner, context, frames, times, values):
    m = context.m
    nr = len(context.res_atoms)
    count = len(frames)
    width = m + nr + 1
    key = ('sasa-layout', tuple(context.atoms['residue_index']), count)
    def load():
        table = layout(context, count)
        return table, table.get_total_buffer_size()
    static = owner._cached_part(key, load)
    columns = np.empty((count,width),dtype=np.float64)
    values = np.ascontiguousarray(values,dtype=np.float64)
    columns[:,:m] = values
    for j,points in enumerate(context.res_atoms.values()):
        columns[:,m+j] = np.ascontiguousarray(values[:,points]).sum(axis=1)
    columns[:,-1] = values.sum(axis=1)
    return pa.Table.from_arrays([
        pa.array(np.repeat(np.asarray(frames,dtype=np.int64),width)),
        pa.array(np.repeat(np.asarray(times,dtype=np.float64),width)),
        static['level'], static['point'], static['residue'], pa.array(columns.ravel())],schema=SCHEMA)
