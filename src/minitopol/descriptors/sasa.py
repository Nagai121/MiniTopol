import pyarrow as pa
import numpy as np

from .kernels import sasa_atoms
try:
    from .csr_sasa import _budgeted_sasa
except ImportError:
    _budgeted_sasa=None


SCHEMA = pa.schema([("frame", pa.int64()), ("time_ps", pa.float64()), ("level", pa.string()),
                    ("point", pa.int64()), ("residue", pa.int64()), ("value", pa.float64())])


def output_tables(frame,time,values,residue_index,res_atoms):
    """Same schemas, row order, keys and 4096-row boundaries, by columns."""
    def table(level,point,residue,value):
        n=len(value)
        return pa.Table.from_arrays([
            pa.array(np.full(n,frame),type=pa.int64()),
            pa.array(np.full(n,time),type=pa.float64()),
            pa.array([level]*n,type=pa.string()),
            pa.nulls(n,pa.int64()) if point is None else pa.array(point,type=pa.int64()),
            pa.nulls(n,pa.int64()) if residue is None else pa.array(residue,type=pa.int64()),
            pa.array(value,type=pa.float64())],schema=SCHEMA)
    for part,start in enumerate(range(0,len(values),4096)):
        stop=min(start+4096,len(values))
        yield f'atom-{frame}-{part}',table('atom',np.arange(start,stop),residue_index[start:stop],values[start:stop])
    residues=list(res_atoms)
    aggregates=np.asarray([values[points].sum() for points in res_atoms.values()])
    for offset in range(0,len(residues),4096):
        yield f'residue-{frame}-{offset}',table('residue',None,residues[offset:offset+4096],aggregates[offset:offset+4096])
    yield f'total-{frame}',table('total',None,None,np.asarray([values.sum()]))


def calculate(ctx, writer):
    radii = ctx.atomic_values("radii")
    for block in ctx.coordinate_blocks():
        for index, positions in enumerate(block.positions):
            frame = block.start + index
            if writer.has(f"total-{frame}"):
                continue
            values = (_budgeted_sasa(positions,radii,ctx.budget) if _budgeted_sasa is not None
                      else sasa_atoms(positions,radii))
            for key,table in output_tables(frame,block.time[index],values,ctx.atoms['residue_index'],ctx.res_atoms):
                writer.append(key,table)
