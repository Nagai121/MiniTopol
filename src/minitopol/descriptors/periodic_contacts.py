"""Orthorhombic candidate search; authoritative distances remain MDAnalysis.

No scientific approximation: the inflated search radius is a superset guard,
and the existing float32-difference/double-distance C arithmetic determines
every emitted minimum and <=4.5 decision. Unsupported inputs retain the old
periodic implementation. Scratch is verified before any result publication.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree
from MDAnalysis.lib.distances import calc_bonds
from ..coordinates import box_dimensions
from .contacts import _calculate_original as original
from .contact_scratch import CompressedScratch,verify as verify_scratch,read_records,open_reader

class UseOriginal(Exception):
    pass

DTYPE = np.dtype([('frame', '<i8'), ('r', '<i8'), ('s', '<i8'), ('d', '<f8')])


def _state_table(values, times, n, schema):
    count = len(values)
    return pa.Table.from_arrays([
        pa.array(['state'] * count, type=pa.string()),
        pa.array(values['frame']), pa.array(times[values['frame']]),
        pa.array(values['r']), pa.array(values['s']), pa.array(values['d']),
        pa.nulls(count, type=pa.int64()), pa.array(np.full(count, n, dtype=np.int64)),
        pa.array(np.ones(count))], schema=schema)



def _frame_contacts(xyz,box,atom_residue,allowed,budget):
    dimensions=box_dimensions(box)
    if not np.all(dimensions[3:]==90.) or np.any(dimensions[:3]<1.) or np.any(dimensions[:3]>1e4):
        raise UseOriginal()
    if xyz.dtype!=np.dtype('f4') or not np.isfinite(xyz).all() or np.any(np.abs(xyz)>1e6):
        raise UseOriginal()
    size=len(xyz);count=len(allowed)
    if size*256+count*count*9>budget//4:raise UseOriginal()
    lengths=dimensions[:3].astype(float)
    # distance_array and calc_bonds both subtract float32 coordinates before
    # the double MIC/sqrt. The periodic Euclidean metric is 1-Lipschitz, so
    # 32 eps32*(max(abs(x))+max(box)+1) safely exceeds subtraction and modulo
    # roundoff for the guarded range. All false positives are checked exactly.
    margin=32*np.finfo(np.float32).eps*(float(np.abs(xyz).max(initial=0))+float(lengths.max())+1.)
    wrapped=np.mod(xyz.astype(float),lengths)
    # A tiny negative float32 value can round its modulo result to L exactly.
    # Canonicalize that representative for cKDTree; exact C distances still
    # use the unmodified original coordinates and the conservative margin.
    wrapped=np.where(wrapped>=lengths,0.,wrapped)
    tree=cKDTree(wrapped,boxsize=lengths);radius=4.5+margin
    pair_count=(int(tree.count_neighbors(tree,radius))-size)//2
    if pair_count*128+size*256+count*count*9>budget//4:raise UseOriginal()
    pairs=tree.query_pairs(radius,output_type='ndarray')
    r=atom_residue[pairs[:,0]];s=atom_residue[pairs[:,1]]
    low=np.minimum(r,s);high=np.maximum(r,s);keep=allowed[low,high]
    pairs=pairs[keep];low=low[keep];high=high[keep]
    distances=calc_bonds(xyz[pairs[:,0]],xyz[pairs[:,1]],box=dimensions)
    minima=np.full((count,count),np.inf)
    np.minimum.at(minima,(low,high),distances)
    finite=np.isfinite(minima)
    np.less_equal(minima,4.5,out=finite,where=finite)
    rr,ss=np.nonzero(finite)
    if len(rr)*256>budget//8:raise UseOriginal()
    return rr,ss,minima[rr,ss]

def _ordered_frames(positions,boxes,atom_residue,allowed,budget):
    workers=next((w for w in (4,2) if 9*w*len(allowed)**2+len(atom_residue)*256*w<=budget//8),1)
    if workers==1:
        for i,xyz in enumerate(positions):
            yield i,_frame_contacts(xyz,boxes[i],atom_residue,allowed,budget)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0,len(positions),workers):
            indices=range(start,min(start+workers,len(positions)))
            results=list(executor.map(lambda i:_frame_contacts(positions[i],boxes[i],atom_residue,allowed,budget//workers),indices))
            for offset,result in enumerate(results):
                yield start+offset,result

def calculate(ctx, writer):
    from .contacts import eligible, SCHEMA
    from .context import PROTEIN_NAMES
    # Once a guarded input chose the original implementation, retain that
    # unfinished partitioning even if runtime support later improves.
    if any(str(key).startswith('state-') and not str(key).startswith('state-compact-') for key in writer.by_key):
        return original(ctx,writer)
    protein = {r for r, p in ctx.res_atoms.items() if ctx.atoms['residue_name'][p[0]] in PROTEIN_NAMES}
    protein.update(ctx.atoms['residue_index'][i] for i in (ctx.annotation['alpha_carbons'] or []))
    if any(ctx.res_annotation.get(r, {}).get('run') is None for r in protein):
        raise ValueError('Contact neighbor exclusions require explicit continuous-segment annotations for protein residues')
    count = len(ctx.residues)
    blocks = (count + ctx.pair_block - 1) // ctx.pair_block
    # Bound handles and total Python buffering; unusual systems retain the old path.
    if blocks * (blocks + 1) // 2 > 512:
        return original(ctx, writer)
    # Conservative native Zstd context reservation: at most one third of the
    # unchanged working budget. More streams retain the original algorithm.
    if blocks*(blocks+1)//2*4*1024**2>ctx.budget//3:
        raise UseOriginal()
    if ctx.budget < 32 * 1024**2 or 9 * count**2 > ctx.budget // 4:
        raise UseOriginal()
    residues = np.asarray(ctx.residues, dtype=np.int64)
    ordinal = {int(r): i for i, r in enumerate(residues)}
    atom_residue = np.asarray([ordinal[r] for r in ctx.atoms['residue_index']], dtype=np.int64)
    allowed = np.zeros((count, count), dtype=bool)
    for r in range(count):
        for s in range(r + 1, count):
            allowed[r, s] = eligible(ctx, int(residues[r]), int(residues[s]))
    times = np.empty(ctx.n)
    ctx.cache_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='contacts-spool-', dir=ctx.cache_path) as folder:
        files = {}; hashes = {}; lengths = {}
        try:
            for block in ctx.registry.blocks(ctx.trajectory, chunk_frames=ctx.chunk):
                for offset,(rr,ss,distances) in _ordered_frames(block.positions,block.box,atom_residue,allowed,ctx.budget):
                    frame=block.start+offset;times[frame]=block.time[offset]
                    keys = (rr // ctx.pair_block) * blocks + ss // ctx.pair_block
                    order = np.argsort(keys, kind='stable'); keys = keys[order]
                    rr = rr[order]; ss = ss[order]; distances=distances[order]
                    boundaries = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
                    # Populate once per frame, then pass zero-copy byte slices.
                    # The original len(rr)*256 guard includes this 32-byte/row array.
                    values = np.empty(len(rr), dtype=DTYPE)
                    values['frame'] = frame; values['r'] = residues[rr]
                    values['s'] = residues[ss]; values['d'] = distances
                    frame_bytes = memoryview(values).cast('B')
                    for start, stop in zip(boundaries[:-1], boundaries[1:]):
                        if start == stop: continue
                        key = int(keys[start])
                        raw = frame_bytes[start*DTYPE.itemsize:stop*DTYPE.itemsize]
                        if key not in files:
                            files[key] = CompressedScratch(Path(folder) / str(key))
                            hashes[key] = hashlib.sha256(); lengths[key] = 0
                        files[key].write(raw); hashes[key].update(raw); lengths[key] += len(raw)
            for file in files.values(): file.close()
            # All scratch files must pass before even one completed part is published.
            for key, expected in hashes.items():
                path = Path(folder) / str(key)
                verify_scratch(path,files[key],lengths[key],expected.digest(),DTYPE.itemsize)
            limit = max(4096, min(16384, ctx.budget // 16384))
            for left in range(0, count, ctx.pair_block):
                for right in range(left, count, ctx.pair_block):
                    key = f'frequency-{left}-{right}'
                    if writer.has(key): continue
                    pairs = [(int(r), int(s)) for r in residues[left:left+ctx.pair_block]
                             for s in residues[right:right+ctx.pair_block] if r < s and eligible(ctx, int(r), int(s))]
                    if not pairs:
                        writer.append(key, pa.Table.from_pylist([], schema=SCHEMA)); continue
                    counts = np.zeros((ctx.pair_block, ctx.pair_block), dtype=np.int64)
                    path = Path(folder) / str((left // ctx.pair_block) * blocks + right // ctx.pair_block)
                    part = 0; last_count = 0
                    if path.exists():
                        with open_reader(path) as file:
                            while True:
                                values = read_records(file,limit,DTYPE)
                                if not len(values): break
                                ri = np.searchsorted(residues, values['r']) - left
                                si = np.searchsorted(residues, values['s']) - right
                                np.add.at(counts, (ri, si), 1)
                                writer.append(f'state-compact-{left}-{right}-{part}', _state_table(values, times, ctx.n, SCHEMA))
                                part += 1; last_count = len(values)
                    # Legacy always appends a terminal part, including an empty one.
                    if last_count == 0 or last_count == limit:
                        writer.append(f'state-compact-{left}-{right}-{part}', pa.Table.from_pylist([], schema=SCHEMA))
                    rows = []
                    for r, s in pairs:
                        value = int(counts[ordinal[r]-left, ordinal[s]-right])
                        if value:
                            rows.append(dict(level='frequency', frame=None, time_ps=None, residue_i=r, residue_j=s,
                                             distance_angstrom=None, contact_frames=value, frames=ctx.n, value=value/ctx.n))
                    writer.append(key, pa.Table.from_pylist(rows, schema=SCHEMA))
        finally:
            for file in files.values(): file.close()
