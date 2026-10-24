"""Rebuildable block-partitioned binary scratch; published layout is unchanged.

Every scratch byte is hashed on creation and re-read before any publication.
No journal/checksum/fsync changes are made to the final DatasetWriter.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree
from .accelerated import residue_minima
from .spatial_contacts_legacy import UseOriginal, calculate as original, _frame_hash, _verify_scratch

DTYPE = np.dtype([('frame', '<i8'), ('r', '<i8'), ('s', '<i8'), ('d', '<f8')])


def _state_table(values, times, n, schema):
    count = len(values)
    return pa.Table.from_arrays([
        pa.array(['state'] * count, type=pa.string()),
        pa.array(values['frame']), pa.array(times[values['frame']]),
        pa.array(values['r']), pa.array(values['s']), pa.array(values['d']),
        pa.nulls(count, type=pa.int64()), pa.array(np.full(count, n, dtype=np.int64)),
        pa.array(np.ones(count))], schema=schema)



def _frame_contacts(xyz,atom_residue,allowed,budget,parallel):
    workers=2 if parallel is True else int(parallel)
    tree=cKDTree(xyz)
    pair_bytes=int(tree.count_neighbors(tree,4.50001))*8
    if pair_bytes>budget//8:
        raise UseOriginal()
    if parallel and pair_bytes>budget//(8*workers):
        return None
    pairs=tree.query_pairs(4.50001,output_type='ndarray')
    minima=residue_minima(xyz,pairs,atom_residue,allowed,4.500001)
    finite=np.isfinite(minima)
    np.less_equal(minima,4.5,out=finite,where=finite)
    rr,ss=np.nonzero(finite)
    del finite
    if len(rr)*256>budget//8:
        raise UseOriginal()
    if parallel and len(rr)*256>budget//(8*workers):
        return None
    return rr,ss,minima[rr,ss]

def _ordered_frames(positions,atom_residue,allowed,budget):
    # One bool mask is reused in-place, including nonfinite-value semantics.
    # Per worker: float64 minima + bool mask =9 bytes/residue pair. Keep all
    # matrices within the previous quarter-budget reservation; never >6 frames.
    workers=next((w for w in (6,4,2) if 9*w*len(allowed)**2<=budget//4),1)
    if workers==1:
        for i,xyz in enumerate(positions):
            yield i,_frame_contacts(xyz,atom_residue,allowed,budget,False)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0,len(positions),workers):
            batch=positions[start:start+workers]
            results=list(executor.map(lambda xyz:_frame_contacts(xyz,atom_residue,allowed,budget,workers),batch))
            for offset,(xyz,result) in enumerate(zip(batch,results)):
                if result is None:
                    result=_frame_contacts(xyz,atom_residue,allowed,budget,False)
                yield start+offset,result

def calculate(ctx, writer):
    from .contacts import eligible, SCHEMA
    from .context import PROTEIN_NAMES
    protein = {r for r, p in ctx.res_atoms.items() if ctx.atoms['residue_name'][p[0]] in PROTEIN_NAMES}
    protein.update(ctx.atoms['residue_index'][i] for i in (ctx.annotation['alpha_carbons'] or []))
    if any(ctx.res_annotation.get(r, {}).get('run') is None for r in protein):
        raise ValueError('Contact neighbor exclusions require explicit continuous-segment annotations for protein residues')
    count = len(ctx.residues)
    blocks = (count + ctx.pair_block - 1) // ctx.pair_block
    # Bound handles and total Python buffering; unusual systems retain the old path.
    if blocks * (blocks + 1) // 2 > 512:
        return original(ctx, writer)
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
                for offset,(rr,ss,distances) in _ordered_frames(block.positions,atom_residue,allowed,ctx.budget):
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
                            files[key] = (Path(folder) / str(key)).open('wb', buffering=16384)
                            hashes[key] = hashlib.sha256(); lengths[key] = 0
                        files[key].write(raw); hashes[key].update(raw); lengths[key] += len(raw)
            for file in files.values(): file.close()
            # All scratch files must pass before even one completed part is published.
            for key, expected in hashes.items():
                path = Path(folder) / str(key)
                if path.stat().st_size != lengths[key] or lengths[key] % DTYPE.itemsize:
                    raise IOError('Spatial contact scratch length mismatch')
                with path.open('rb') as file:
                    actual = hashlib.file_digest(file, 'sha256').digest()
                if actual != expected.digest():
                    raise IOError('Spatial contact scratch checksum mismatch')
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
                        with path.open('rb') as file:
                            while True:
                                values = np.fromfile(file, dtype=DTYPE, count=limit)
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
