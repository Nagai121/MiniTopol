"""Nonperiodic Contacts fallback preserving part ordering and keys."""
import sqlite3,tempfile,hashlib
import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree
from .accelerated import residue_minima


class UseOriginal(Exception):
    pass


def _frame_hash(rows):
    return hashlib.sha256(np.asarray(rows,dtype=[('r','<i8'),('s','<i8'),('d','<f8')]).tobytes()).digest()


def _verify_scratch(con,expected):
    con.execute('CREATE INDEX by_frame ON states(frame,r,s)')
    for frame,digest in enumerate(expected):
        rows=con.execute('SELECT r,s,d FROM states WHERE frame=? ORDER BY r,s',[frame]).fetchall()
        if _frame_hash(rows)!=digest:
            raise IOError('Spatial contact scratch checksum mismatch')
    con.execute('DROP INDEX by_frame')


def calculate(ctx,writer):
    from .contacts import eligible,SCHEMA,minimum_residue_distance
    from .context import PROTEIN_NAMES
    protein={r for r,points in ctx.res_atoms.items() if ctx.atoms['residue_name'][points[0]] in PROTEIN_NAMES}
    protein.update(ctx.atoms['residue_index'][i] for i in (ctx.annotation['alpha_carbons'] or []))
    if any(ctx.res_annotation.get(r,{}).get('run') is None for r in protein):
        raise ValueError('Contact neighbor exclusions require explicit continuous-segment annotations for protein residues')
    if ctx.budget < 32*1024**2 or 9*len(ctx.residues)**2 > ctx.budget//4:
        raise UseOriginal()
    residues=np.asarray(ctx.residues,dtype=np.int64);ordinal={int(r):i for i,r in enumerate(residues)}
    atom_residue=np.asarray([ordinal[r] for r in ctx.atoms['residue_index']],dtype=np.int64)
    allowed=np.zeros((len(residues),len(residues)),dtype=np.bool_)
    for r in range(len(residues)):
        for s in range(r+1,len(residues)):
            allowed[r,s]=eligible(ctx,int(residues[r]),int(residues[s]))
    times=np.empty(ctx.n);digests=[]
    ctx.cache_path.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='contacts-spatial-',dir=ctx.cache_path) as scratch:
        con=sqlite3.connect(scratch+'/states.sqlite')
        try:
            # Rebuildable scratch only; published parts still use normal fsync,
            # hashes, journal commit and completion verification.
            con.execute('PRAGMA journal_mode=OFF');con.execute('PRAGMA synchronous=OFF')
            con.execute('PRAGMA cache_size=-8192');con.execute('PRAGMA temp_store=FILE')
            con.execute('CREATE TABLE states(bi INTEGER,bj INTEGER,frame INTEGER,r INTEGER,s INTEGER,d REAL)')
            for block in ctx.registry.blocks(ctx.trajectory,chunk_frames=ctx.chunk):
                for offset,xyz in enumerate(block.positions):
                    frame=block.start+offset;times[frame]=block.time[offset]
                    # Margin exceeds float32 subtraction error at a 4.5 A cutoff.
                    # Every emitted distance is recomputed with original precision.
                    tree=cKDTree(xyz)
                    neighbor_counts=tree.query_ball_point(xyz,4.50001,return_length=True)
                    if int(neighbor_counts.sum())*8 > ctx.budget//8:
                        raise UseOriginal()
                    pairs=tree.query_pairs(4.50001,output_type='ndarray')
                    minima=residue_minima(xyz,pairs,atom_residue,allowed,4.500001)
                    rr,ss=np.nonzero(np.isfinite(minima))
                    if len(rr)*256 > ctx.budget//8:
                        raise UseOriginal()
                    # Strict float32 displacements and float64 distance arithmetic
                    # are verified against the original distance kernel; retain threshold.
                    rows=[]
                    for r,s in zip(rr,ss):
                        left,right=int(residues[r]),int(residues[s])
                        distance=float(minima[r,s])
                        if distance<=4.5:rows.append((left,right,distance))
                    digests.append(_frame_hash(rows))
                    con.executemany('INSERT INTO states VALUES (?,?,?,?,?,?)',
                        ((ordinal[r]//ctx.pair_block,ordinal[s]//ctx.pair_block,frame,r,s,d) for r,s,d in rows))
                con.commit()
            _verify_scratch(con,digests)
            con.execute('CREATE INDEX by_block ON states(bi,bj,frame,r,s)')
            for left in range(0,len(residues),ctx.pair_block):
                for right in range(left,len(residues),ctx.pair_block):
                    key=f'frequency-{left}-{right}'
                    if writer.has(key):continue
                    pairs=[(int(r),int(s)) for r in residues[left:left+ctx.pair_block] for s in residues[right:right+ctx.pair_block]
                           if r<s and eligible(ctx,int(r),int(s))]
                    if not pairs:
                        writer.append(key,pa.Table.from_pylist([],schema=SCHEMA));continue
                    counts={pair:0 for pair in pairs}
                    cursor=con.execute('SELECT frame,r,s,d FROM states WHERE bi=? AND bj=? ORDER BY frame,r,s',
                                       [left//ctx.pair_block,right//ctx.pair_block])
                    current=cursor.fetchone()
                    rows=[];part=0
                    limit=max(4096,min(16384,ctx.budget//16384))
                    while current is not None:
                        frame,r,s,distance=current;counts[r,s]+=1
                        rows.append(dict(level='state',frame=frame,time_ps=times[frame],residue_i=r,residue_j=s,
                                         distance_angstrom=distance,contact_frames=None,frames=ctx.n,value=1.))
                        if len(rows)>=limit:
                            writer.append(f'state-compact-{left}-{right}-{part}',pa.Table.from_pylist(rows,schema=SCHEMA))
                            rows=[];part+=1
                        current=cursor.fetchone()
                    writer.append(f'state-compact-{left}-{right}-{part}',pa.Table.from_pylist(rows,schema=SCHEMA))
                    rows=[dict(level='frequency',frame=None,time_ps=None,residue_i=r,residue_j=s,distance_angstrom=None,
                               contact_frames=count,frames=ctx.n,value=count/ctx.n) for (r,s),count in counts.items() if count]
                    writer.append(key,pa.Table.from_pylist(rows,schema=SCHEMA))
        finally:con.close()
