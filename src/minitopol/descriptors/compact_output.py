"""Bounded atomic output batches preserving scientific values and row order."""
import numpy as np
import pyarrow as pa
from . import sasa
from .kernels import backbone_frames,secondary_labels
from .precision64 import backbone_frames64
from .correlation import atom_correlation

def _frames(ctx,name=None):
    if name=='sasa':
        # Native counts/bitplanes reserve64B per atom-frame in half budget.
        return max(1,min(1024,(ctx.budget//2)//max(1,ctx.m*64)))
    # Conservative upper bound for simultaneous Python rows, Arrow tables,
    # concat metadata and coordinates; fallback to a single frame if needed.
    if name not in ('sasa','ramachandran','secondary_structure'):
        return max(1,min(16,ctx.budget//max(1,ctx.m*1024*2)))
    # Columnar tables, streaming IPC hashing and constant dictionary metadata
    # remove the earlier repeated Python/string/output-buffer reservations.
    # Reserve half the working budget, conservatively 256 bytes/output row;
    # coordinate/FFT views keep the other half. No frame or atom is omitted.
    rows=ctx.m+len(ctx.residues)+1 if name=='sasa' else len(ctx.residues)*(2 if name=='ramachandran' else 1)
    return max(1,min(512,(ctx.budget//2)//max(1,rows*256)))

class _CommittedFrames:
    def __init__(self,ctx,writer,batch):self.ctx,self.writer,self.batch=ctx,writer,batch
    def has(self,frame):
        start=(frame//self.batch)*self.batch
        return self.writer.has(f'frames-{start}-{min(start+self.batch,self.ctx.n)}')


def _backbone_table(name,module,ctx,frame,time,angles,why):
    if name=='ramachandran':
        n=len(ctx.residues)*2
        values=np.column_stack((angles['phi'],angles['psi'])).ravel()
        valid=np.isfinite(values)
        # Match the previous Arrow builder's zero-filled null slots as well.
        values=np.where(valid,values,0.)
        return pa.Table.from_arrays([
            pa.array(np.full(n,frame,dtype=np.int64)),pa.array(np.full(n,time,dtype=np.float64)),
            pa.array(np.repeat(ctx.residues,2),type=pa.int64()),
            pa.array(['phi','psi']*len(ctx.residues),type=pa.string()),
            pa.array(values,mask=~valid,type=pa.float64()),
            pa.array(np.column_stack((why['phi'],why['psi'])).ravel().tolist(),type=pa.string())],schema=module.SCHEMA)
    labels=secondary_labels(angles['phi'],angles['psi']);n=len(ctx.residues)
    return pa.Table.from_arrays([
        pa.array(np.full(n,frame,dtype=np.int64)),pa.array(np.full(n,time,dtype=np.float64)),
        pa.array(ctx.residues,type=pa.int64()),pa.array(labels,type=pa.string()),
        pa.array([(why['phi'][i] or why['psi'][i]) if labels[i] is None else None
                  for i in range(n)],type=pa.string())],schema=module.SCHEMA)

def _secondary_columns(module,ctx,writer):
    batch=_frames(ctx,'secondary_structure');view=_CommittedFrames(ctx,writer,batch)
    frames=[];times=[];labels=[];reasons=[];start=None;n=len(ctx.residues)
    for frame,time,angles,why in backbone_frames(ctx,view):
        if start is None:start=frame
        values=secondary_labels(angles['phi'],angles['psi'])
        frames.append(frame);times.append(time);labels.extend(values)
        reasons.extend([(why['phi'][i] or why['psi'][i]) if values[i] is None else None for i in range(n)])
        if (frame+1)%batch==0 or frame+1==ctx.n:
            table=pa.Table.from_arrays([
                pa.array(np.repeat(np.asarray(frames,dtype='i8'),n)),
                pa.array(np.repeat(np.asarray(times,dtype='f8'),n)),
                pa.array(np.tile(np.asarray(ctx.residues,dtype='i8'),len(frames))),
                pa.array(labels,type=pa.string()),pa.array(reasons,type=pa.string())],schema=module.SCHEMA)
            writer.append(f'frames-{start}-{frame+1}',table)
            frames=[];times=[];labels=[];reasons=[];start=None
    assert not frames

def _rama_columns(module,ctx,writer):
    from ..backbone_columns import RamaBatch
    batch=_frames(ctx,'ramachandran');view=_CommittedFrames(ctx,writer,batch);builder=None
    for frame,time,angles,why in backbone_frames64(ctx,view):
        if builder is None:builder=RamaBatch(ctx.residues,module.SCHEMA)
        builder.append(frame,time,angles,why)
        if (frame+1)%batch==0 or frame+1==ctx.n:
            writer.append(f'frames-{builder.frames[0]}-{frame+1}',builder.table());builder=None
    assert builder is None

def _backbone(name,module,ctx,writer):
    if name=='ramachandran':return _rama_columns(module,ctx,writer)
    if name=='secondary_structure':return _secondary_columns(module,ctx,writer)
    batch=_frames(ctx,name);view=_CommittedFrames(ctx,writer,batch);tables=[];start=None
    frames=backbone_frames64(ctx,view) if name=='ramachandran' else backbone_frames(ctx,view)
    for frame,time,angles,why in frames:
        if start is None:start=frame
        table=_backbone_table(name,module,ctx,frame,time,angles,why)
        tables.append(table)
        if (frame+1)%batch==0 or frame+1==ctx.n:
            writer.append(f'frames-{start}-{frame+1}',pa.concat_tables(tables));tables=[];start=None
    assert not tables

def _sasa(ctx,writer):
    from .sasa_batch import table
    from .gpu_sasa_batch import evaluate_counts, fits_planned_reservation
    batch=_frames(ctx,'sasa');radii=ctx.atomic_values('radii');values_list=[];frames=[];times=[];start=None
    from .shared_gyration import prepare, observe, finish
    shared_shape = prepare(ctx)
    from .sasa_alignment import coordinate_blocks as sasa_coordinates
    for block in sasa_coordinates(ctx):
        observe(shared_shape, block)
        saved=getattr(ctx,'_correlation_block',None)
        held=getattr(saved[2],'nbytes',ctx.budget) if isinstance(saved,tuple) and len(saved)>2 else 0
        coordinates=sum(getattr(x,'nbytes',0) for x in (block.positions,getattr(block,'box',None),block.time))
        shared_held = (shared_shape['held'] if shared_shape is not None else 0)+getattr(ctx,'_sasa_alignment_held',0)
        available=ctx.budget//2-held-coordinates-ctx.budget//128-shared_held
        offset=0
        while offset<len(block.positions):
            frame=block.start+offset;first=(frame//batch)*batch;last=min(first+batch,ctx.n)
            count=min(32,len(block.positions)-offset,last-block.start-offset)
            # Conservative scheduling estimate only; evaluate_batch still checks
            # actual device shared memory, remaining working budget and VRAM.
            while count>1:
                payload=count*ctx.m*3*8+radii.nbytes+960*3*8+count*ctx.m*128*4+count*ctx.m*12+4
                if 2*payload+2*1024**2<=min(ctx.budget//2,available):break
                count=max(1,count//2)
            while count>1 and not fits_planned_reservation(count,ctx.m,ctx.budget,available):
                count=max(1,count//2)
            stop=offset+count
            if writer.has(f'frames-{first}-{last}'):
                offset=stop;continue
            results=evaluate_counts(block.positions[offset:stop],radii,ctx.budget,available=available)
            if results is None:
                results=[sasa._budgeted_sasa(xyz,radii,ctx.budget) if sasa._budgeted_sasa is not None else sasa.sasa_atoms(xyz,radii) for xyz in block.positions[offset:stop]]
                from ..sasa_counts import factors
                fac=factors(radii);floating=np.asarray(results,dtype='f8');estimate=np.rint(floating/fac*960)
                if not np.isfinite(estimate).all() or estimate.min(initial=0)<0 or estimate.max(initial=0)>960:raise IOError('CPU accessibility count bound')
                results=estimate.astype('<u2')
                if not np.array_equal((results.astype('f8')/960*fac).view('u8'),floating.view('u8')):raise IOError('CPU accessibility conversion not exact')
            for local,values in enumerate(results):
                frame=block.start+offset+local
                if start is None:start=frame
                values_list.append(values);frames.append(frame);times.append(block.time[offset+local])
                if frame+1==last:
                    writer.append_native(f'frames-{start}-{last}',np.asarray(values_list,dtype='<u2'),np.asarray(frames,dtype='<i8'),np.asarray(times,dtype='<f8'))
                    values_list=[];frames=[];times=[];start=None
            offset=stop
    assert not values_list
    finish(ctx, shared_shape)


def _time_correlation_points(ctx,writer):
    step=ctx.timestep();mean,variance=ctx.statistics()
    # Output batching only; FFT kernels/accumulation and coordinate chunks unchanged.
    output_chunk=max(1,min(ctx.n,ctx.budget//1024))
    for point in ctx.ca_indices():
        if writer.has(f'{point}-done'):continue
        residual=variance[point].sum();dots=None
        if residual>0:
            from .shared_ca_correlation import take
            dots=take(ctx,point,mean[point])
            if dots is None:dots,_=atom_correlation(ctx,point,mean[point],ctx.cache_path/'time-correlation-scratch.zarr')
        for start in range(0,ctx.n,output_chunk):
            stop=min(start+output_chunk,ctx.n);lag=np.arange(start,stop)
            value=dots[start:stop]/((ctx.n-lag)*residual) if residual>0 else np.zeros(len(lag))
            if residual>0:value[lag==0]=1
            table=pa.table({'point':np.full(len(lag),point),'residue':np.full(len(lag),ctx.atoms['residue_index'][point]),
                            'lag_frames':lag,'lag_ps':lag*step,'origins':ctx.n-lag,
                            'value':pa.array(value,mask=np.full(len(lag),residual==0)),
                            'reason':pa.array(['zero_variance' if residual==0 else None]*len(lag),type=pa.string())})
            writer.append(f'{point}-done' if stop==ctx.n else f'{point}-{start}',table)

def _timecorr_group_size(ctx):
    # Retained tables + concat + current arithmetic reserved conservatively.
    if ctx.n>max(1,ctx.budget//1024):return 1
    return max(1,min(8,(ctx.budget//2)//max(1,ctx.n*1024)))

class _TimecorrGroups:
    def __init__(self,writer,points,width):
        self.writer=writer;self.groups={};self.tables=[];self.current=None
        for start in range(0,len(points),width):
            group=points[start:start+width];key=f'points-{start}-{start+len(group)}-done'
            for i,p in enumerate(group):self.groups[f'{p}-done']=(key,i,len(group))
    def has(self,key):return self.writer.has(self.groups[str(key)][0])
    def append(self,key,table):
        target,index,count=self.groups[str(key)]
        if index!=len(self.tables) or (self.current is not None and self.current!=target):raise IOError('Timecorrelation group order')
        self.current=target;self.tables.append(table)
        if index+1==count:
            self.writer.append(target,pa.concat_tables(self.tables));self.tables=[];self.current=None
    def finish(self):
        if self.tables:raise IOError('Incomplete timecorrelation publication group')

def _time_correlation(ctx,writer):
    width=_timecorr_group_size(ctx)
    if width==1:return _time_correlation_points(ctx,writer)
    grouped=_TimecorrGroups(writer,list(map(int,ctx.ca_indices())),width)
    _time_correlation_points(ctx,grouped);grouped.finish()


def calculate(name,module,ctx,writer):
    if name=='sasa':
        from ..gpu_stage import lease
        def prepare_while_waiting():
            # Same existing bounded cache preparation, no extra worker/thread.
            try:ctx.prepare_aligned()
            except ValueError:
                # Unsupported alignment remains a later descriptor failure;
                # it must not change independent SASA's scientific support.
                return
            work=getattr(ctx,'_gpu_wait_work',None)
            if work is not None:work()
        with lease(on_wait=prepare_while_waiting):
            return _sasa(ctx,writer)
    if name in ('ramachandran','secondary_structure'):return _backbone(name,module,ctx,writer)
    if name=='time_correlation':return _time_correlation(ctx,writer)
    return module.calculate(ctx,writer)
