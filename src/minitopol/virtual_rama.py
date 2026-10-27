"""Experimental A-backed exact Ramachandran recipe, no stored angle arrays."""
import contextlib,hashlib,json
from pathlib import Path
import duckdb,pyarrow as pa
from filelock import FileLock
from minitopol.descriptors.context import AnalysisContext
from minitopol.descriptors.precision64 import backbone_frames64
from minitopol.descriptors.compact_output import _frames,_backbone_table
from minitopol.descriptors import ramachandran
from minitopol.descriptors.results import with_result_metadata,_payload_hash

class ReadOnlyWorkspace:
    def __init__(self,path,locks):self.path=Path(path);self.locks=Path(locks)
    def trajectory_path(self,tid):return self.path/'trajectories'/tid
    @contextlib.contextmanager
    def catalog(self):
        con=duckdb.connect(str(self.path/'integrated.duckdb'),read_only=True)
        try:yield con
        finally:con.close()
    def lock(self,name):return FileLock(str(self.locks/(hashlib.sha256(name.encode()).hexdigest()+'.lock')))

class NeverCommitted:
    def has(self,frame):return False

def tables(workspace,info):
    params=info['definition']['parameters']
    ctx=AnalysisContext(workspace,info['trajectory'],coordinate_mode=params['coordinate_mode'],initial_whole=params['initial_whole'],half_box_steps=params['half_box_steps'],memory_mb=128)
    assert ctx.annotation==params['annotation']
    batch=_frames(ctx,'ramachandran');parts=[];start=None
    for frame,time,angles,why in backbone_frames64(ctx,NeverCommitted()):
        if start is None:start=frame
        parts.append(_backbone_table('ramachandran',ramachandran,ctx,frame,time,angles,why))
        if (frame+1)%batch==0 or frame+1==ctx.n:
            base=pa.concat_tables(parts);yield f'frames-{start}-{frame+1}',base
            parts=[];start=None
    assert not parts

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def dependencies(workspace,tid):
    root=workspace.trajectory_path(tid)
    return {str(path.relative_to(root)):sha_file(path) for path in root.rglob('*') if path.is_file()}

def recipe(workspace,info):
    hashes=[dict(key=key,rows=len(table),logical_payload_sha256=_payload_hash(table)) for key,table in tables(workspace,info)]
    return dict(schema='VRAM1',info=info,source_module_sha256=sha_file(Path(__file__)),dependencies=dependencies(workspace,info['trajectory']),parts=hashes,contract='All logical angle values/null/reasons recomputed from exact A+annotation using fixed implementation. Existing angle arrays not stored. Reader CPU latency is material and must be measured. Not yet integrated with ResultStore/MCP/recovery.')

def read(workspace,manifest):
    if dependencies(workspace,manifest['info']['trajectory'])!=manifest['dependencies']:raise IOError('Virtual Rama dependency changed')
    if sha_file(Path(__file__))!=manifest['source_module_sha256']:raise IOError('Virtual reader source changed')
    count=0
    for (key,base),expected in zip(tables(workspace,manifest['info']),manifest['parts'],strict=True):
        if key!=expected['key'] or len(base)!=expected['rows'] or _payload_hash(base)!=expected['logical_payload_sha256']:raise IOError('Virtual scientific content differs')
        count+=len(base);yield with_result_metadata(base,manifest['info'])
