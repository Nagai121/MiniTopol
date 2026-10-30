"""Schema5: computed-and-hashed A-backed Ramachandran, regenerated on read."""
import contextlib,hashlib,json,sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import pyarrow as pa
from .. import virtual_rama as reader
from ..storage import atomic_json,canonical,digest,fingerprint,read_json
from .datasets import DatasetWriter,validate_table
from .results import _payload_hash,with_result_metadata

def environment():
    packages={}
    for package in ['numpy','scipy','MDAnalysis','numba','pyarrow','zarr','numcodecs']:
        try:packages[package]=version(package)
        except PackageNotFoundError:packages[package]=None
    return dict(python=sys.version,packages=packages)
def source_hashes():
    root=Path(__file__).resolve().parents[1]
    names=['descriptors/sasa_alignment.py','descriptors/aligned_rmsd_handoff.py','descriptors/alignment_cache.py','descriptors/shared_aligned_ca.py','descriptors/rmsd.py','descriptors/dynamic_correlation.py','backbone_columns.py','descriptors/shared_raw_backbone.py','descriptors/builtins.py','position_moddelta.py','position_xor.py','position_grid.py','agx_zarr.py','virtual_rama.py','registry.py','coordinates.py','annotations.py','descriptors/virtual_rama_store.py','descriptors/precision64.py','descriptors/kernels.py','descriptors/context.py','descriptors/compact_output.py','descriptors/ramachandran.py']
    return {name:reader.sha_file(root/name) for name in names}
def dependencies(workspace,tid):
    root=workspace.trajectory_path(tid)
    paths=[root/name for name in ['manifest.json','atoms.parquet','annotations.parquet']]
    paths+=sorted(p for p in (root/'A.zarr').rglob('*') if p.is_file())
    return {str(p.relative_to(root)):reader.sha_file(p) if p.is_file() else None for p in paths}
def axes(path):return path.parent/'recipe-axes'
def workspace_root(path):
    for parent in path.parents:
        if (parent/'integrated.duckdb').is_file():return parent
    raise IOError('Virtual result outside Workspace')
def verify_axes(path,info):
    from ..workspace import Workspace
    meta=read_json(axes(path));root=workspace_root(path);workspace=reader.ReadOnlyWorkspace(root,root/'locks')
    if meta['info']!=info or meta['source_hashes']!=source_hashes() or meta['environment']!=environment():raise IOError('Virtual recipe scientific/source/runtime contract changed')
    if dependencies(workspace,info['trajectory'])!=meta['dependencies']:raise IOError('Virtual A/annotation dependency changed')
    return meta
def verify_part(path,part,info):
    if fingerprint(path)['sha256']!=part['sha256'] or fingerprint(axes(path))['sha256']!=part['axes_sha256']:raise IOError('Virtual recipe checksum')
    saved=read_json(path)
    if saved!=dict(key=part['key'],rows=part['rows'],payload_hash=part['payload_hash'],undefined_values=part['undefined_values'],info_sha256=digest(info)):raise IOError('Virtual part mismatch')
    return saved
def verify_dataset(path,completed):
    if completed.get('schema')!=5 or digest(completed['info'])!=path.stem or not completed['parts']:raise IOError('Virtual dataset identity')
    verify_axes(path.parent/completed['parts'][0]['file'],completed['info']);rows=0;keys=set()
    for i,part in enumerate(completed['parts']):
        if part['file']!=f'{path.stem}/part-{i:08d}.vram' or part['key'] in keys:raise IOError('Virtual part order/duplicate')
        verify_part(path.parent/part['file'],part,completed['info']);rows+=part['rows'];keys.add(part['key'])
    if rows!=completed['rows']:raise IOError('Virtual dataset row count')
class RecipeWriter(DatasetWriter):
    def __enter__(self):
        self.lock.acquire()
        try:
            if self.marker.exists():
                completed=read_json(self.marker)
                if completed['info']!=self.info:raise IOError('Virtual scientific definition changed')
                verify_dataset(self.path,completed);self.store._register(self.result_id,self.relative,completed);self.complete=True;self.state=completed
            elif self.progress.exists():
                from .progress_journal import load
                self.state,self.journal=load(self.progress);self.state['schema']=5
                if self.state['info']!=self.info or self.state['layout']!=self.layout:raise ValueError('Virtual partial requires identical layout')
                if self.state['parts']:verify_axes(self.path.parent/self.state['parts'][0]['file'],self.info)
                for i,part in enumerate(self.state['parts']):
                    if part['file']!=f'{self.result_id}/part-{i:08d}.vram':raise IOError('Wrong partial format')
                    verify_part(self.path.parent/part['file'],part,self.info)
            else:
                self.directory.mkdir(parents=True,exist_ok=True);self.state=dict(schema=5,info=self.info,layout=self.layout,parts=[])
                from .progress_journal import start
                start(self.progress,self.state);self.journal=True
            self.by_key={p['key']:p for p in self.state['parts']}
            if not self.complete:
                meta=dict(schema='VRAM2',info=self.info,source_hashes=source_hashes(),environment=environment(),dependencies=dependencies(self.workspace,self.info['trajectory']))
                target=self.directory/'recipe-axes'
                if target.exists():
                    if read_json(target)!=meta:raise IOError('Virtual dependency/source axes changed')
                else:atomic_json(target,meta)
                self.axes_sha256=fingerprint(target)['sha256']
            return self
        except BaseException:self.lock.release();raise
    def append(self,key,table):
        if self.complete:raise ValueError('Completed virtual result immutable')
        validate_table(table,self.info['stop']);key=str(key);checksum=_payload_hash(table)
        if self.has(key):
            if self.by_key[key]['payload_hash']!=checksum:raise ValueError('Resumed virtual values differ')
            return
        file=f'{self.result_id}/part-{len(self.state["parts"]):08d}.vram';target=self.path.parent/file
        atomic_json(target,dict(key=key,rows=len(table),payload_hash=checksum,undefined_values=table['value'].null_count,info_sha256=digest(self.info)))
        part=dict(key=key,file=file,rows=len(table),sha256=fingerprint(target)['sha256'],axes_sha256=self.axes_sha256,payload_hash=checksum,undefined_values=table['value'].null_count)
        self.state['parts'].append(part)
        if self.journal:
            from .progress_journal import append
            append(self.progress,len(self.state['parts'])-1,part)
        else:atomic_json(self.progress,self.state)
        self.by_key[key]=part
    def __exit__(self,kind,error,traceback):
        try:
            if kind is None and not self.complete:
                completed={**self.state,'rows':sum(p['rows'] for p in self.state['parts'])};verify_dataset(self.path,completed);atomic_json(self.marker,completed);self.store._register(self.result_id,self.relative,completed);self.state=completed;self.progress.unlink(missing_ok=True)
        finally:self.lock.release()

def logical_tables(workspace,relative,*,connection=None,start=None,stop=None):
    path=workspace.path/relative;completed=read_json(path.with_suffix('.json'));verify_dataset(path,completed)
    proxy=reader.ReadOnlyWorkspace(workspace.path,workspace.path/'locks')
    if connection is not None:
        @contextlib.contextmanager
        def borrowed():yield connection
        proxy.catalog=borrowed
    parts=completed['parts'];keys=None
    if start is not None or stop is not None:
        import re
        selected=[]
        for part in parts:
            match=re.fullmatch(r'frames-(\d+)-(\d+)',part['key'])
            if match:
                first,last=map(int,match.groups())
                if (start is not None and last<=start) or (stop is not None and first>=stop):continue
            selected.append(part)
        parts=selected if selected else parts[:1]
        keys={part['key'] for part in parts}
    for (key,base),part in zip(reader.tables(proxy,completed['info'],keys=keys),parts,strict=True):
        if key!=part['key'] or len(base)!=part['rows'] or _payload_hash(base)!=part['payload_hash'] or base['value'].null_count!=part['undefined_values']:raise IOError('Virtual reconstruction scientific payload differs')
        yield with_result_metadata(base,completed['info'])
def read_rows(workspace,relative,limit,*,connection=None):
    rows=[]
    for table in logical_tables(workspace,relative,connection=connection):
        rows.extend(table.slice(0,limit-len(rows)).to_pylist())
        if len(rows)>=limit:break
    return rows
def query(workspace,*,trajectory=None,definition_id=None,columns=None,start=None,stop=None):
    import pyarrow.compute as pc
    clauses=["descriptor='ramachandran'"];args=[]
    if trajectory is not None:clauses.append('trajectory=?');args.append(trajectory)
    if definition_id is not None:clauses.append('json_extract_string(definition,\'$.name\')=\'ramachandran\'')
    with workspace.catalog() as con:paths=con.execute('SELECT path FROM results WHERE '+' AND '.join(clauses)+' ORDER BY id',args).fetchall()
    if not paths:raise LookupError('No virtual Rama results')
    tables=[]
    for (relative,) in paths:
        for table in logical_tables(workspace,relative,start=start,stop=stop):
            if definition_id is not None and table['definition_id'][0].as_py()!=definition_id:continue
            if start is not None:table=table.filter(pc.greater_equal(table['frame'],int(start)))
            if stop is not None:table=table.filter(pc.less(table['frame'],int(stop)))
            if columns:table=table.select(columns)
            tables.append(table)
    return pa.concat_tables(tables)
