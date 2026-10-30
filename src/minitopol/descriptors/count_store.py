"""SASA schema4 count parts with scientific checksums and durable resume."""
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from .. import sasa_counts as codec
from ..storage import atomic_json,canonical,digest,fingerprint,read_json
from .context import AnalysisContext
from .datasets import DatasetWriter,validate_table
from .results import _payload_hash,with_result_metadata

def axes_path(path):return path.parent/'axes'
def verify_part(path,part,info,*,expand=True):
    if not path.is_file() or fingerprint(path)['sha256']!=part['sha256']:raise IOError('SASA count part checksum/missing file')
    axes=axes_path(path)
    if not axes.is_file() or fingerprint(axes)['sha256']!=part['axes_sha256']:raise IOError('SASA count axes checksum/missing file')
    meta=read_json(axes)
    if meta['scientific_metadata']!=canonical(info):raise IOError('SASA scientific metadata mismatch')
    if part.get('hash_kind')=='native-counts-v1':
        from ..direct_counts import extract
        counts,frames,times=extract(path.read_bytes())
        width=len(meta['residue_index'])+len(set(meta['residue_index']))+1
        if counts.shape[1]!=len(meta['residue_index']) or len(frames)*width!=part['rows'] or native_hash(counts,frames,times,meta)!=part['payload_hash']:raise IOError('Native SASA scientific hash/rows')
        if not expand:return None
        return codec.decode(path.read_bytes(),meta)
    table=codec.decode(path.read_bytes(),meta)
    if len(table)!=part['rows'] or _payload_hash(table.replace_schema_metadata(None))!=part['payload_hash']:raise IOError('SASA logical payload checksum/rows mismatch')
    return table

def verified_range(path,part,info,start,stop):
    # Completion already validates full scientific payload; immutable byte hashes
    # bind every later range read to that verified result without full row expansion.
    if fingerprint(path)['sha256']!=part['sha256']:raise IOError('SASA range part checksum')
    meta_path=axes_path(path)
    if fingerprint(meta_path)['sha256']!=part['axes_sha256']:raise IOError('SASA range axes checksum')
    meta=read_json(meta_path)
    if meta['scientific_metadata']!=canonical(info):raise IOError('SASA range scientific metadata')
    return codec.decode(path.read_bytes(),meta,start=start,stop=stop)

def verify_dataset(path,completed):
    if completed.get('schema') not in (4,6) or digest(completed['info'])!=path.stem or not completed['parts']:raise IOError('SASA completion identity mismatch')
    rows=0;keys=set()
    for index,part in enumerate(completed['parts']):
        if part['file']!=f'{path.stem}/part-{index:08d}.sct' or part['key'] in keys:raise IOError('SASA count part order mismatch')
        verify_part(path.parent/part['file'],part,completed['info'],expand=False);rows+=part['rows'];keys.add(part['key'])
    if rows!=completed['rows']:raise IOError('SASA completed row count mismatch')

def logical_tables(workspace,relative,*,batch_size=65536,start=None,stop=None):
    path=workspace.path/relative;completed=read_json(path.with_suffix('.json'))
    if completed.get('schema') not in (4,6):
        for part in completed['parts']:
            for batch in pq.ParquetFile(path.parent/part['file']).iter_batches(batch_size=batch_size):yield pa.Table.from_batches([batch])
        return
    selected=[]
    for part in completed['parts']:
        import re
        match=re.fullmatch(r'frames-(\d+)-(\d+)',part['key'])
        if match:
            first,last=map(int,match.groups())
            if (start is not None and last<=start) or (stop is not None and first>=stop):continue
        selected.append(part)
    # Preserve an empty query's exact schema through one validated part.
    if not selected:selected=completed['parts'][:1]
    for part in selected:
        if part.get('hash_kind')=='native-counts-v1' and start is None and stop is None:
            source=path.parent/part['file'];axis=axes_path(source)
            if not axis.is_file() or fingerprint(axis)['sha256']!=part['axes_sha256']:raise IOError('SASA native stream axes checksum')
            meta=read_json(axis)
            if meta['scientific_metadata']!=canonical(completed['info']):raise IOError('SASA native stream scientific metadata')
            from ..native_windows import tables
            for table in tables(source.read_bytes(),meta,part):
                for batch in table.to_batches(max_chunksize=batch_size):
                    yield with_result_metadata(pa.Table.from_batches([batch]).replace_schema_metadata(None),completed['info'])
            continue
        table=(verified_range(path.parent/part['file'],part,completed['info'],start,stop) if start is not None or stop is not None else verify_part(path.parent/part['file'],part,completed['info']))
        if len(table)==0:
            yield with_result_metadata(table.replace_schema_metadata(None),completed['info'])
        for batch in table.to_batches(max_chunksize=batch_size):
            yield with_result_metadata(pa.Table.from_batches([batch]).replace_schema_metadata(None),completed['info'])

def read_rows(workspace,relative,limit):
    if limit<=0:return []
    import re
    path=workspace.path/relative;completed=read_json(path.with_suffix('.json'))
    first=None;last=None
    if completed.get('schema') in (4,6):
        match=re.fullmatch(r'frames-(\d+)-(\d+)',completed['parts'][0]['key'])
        if match:
            meta=read_json(axes_path(path.parent/completed['parts'][0]['file']))
            residues=meta['residue_index'];width=len(residues)+len(set(residues))+1
            first=int(match.group(1));last=first+(limit+width-1)//width
    rows=[]
    for table in logical_tables(workspace,relative,batch_size=max(1,limit),start=first,stop=last):
        rows.extend(table.slice(0,limit-len(rows)).to_pylist())
        if len(rows)>=limit:break
    return rows

def query(workspace,*,trajectory=None,definition_id=None,columns=None,start=None,stop=None):
    clauses=["descriptor='sasa'"];args=[]
    if trajectory is not None:clauses.append('trajectory=?');args.append(trajectory)
    if definition_id is not None:clauses.append('definition=?');args.append(canonical(read_json_definition(workspace,definition_id)))
    with workspace.catalog() as con:paths=con.execute('SELECT path FROM results WHERE '+' AND '.join(clauses)+' ORDER BY id',args).fetchall()
    if not paths:raise LookupError('No completed SASA results match')
    tables=[]
    for (relative,) in paths:
        for table in logical_tables(workspace,relative,start=start,stop=stop):
            if start is not None:table=table.filter(pc.greater_equal(table['frame'],int(start)))
            if stop is not None:table=table.filter(pc.less(table['frame'],int(stop)))
            if columns:table=table.select(columns)
            tables.append(table)
    return pa.concat_tables(tables)

def read_json_definition(workspace,definition_id):
    import json
    with workspace.catalog() as con:row=con.execute('SELECT definition FROM descriptor_definitions WHERE id=?',[definition_id]).fetchone()
    if row is None:raise LookupError('Unknown descriptor definition')
    return json.loads(row[0])

def native_hash(counts,frames,times,meta):
    import hashlib,numpy as np
    h=hashlib.sha256(b'SASA-native-counts-v1')
    for value,dtype in ((counts,'<u2'),(frames,'<i8'),(times,'<f8')):h.update(np.ascontiguousarray(value,dtype=dtype).tobytes())
    h.update(canonical(meta).encode());return h.hexdigest()

class CountWriter(DatasetWriter):
    def __enter__(self):
        self.lock.acquire()
        try:
            if self.marker.exists():
                completed=read_json(self.marker)
                if completed['info']!=self.info:raise IOError('SASA dataset definition changed')
                verify_dataset(self.path,completed);self.store._register(self.result_id,self.relative,completed);self.complete=True;self.state=completed
            elif self.progress.exists():
                from .progress_journal import load
                self.state,self.journal=load(self.progress);self.state['schema']=6
                if self.state['info']!=self.info or self.state['layout']!=self.layout:raise ValueError('Resume count storage with same layout')
                for index,part in enumerate(self.state['parts']):
                    if part['file']!=f'{self.result_id}/part-{index:08d}.sct':raise IOError('Old/invalid SASA partial format; use separate Workspace')
                    verify_part(self.path.parent/part['file'],part,self.info,expand=False)
            else:
                self.directory.mkdir(parents=True,exist_ok=True);self.state=dict(schema=6,info=self.info,layout=self.layout,parts=[])
                from .progress_journal import start
                start(self.progress,self.state);self.journal=True
            self.by_key={p['key']:p for p in self.state['parts']}
            if not self.complete:
                params=self.info['definition']['parameters'];ctx=AnalysisContext(self.workspace,self.info['trajectory'],coordinate_mode=params['coordinate_mode'],initial_whole=params['initial_whole'],half_box_steps=params['half_box_steps'],memory_mb=128)
                if ctx.annotation!=params['annotation']:raise ValueError('Atomic annotation changed during SASA calculation')
                meta=dict(schema=2,trajectory=self.info['trajectory'],residue_index=ctx.atoms['residue_index'],factors_f8_hex=codec.factors(ctx.atomic_values('radii')).astype('<f8').tobytes().hex(),scientific_metadata=canonical(self.info))
                axes=self.directory/'axes'
                if axes.exists():
                    if read_json(axes)!=meta:raise IOError('SASA count axes differ')
                else:
                    if self.state['parts']:raise IOError('SASA axes missing from partial data')
                    atomic_json(axes,meta)
                self.count_meta=meta;self.axes_sha256=fingerprint(axes)['sha256']
            return self
        except BaseException:self.lock.release();raise

    def append(self,key,table):
        if self.complete:raise ValueError('Completed count datasets cannot be overwritten')
        validate_table(table,self.info['stop']);key=str(key);checksum=_payload_hash(table)
        if self.has(key):
            if self.by_key[key]['payload_hash']!=checksum:raise ValueError('Resumed SASA values differ')
            return
        file=f'{self.result_id}/part-{len(self.state["parts"]):08d}.sct';child=self.path.parent/file
        raw=codec.encode(table,self.count_meta);codec.publish(child,raw)
        part=dict(key=key,file=file,rows=len(table),sha256=fingerprint(child)['sha256'],axes_sha256=self.axes_sha256,payload_hash=checksum,undefined_values=table['value'].null_count)
        self.state['parts'].append(part)
        if self.journal:
            from .progress_journal import append
            append(self.progress,len(self.state['parts'])-1,part)
        else:atomic_json(self.progress,self.state)
        self.by_key[key]=part

    def append_native(self,key,counts,frames,times):
        import numpy as np
        from ..direct_counts import encode_native
        if self.complete:raise ValueError('Completed native count dataset')
        key=str(key);counts=np.asarray(counts);frames=np.asarray(frames);times=np.asarray(times)
        if not len(frames) or frames[0]<self.info['start'] or frames[-1]>=self.info['stop'] or counts.shape!=(len(frames),len(self.count_meta['residue_index'])) or not np.isfinite(times).all():raise ValueError('Native SASA axes/range')
        checksum=native_hash(counts,frames,times,self.count_meta)
        if self.has(key):
            if self.by_key[key]['payload_hash']!=checksum:raise ValueError('Resumed native counts differ')
            return
        raw=encode_native(counts,frames,times)
        file=f'{self.result_id}/part-{len(self.state["parts"]):08d}.sct';child=self.path.parent/file;codec.publish(child,raw)
        width=counts.shape[1]+len(set(self.count_meta['residue_index']))+1
        part=dict(key=key,file=file,rows=len(frames)*width,sha256=fingerprint(child)['sha256'],axes_sha256=self.axes_sha256,payload_hash=checksum,hash_kind='native-counts-v1',undefined_values=0)
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
