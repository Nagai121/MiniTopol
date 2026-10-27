"""SCT2: ten bitplanes, lossless SASA count blocks, exact reconstruction of scientific rows."""
import hashlib,json,os,struct
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
from numcodecs import Zstd
from minitopol.descriptors.sasa_batch import table as expanded_table
MAGIC=b'SCT2\x00\x00\x00\x00'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def factors(radii):
 inflated=radii+1.4;unique,inverse=np.unique(inflated,return_inverse=True)
 return np.asarray([4*np.pi*r**2 for r in unique])[inverse]
def context(metadata):
 residue=metadata['residue_index'];groups={r:[] for r in sorted(set(residue))}
 for i,r in enumerate(residue):groups[r].append(i)
 return SimpleNamespace(m=len(residue),res_atoms=groups,atoms={'residue_index':residue})
def encode(table,metadata):
 ctx=context(metadata);width=ctx.m+len(ctx.res_atoms)+1
 assert len(table)%width==0 and table['value'].null_count==0
 count=len(table)//width;values=table['value'].to_numpy().reshape(count,width)
 fac=np.frombuffer(bytes.fromhex(metadata['factors_f8_hex']),dtype='<f8')
 estimate=np.rint(values[:,:ctx.m]/fac*960)
 assert np.isfinite(estimate).all() and estimate.min(initial=0)>=0 and estimate.max(initial=0)<=960
 counts=estimate.astype('<u2')
 # This assertion makes the encoding exact rather than an approximation.
 assert np.array_equal(((counts.astype('f8')/960)*fac).view('u8'),values[:,:ctx.m].view('u8'))
 frames=table['frame'].to_numpy().reshape(count,width)[:,0]
 assert np.array_equal(frames,np.arange(frames[0],frames[0]+count))
 times=table['time_ps'].to_numpy().reshape(count,width)[:,0].astype('<f8')
 lanes=np.ascontiguousarray(counts.T).ravel()
 bits=((lanes[None,:]>>np.arange(10,dtype='<u2')[:,None])&1).astype('u1')
 raw=np.packbits(bits,axis=1,bitorder='little').tobytes();traw=times.tobytes();codec=Zstd(level=3)
 payload=bytes(codec.encode(raw));tpayload=bytes(codec.encode(traw))
 header=dict(schema=1,frames=count,atoms=ctx.m,start=int(frames[0]),counts_length=len(payload),time_length=len(tpayload),counts_raw_sha256=sha(raw),time_raw_sha256=sha(traw),payload_sha256=sha(payload+tpayload))
 h=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
 return MAGIC+struct.pack('<Q',len(h))+h+payload+tpayload
def decode(raw,metadata,*,start=None,stop=None):
 if raw[:8]!=MAGIC:raise IOError('SASA count block magic mismatch')
 if len(raw)<16:raise IOError('Truncated count block')
 size=struct.unpack('<Q',raw[8:16])[0]
 if not 0<size<=4096 or 16+size>len(raw):raise IOError('Invalid count header length')
 h=json.loads(raw[16:16+size]);payload=raw[16+size:]
 if len(payload)!=h['counts_length']+h['time_length'] or sha(payload)!=h['payload_sha256']:raise IOError('Count block payload checksum/size mismatch')
 codec=Zstd(level=3);a=bytes(codec.decode(payload[:h['counts_length']]));b=bytes(codec.decode(payload[h['counts_length']:]))
 if sha(a)!=h['counts_raw_sha256'] or sha(b)!=h['time_raw_sha256']:raise IOError('Count decoded payload checksum mismatch')
 m=h['atoms'];n=h['frames']
 packed_columns=(m*n+7)//8
 if len(a)!=packed_columns*10 or len(b)!=n*8 or m!=len(metadata['residue_index']):raise IOError('Count axes size mismatch')
 bits=np.unpackbits(np.frombuffer(a,dtype='u1').reshape(10,packed_columns),axis=1,count=m*n,bitorder='little').astype('<u2')
 lanes=np.sum(bits<<np.arange(10,dtype='<u2')[:,None],axis=0,dtype='<u2')
 counts=lanes.reshape(m,n).T
 if counts.max(initial=0)>960:raise IOError('Invalid accessible count')
 fac=np.frombuffer(bytes.fromhex(metadata['factors_f8_hex']),dtype='<f8')
 first=max(0,min(n,(start-h['start']) if start is not None else 0))
 last=max(first,min(n,(stop-h['start']) if stop is not None else n))
 values=counts[first:last].astype('f8')/960*fac
 times=np.frombuffer(b,dtype='<f8')[first:last];frames=np.arange(h['start']+first,h['start']+last)
 restored=expanded_table(context(metadata),frames,times,values)
 return restored.replace_schema_metadata({b'minitopol':metadata['scientific_metadata'].encode()})
def publish(path,raw):
 path=Path(path);temporary=path.with_suffix('.tmp')
 with temporary.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 os.replace(temporary,path)

from minitopol.descriptors.sasa import SCHEMA
def expanded_table(ctx,frames,times,values):
    m=ctx.m;residues=list(ctx.res_atoms);nr=len(residues);width=m+nr+1;count=len(frames)
    columns=np.empty((count,width),dtype=np.float64)
    values=np.ascontiguousarray(values,dtype=np.float64)
    columns[:,:m]=values
    for j,points in enumerate(ctx.res_atoms.values()):
        columns[:,m+j]=np.ascontiguousarray(values[:,points]).sum(axis=1)
    columns[:,-1]=values.sum(axis=1)
    point=np.r_[np.arange(m,dtype=np.int64),np.zeros(nr+1,dtype=np.int64)]
    residue=np.r_[np.asarray(ctx.atoms['residue_index'],dtype=np.int64),np.asarray(residues,dtype=np.int64),np.zeros(1,dtype=np.int64)]
    point_mask=np.r_[np.zeros(m,dtype=bool),np.ones(nr+1,dtype=bool)]
    residue_mask=np.r_[np.zeros(m+nr,dtype=bool),np.ones(1,dtype=bool)]
    return pa.Table.from_arrays([
        pa.array(np.repeat(np.asarray(frames,dtype=np.int64),width)),
        pa.array(np.repeat(np.asarray(times,dtype=np.float64),width)),
        pa.array((['atom']*m+['residue']*nr+['total'])*count,type=pa.string()),
        pa.array(np.tile(point,count),mask=np.tile(point_mask,count)),
        pa.array(np.tile(residue,count),mask=np.tile(residue_mask,count)),
        pa.array(columns.ravel())],schema=SCHEMA)
