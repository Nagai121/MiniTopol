"""SCT3 reversible Gray-coded bitplanes; existing native SCT packing reused.

No scientific count changes: g[k]=binary[k] XOR binary[k+1], inverse prefix
XOR from plane9 down. Zstd3 remains unchanged. All decoded counts still must
be<=960. Shape-derived buffers are bounded before decompressing into outputs.
This module is explicit/local until its producer/reader integration is tested.
"""
import json,struct
import numpy as np
from numcodecs import Zstd
from .direct_counts import _packed_counts,sha
MAGIC=b'SCT3\x00\x00\x00\x00'
BUDGET=128*1024**2

def _bounds(n,m):
 if type(n) is not int or type(m) is not int or n<1 or m<1 or n>65536 or 64*n*m+32*n>BUDGET:raise ValueError('SCT3 shape/working buffer bound')

def _wrap(packed,n,m,start,times):
 raw=packed.tobytes();traw=np.asarray(times,dtype='<f8').tobytes();codec=Zstd(level=3);p=bytes(codec.encode(raw));t=bytes(codec.encode(traw))
 h=dict(schema=1,count_code='gray10-v1',frames=n,atoms=m,start=int(start),counts_length=len(p),time_length=len(t),counts_raw_sha256=sha(raw),time_raw_sha256=sha(traw),payload_sha256=sha(p+t))
 header=json.dumps(h,sort_keys=True,separators=(',',':')).encode()
 return MAGIC+struct.pack('<Q',len(header))+header+p+t

def encode_native(counts,frames,times):
 if counts.dtype!=np.dtype('u2') or counts.ndim!=2 or counts.max(initial=0)>960:raise ValueError('Native SASA counts')
 n,m=map(int,counts.shape);_bounds(n,m)
 if not np.array_equal(frames,np.arange(frames[0],frames[0]+n)) or len(times)!=n:raise ValueError('Native SASA frames')
 packed=_packed_counts(counts)
 for k in range(9):np.bitwise_xor(packed[k],packed[k+1],out=packed[k])
 return _wrap(packed,n,m,frames[0],times)

def packed_binary(raw):
 if len(raw)<16 or raw[:8]!=MAGIC:raise IOError('SCT3 magic/truncation')
 length=struct.unpack('<Q',raw[8:16])[0]
 if not 0<length<=4096 or len(raw)<16+length:raise IOError('SCT3 header bounds')
 h=json.loads(raw[16:16+length]);n,m=h['frames'],h['atoms'];_bounds(n,m)
 if h.get('schema')!=1 or h.get('count_code')!='gray10-v1' or type(h.get('start')) is not int:raise IOError('SCT3 code contract')
 p=raw[16+length:];a,b=h['counts_length'],h['time_length']
 if type(a) is not int or type(b) is not int or a<1 or b<1 or len(p)!=a+b or sha(p)!=h['payload_sha256']:raise IOError('SCT3 payload checksum/length')
 total=n*m;columns=(total+7)//8;packed=np.empty((10,columns),dtype='u1');timebytes=np.empty(n*8,dtype='u1');codec=Zstd(level=3)
 codec.decode(p[:a],out=packed);codec.decode(p[a:],out=timebytes)
 if sha(packed)!=h['counts_raw_sha256'] or sha(timebytes)!=h['time_raw_sha256']:raise IOError('SCT3 raw checksum')
 if total%8 and np.any(packed[:,-1] & np.uint8((0xff<<(total%8))&255)):raise IOError('SCT3 nonzero tail')
 for k in range(8,-1,-1):np.bitwise_xor(packed[k],packed[k+1],out=packed[k])
 invalid=packed[6]&packed[7]&packed[8]&packed[9]&np.bitwise_or.reduce(packed[:6],axis=0)
 if total%8:invalid[-1]&=(1<<(total%8))-1
 if np.any(invalid):raise IOError('SCT3 decoded accessibility bound')
 timebytes.flags.writeable=False
 return packed,timebytes,h

def extract(raw):
 packed,timebytes,h=packed_binary(raw);n,m=h['frames'],h['atoms'];total=n*m
 from .count_unpack import _unpacked_byte_counts
 counts=_unpacked_byte_counts(packed,n,m)
 if counts.max(initial=0)>960:raise IOError('SCT3 decoded accessibility bound')
 timebytes.flags.writeable=False
 return counts,np.arange(h['start'],h['start']+n),timebytes.view('<f8')
