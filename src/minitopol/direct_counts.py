"""Encode native SASA count matrices without expanded floating Arrow rows."""
import hashlib,json,struct
import numpy as np
from numcodecs import Zstd
try:
 from numba import njit
except ImportError:
 def njit(**options):
  return lambda function:function
MAGIC=b'SCT2\x00\x00\x00\x00'
def sha(x):return hashlib.sha256(x).hexdigest()
def extract(raw):
 assert raw[:8]==MAGIC
 length=struct.unpack('<Q',raw[8:16])[0];assert 0<length<=4096
 h=json.loads(raw[16:16+length]);p=raw[16+length:];assert len(p)==h['counts_length']+h['time_length'] and sha(p)==h['payload_sha256']
 codec=Zstd(level=3);a=bytes(codec.decode(p[:h['counts_length']]));b=bytes(codec.decode(p[h['counts_length']:]))
 assert sha(a)==h['counts_raw_sha256'] and sha(b)==h['time_raw_sha256']
 m,n=h['atoms'],h['frames'];cols=(m*n+7)//8;assert len(a)==cols*10 and len(b)==n*8
 bits=np.unpackbits(np.frombuffer(a,'u1').reshape(10,cols),axis=1,count=m*n,bitorder='little').astype('<u2')
 counts=np.sum(bits<<np.arange(10,dtype='<u2')[:,None],axis=0,dtype='<u2').reshape(m,n).T
 assert counts.max(initial=0)<=960
 return counts,np.arange(h['start'],h['start']+n),np.frombuffer(b,dtype='<f8')
@njit(nogil=True,cache=True)
def _packed_counts(counts):
 n,m=counts.shape;total=n*m
 packed=np.zeros((10,(total+7)//8),dtype=np.uint8)
 for index in range(total):
  value=counts[index%n,index//n];column=index//8;shift=index%8
  for bit in range(10):
   packed[bit,column]|=((value>>bit)&1)<<shift
 return packed

def encode_native(counts,frames,times):
 if counts.dtype!=np.dtype('u2') or counts.ndim!=2 or counts.max(initial=0)>960:raise ValueError('Native SASA counts')
 n,m=counts.shape
 if not n or not np.array_equal(frames,np.arange(frames[0],frames[0]+n)) or len(times)!=n:raise ValueError('Native SASA frames')
 raw=_packed_counts(counts).tobytes();traw=np.asarray(times,dtype='<f8').tobytes();codec=Zstd(level=3);p=bytes(codec.encode(raw));t=bytes(codec.encode(traw))
 h=dict(schema=1,frames=n,atoms=m,start=int(frames[0]),counts_length=len(p),time_length=len(t),counts_raw_sha256=sha(raw),time_raw_sha256=sha(traw),payload_sha256=sha(p+t));header=json.dumps(h,sort_keys=True,separators=(',',':')).encode()
 return MAGIC+struct.pack('<Q',len(header))+header+p+t
