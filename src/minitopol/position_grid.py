"""AGX1: integer-grid predictor plus mandatory lossless float-bit residual.

Prediction never replaces source precision. XOR residuals preserve every bit,
including off-grid values, signed zeros and non-finite raw-mode values.
"""
import hashlib,json,os,struct
from pathlib import Path
import numpy as np
from numcodecs import Blosc
from numcodecs.blosc import _cbuffer_sizes
MAGIC=b'AGX1\x00\x00\x00\x00';MAX_RAW=8*1024**2
def sha(raw):return hashlib.sha256(raw).hexdigest()
def compress(array):return bytes(Blosc(cname='zstd',clevel=3,shuffle=Blosc.BITSHUFFLE).encode(np.ascontiguousarray(array)))
def decompress(payload,expected):
 try:nbytes,cbytes,_=_cbuffer_sizes(payload)
 except Exception as error:raise IOError('Invalid Blosc header') from error
 if nbytes!=expected or cbytes!=len(payload) or expected>MAX_RAW:raise IOError('AGX compressed-size bounds mismatch')
 raw=bytes(Blosc().decode(payload))
 if len(raw)!=expected:raise IOError('AGX decoded size mismatch')
 return raw
def encode(x):
 if x.dtype!=np.dtype('f4') or x.ndim!=3 or x.shape[2]!=3 or not np.little_endian:raise ValueError('AGX1 requires native little-endian float32 position blocks')
 x=np.ascontiguousarray(x)
 if not x.size or x.nbytes>MAX_RAW:raise ValueError('AGX block exceeds bounded working size')
 n,m,_=x.shape;mode='bits';nonzero=x.size
 if np.isfinite(x).all() and float(np.abs(x).max(initial=0))*100<(2**30-1):
  mode='grid';k=np.rint(x.astype('f8')*100).astype('i4')
  pred=k.astype('f4')*np.float32(.001)*np.float32(10)
  residual=x.view('u4')^pred.view('u4');nonzero=int(np.count_nonzero(residual))
  delta=np.empty_like(k);delta[0]=k[0];np.subtract(k[1:],k[:-1],out=delta[1:])
  zigzag=(delta.view('u4')<<np.uint32(1))^((delta>>np.int32(31)).view('u4'))
  first=compress(zigzag.transpose(1,2,0))
  second=compress(residual.transpose(1,2,0)) if nonzero else b''
 else:first=compress(x.view('u4').transpose(1,2,0));second=b''
 h=dict(schema=1,mode=mode,shape=[n,m,3],first_length=len(first),residual_length=len(second),raw_position_sha256=sha(x.tobytes()),payload_sha256=sha(first+second),residual_nonzero=nonzero,predictor='rn_float32(k)*rn_float32(0.001)*rn_float32(10)',order='atom,axis,time',codec='Blosc-Zstd3-bitshuffle')
 header=json.dumps(h,sort_keys=True,separators=(',',':')).encode()
 return MAGIC+struct.pack('<Q',len(header))+header+first+second,h
def decode(raw):
 if raw[:8]!=MAGIC or len(raw)<16:raise IOError('AGX magic/truncation')
 length=struct.unpack('<Q',raw[8:16])[0]
 if not 0<length<=4096 or 16+length>len(raw):raise IOError('AGX header bounds')
 try:h=json.loads(raw[16:16+length])
 except (ValueError,UnicodeError) as error:raise IOError('AGX invalid header') from error
 n,m,axis=h['shape'];expected=n*m*3*4
 if axis!=3 or n<1 or m<1 or expected>MAX_RAW:raise IOError('AGX shape bounds')
 payload=raw[16+length:]
 if len(payload)!=h['first_length']+h['residual_length'] or sha(payload)!=h['payload_sha256']:raise IOError('AGX payload checksum/size')
 data=decompress(payload[:h['first_length']],expected);u=np.frombuffer(data,dtype='<u4').reshape(m,3,n).transpose(2,0,1)
 if h['mode']=='grid':
  delta=(u>>np.uint32(1)).astype('i8')^-((u&np.uint32(1)).astype('i8'));k=np.cumsum(delta,axis=0,dtype='i8')
  if k.min(initial=0)<np.iinfo('i4').min or k.max(initial=0)>np.iinfo('i4').max:raise IOError('AGX reconstructed integer overflow')
  pred=np.ascontiguousarray(k.astype('f4')*np.float32(.001)*np.float32(10))
  if h['residual_length']:
   residual=np.frombuffer(decompress(payload[h['first_length']:],expected),dtype='<u4').reshape(m,3,n).transpose(2,0,1)
   result=np.ascontiguousarray(pred.view('u4')^residual).view('f4')
  else:result=pred
 elif h['mode']=='bits':result=np.ascontiguousarray(u).view('f4')
 else:raise IOError('Unknown AGX mode')
 if sha(result.tobytes())!=h['raw_position_sha256']:raise IOError('AGX source coordinate-bit checksum mismatch')
 return result
def publish(path,raw):
 path=Path(path);tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
