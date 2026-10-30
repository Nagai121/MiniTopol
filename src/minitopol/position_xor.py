"""AGX2 experimental temporal float-bit XOR, exact and block bounded."""
import hashlib,json,struct
import numpy as np
from numcodecs import Blosc
from numcodecs.blosc import _cbuffer_sizes
MAGIC=b'AGX2\x00\x00\x00\x00'
MAX_RAW=8*1024**2
def sha(value):return hashlib.sha256(value).hexdigest()
def encode(x):
    if x.dtype!=np.dtype('f4') or x.ndim!=3 or x.shape[2]!=3 or not np.little_endian:
        raise ValueError('AGX2 requires native little-endian float32 positions')
    if not x.size or x.nbytes>MAX_RAW:raise ValueError('AGX2 block size bound')
    x=np.ascontiguousarray(x);bits=x.view('u4');delta=np.empty_like(bits)
    delta[0]=bits[0];np.bitwise_xor(bits[1:],bits[:-1],out=delta[1:])
    packed=np.ascontiguousarray(delta.transpose(1,2,0))
    payload=bytes(Blosc(cname='zstd',clevel=3,shuffle=Blosc.BITSHUFFLE).encode(packed))
    header=dict(schema=1,shape=list(x.shape),payload_length=len(payload),raw_position_sha256=sha(x.tobytes()),payload_sha256=sha(payload),predictor='temporal_u32_xor',order='atom,axis,time',codec='Blosc-Zstd3-bitshuffle')
    raw=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
    return MAGIC+struct.pack('<Q',len(raw))+raw+payload,header
def decode(raw):
    if len(raw)<16 or raw[:8]!=MAGIC:raise IOError('AGX2 magic/truncation')
    size=struct.unpack('<Q',raw[8:16])[0]
    if not 0<size<=4096 or 16+size>len(raw):raise IOError('AGX2 header bounds')
    try:header=json.loads(raw[16:16+size])
    except (ValueError,UnicodeError) as error:raise IOError('AGX2 invalid header') from error
    shape=header.get('shape')
    if not isinstance(shape,list) or len(shape)!=3 or any(type(v) is not int for v in shape):raise IOError('AGX2 shape type')
    n,m,axis=shape;expected=n*m*3*4
    if axis!=3 or n<1 or m<1 or expected>MAX_RAW:raise IOError('AGX2 shape bounds')
    if header.get('schema')!=1 or header.get('predictor')!='temporal_u32_xor' or header.get('order')!='atom,axis,time' or header.get('codec')!='Blosc-Zstd3-bitshuffle':raise IOError('AGX2 predictor contract')
    payload=raw[16+size:]
    if len(payload)!=header['payload_length'] or len(payload)>MAX_RAW+64 or sha(payload)!=header['payload_sha256']:raise IOError('AGX2 payload checksum/size')
    if len(payload)<16:raise IOError('AGX2 truncated Blosc header')
    try:nbytes,cbytes,_=_cbuffer_sizes(payload)
    except Exception as error:raise IOError('AGX2 invalid Blosc header') from error
    if nbytes!=expected or cbytes!=len(payload):raise IOError('AGX2 compressed shape bounds')
    decoded=bytes(Blosc().decode(payload))
    if len(decoded)!=expected:raise IOError('AGX2 decoded size')
    delta=np.frombuffer(decoded,dtype='<u4').reshape(m,3,n).transpose(2,0,1)
    result=np.ascontiguousarray(np.bitwise_xor.accumulate(delta,axis=0,dtype='u4')).view('f4')
    if sha(result.tobytes())!=header['raw_position_sha256']:raise IOError('AGX2 source bit checksum')
    return result
