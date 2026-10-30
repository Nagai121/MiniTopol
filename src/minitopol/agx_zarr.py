"""Registered AGX1 Zarr bytes codec; logical float32 arrays are unchanged."""
import numpy as np
import numcodecs
from numcodecs.abc import Codec
from numcodecs.blosc import set_nthreads
from zarr.codecs.numcodecs._codecs import _NumcodecsBytesBytesCodec
from zarr.registry import register_codec
from .position_grid import encode,decode,MAX_RAW

class AGXNumcodec(Codec):
    codec_id='minitopol_agx1'
    def __init__(self,atoms):
        self.atoms=int(atoms)
        if self.atoms<1:raise ValueError('AGX atoms')
    def encode(self,buf):
        raw=memoryview(buf).tobytes()
        if not raw or len(raw)>MAX_RAW or len(raw)%(12*self.atoms):raise ValueError('AGX chunk shape/budget')
        set_nthreads(2)
        x=np.frombuffer(raw,dtype='<f4').reshape(-1,self.atoms,3)
        return encode(x)[0]
    def decode(self,buf,out=None):
        x=decode(bytes(buf))
        if x.shape[1]!=self.atoms:raise IOError('AGX atom-axis mismatch')
        raw=x.tobytes()
        if out is None:return raw
        target=memoryview(out).cast('B')
        if target.nbytes!=len(raw):raise ValueError('AGX output buffer shape')
        target[:]=raw;return out

numcodecs.register_codec(AGXNumcodec)
class AGXCodec(_NumcodecsBytesBytesCodec,codec_name='minitopol_agx1'):
    pass
register_codec('numcodecs.minitopol_agx1',AGXCodec)

# New writes use AGX2; retain independent AGX1 registration for old A readers.
AGX1Codec=AGXCodec
class AGX2Numcodec(AGXNumcodec):
    codec_id='minitopol_agx2'
    def encode(self,buf):
        from .position_xor import encode as xor_encode
        raw=memoryview(buf).tobytes()
        if not raw or len(raw)>MAX_RAW or len(raw)%(12*self.atoms):raise ValueError('AGX2 chunk shape/budget')
        set_nthreads(2)
        x=np.frombuffer(raw,dtype='<f4').reshape(-1,self.atoms,3)
        return xor_encode(x)[0]
    def decode(self,buf,out=None):
        from .position_xor import decode as xor_decode
        x=xor_decode(bytes(buf))
        if x.shape[1]!=self.atoms:raise IOError('AGX2 atom-axis mismatch')
        raw=x.tobytes()
        if out is None:return raw
        target=memoryview(out).cast('B')
        if target.nbytes!=len(raw):raise ValueError('AGX2 output buffer shape')
        target[:]=raw;return out
numcodecs.register_codec(AGX2Numcodec)
class AGX2Codec(_NumcodecsBytesBytesCodec,codec_name='minitopol_agx2'):
    pass
register_codec('numcodecs.minitopol_agx2',AGX2Codec)
AGXCodec=AGX2Codec

# New AGX3 raw-bit modular predictor; keep independent AGX1/2 readers.
class AGX3Numcodec(AGXNumcodec):
    codec_id='minitopol_agx3'
    def encode(self,buf):
        from .position_moddelta import encode as xor_encode
        raw=memoryview(buf).tobytes()
        if not raw or len(raw)>MAX_RAW or len(raw)%(12*self.atoms):raise ValueError('AGX3 chunk shape/budget')
        set_nthreads(2)
        x=np.frombuffer(raw,dtype='<f4').reshape(-1,self.atoms,3)
        return xor_encode(x)[0]
    def decode(self,buf,out=None):
        from .position_moddelta import decode as xor_decode
        x=xor_decode(bytes(buf))
        if x.shape[1]!=self.atoms:raise IOError('AGX3 atom-axis mismatch')
        raw=x.tobytes()
        if out is None:return raw
        target=memoryview(out).cast('B')
        if target.nbytes!=len(raw):raise ValueError('AGX3 output buffer shape')
        target[:]=raw;return out
numcodecs.register_codec(AGX3Numcodec)
class AGX3Codec(_NumcodecsBytesBytesCodec,codec_name='minitopol_agx3'):
    pass
register_codec('numcodecs.minitopol_agx3',AGX3Codec)
AGXCodec=AGX3Codec
