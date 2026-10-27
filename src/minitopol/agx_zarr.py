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
