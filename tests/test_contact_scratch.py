import hashlib,io
import numpy as np
import pytest
from minitopol.descriptors import contact_scratch as scratch
from minitopol.descriptors.periodic_contacts import DTYPE

@pytest.mark.parametrize('count',[0,1,10003])
def test_compressed_scratch_preserves_every_byte_and_closes_handles(tmp_path,count):
    values=np.empty(count,dtype=DTYPE);values['frame']=np.arange(count)//17;values['r']=np.arange(count)%11;values['s']=20+np.arange(count)%13;values['d']=np.random.default_rng(8).uniform(0,4.5,count)
    raw=values.tobytes();path=tmp_path/'scratch';writer=scratch.CompressedScratch(path)
    for start in range(0,len(raw),37):writer.write(raw[start:start+37])
    writer.close();writer.close();scratch.verify(path,writer,len(raw),hashlib.sha256(raw).digest(),DTYPE.itemsize)
    decoded=[]
    with scratch.open_reader(path) as stream:
        while True:
            block=scratch.read_records(stream,31,DTYPE)
            if not len(block):break
            decoded.append(block.tobytes())
    assert b''.join(decoded)==raw
    path.unlink();assert not path.exists()

def test_compressed_byte_corruption_is_detected_before_decode(tmp_path):
    path=tmp_path/'scratch';raw=bytes(range(256))*32;writer=scratch.CompressedScratch(path);writer.write(raw);writer.close()
    encoded=bytearray(path.read_bytes());encoded[len(encoded)//2]^=1;path.write_bytes(encoded)
    with pytest.raises(IOError,match='checksum mismatch'):scratch.verify(path,writer,len(raw),hashlib.sha256(raw).digest(),32)

class ShortReads(io.BytesIO):
    def read(self,n=-1):return super().read(min(n,7) if n>=0 else 7)

def test_short_reads_are_assembled_and_torn_rows_rejected():
    raw=bytes(range(32))*3;stream=ShortReads(raw)
    assert scratch.read_records(stream,3,DTYPE).tobytes()==raw
    assert len(scratch.read_records(stream,3,DTYPE))==0
    with pytest.raises(IOError,match='length mismatch'):scratch.read_records(ShortReads(raw[:-1]),3,DTYPE)
