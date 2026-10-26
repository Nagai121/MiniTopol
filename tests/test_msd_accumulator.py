from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pytest
import zarr
from scipy.signal import fftconvolve
from minitopol.descriptors import msd
from minitopol.descriptors.correlation import _MemoryArray
from minitopol.storage import read_json


def test_missing_optional_codec_retains_default_lossless_storage(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pcodec", None)
    assert msd._slot_encoding() == {}

class Writer:
    def __init__(self):self.tables=[]
    def has(self,key):return False
    def append(self,key,table):self.tables.append(table)
    def values(self):return pa.concat_tables(self.tables)['value'].to_numpy()

def context(path,data,budget):
    path.mkdir(parents=True,exist_ok=True);mean=data.mean(axis=0)
    return SimpleNamespace(n=len(data),m=data.shape[1],budget=budget,fft_block=len(data),chunk=64,
                           cache_path=path,timestep=lambda:1.,statistics=lambda:(mean,np.zeros_like(mean)))

@pytest.fixture
def data_and_correlation(monkeypatch):
    data=np.random.default_rng(71).normal(size=(257,7,3)).cumsum(axis=0)
    def correlation(ctx,point,mean,directory):
        a=data[:,point]-mean;prefix=np.empty(len(a)+1).view(_MemoryArray);prefix[0]=0;prefix[1:]=np.cumsum(np.einsum('ij,ij->i',a,a))
        dots=(fftconvolve(a,a[::-1],mode='full',axes=0).sum(axis=1)[len(a)-1:]+0).view(_MemoryArray)
        return dots,prefix
    monkeypatch.setattr(msd,'atom_correlation',correlation)
    return data

def test_accumulator_read_reduction_keeps_bits_and_writes(tmp_path,data_and_correlation,monkeypatch):
    data=data_and_correlation;reads=[];writes=[];get=zarr.Array.__getitem__;put=zarr.Array.__setitem__
    def record_get(self,key):reads.append(1);return get(self,key)
    def record_put(self,key,value):writes.append(1);return put(self,key,value)
    monkeypatch.setattr(zarr.Array,'__getitem__',record_get);monkeypatch.setattr(zarr.Array,'__setitem__',record_put)
    disk=Writer();msd.calculate(context(tmp_path/'disk',data,len(data)*16*64-1),disk);disk_reads=len(reads);disk_writes=len(writes);reads.clear();writes.clear()
    cached=Writer();msd.calculate(context(tmp_path/'cached',data,128*1024**2),cached)
    assert np.array_equal(cached.values().view('u8'),disk.values().view('u8'))
    assert len(reads)<disk_reads and len(writes)==disk_writes==data.shape[1]

@pytest.mark.parametrize('fault_atom',[2,3])
def test_both_committed_slots_resume_exactly(tmp_path,data_and_correlation,monkeypatch,fault_atom):
    data=data_and_correlation;reference=Writer();msd.calculate(context(tmp_path/'reference',data,len(data)*16*64-1),reference)
    ctx=context(tmp_path/'resume',data,128*1024**2);original=msd.atomic_json
    def fail(path,state):
        if state.get('atoms_done')==fault_atom:raise RuntimeError('injected before accumulator commit')
        return original(path,state)
    with monkeypatch.context() as patch:
        patch.setattr(msd,'atomic_json',fail)
        with pytest.raises(RuntimeError,match='before accumulator commit'):msd.calculate(ctx,Writer())
    committed=read_json(ctx.cache_path/'msd'/'state');assert committed['atoms_done']==fault_atom-1 and committed['slot']==(fault_atom-1)%2
    resumed=Writer();msd.calculate(ctx,resumed)
    assert np.array_equal(resumed.values().view('u8'),reference.values().view('u8'))
    assert read_json(ctx.cache_path/'msd'/'state')['atoms_done']==data.shape[1]
