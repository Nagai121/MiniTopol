import numpy as np
import pytest
from MDAnalysis.lib.distances import distance_array
from minitopol.descriptors import periodic_contacts as contacts
from minitopol.coordinates import box_dimensions

def reference(xyz,box,residue,allowed):
    d=distance_array(xyz,xyz,box=box_dimensions(box));n=len(allowed);minimum=np.full((n,n),np.inf)
    for r in range(n):
        for s in range(r+1,n):
            if allowed[r,s]:minimum[r,s]=d[np.ix_(residue==r,residue==s)].min()
    rr,ss=np.nonzero(np.isfinite(minimum)&(minimum<=4.5))
    return rr,ss,minimum[rr,ss]

@pytest.mark.parametrize('seed',range(8))
def test_batched_c_distances_are_exact(seed):
    xyz=np.random.default_rng(seed).uniform(-100,100,(31,3)).astype('f4')
    xyz[:10,0]=np.array([-1e-40,-0.,0.,4.5,np.nextafter(np.float32(4.5),np.float32(np.inf)),9.9,10.,10.1,-10.1,1e5],dtype='f4')
    xyz[:10,1:]=0;box=np.diag([10.,11.,12.]).astype('f4');residue=np.arange(len(xyz))%5
    allowed=np.triu(np.ones((5,5),bool),1);allowed[0,1]=False
    expected=reference(xyz,box,residue,allowed);actual=contacts._frame_contacts(xyz,box,residue,allowed,128*1024**2)
    for x,y in zip(actual[:2],expected[:2]):assert np.array_equal(x,y)
    assert np.array_equal(actual[2].view('u8'),expected[2].view('u8'))

def test_frame_parallelism_keeps_exact_frame_order():
    xyz=np.random.default_rng(22).normal(size=(8,40,3)).astype('f4');boxes=np.repeat(np.diag([20.,21.,22.])[None],8,axis=0).astype('f4');residue=np.arange(40)%6;allowed=np.triu(np.ones((6,6),bool),1)
    outputs=list(contacts._ordered_frames(xyz,boxes,residue,allowed,128*1024**2));assert [i for i,_ in outputs]==list(range(8))
    for (i,result) in outputs:
        expected=reference(xyz[i],boxes[i],residue,allowed)
        for a,b in zip(result[:2],expected[:2]):assert np.array_equal(a,b)
        assert np.array_equal(result[2].view('u8'),expected[2].view('u8'))


def test_triclinic_candidates_preserve_mda_all_pair_reference():
    xyz=np.random.default_rng(21).uniform(-20,20,(31,3)).astype('f4')
    box=np.array([[10.,0.,0.],[2.,11.,0.],[1.,2.,12.]],dtype='f4')
    residue=np.arange(31)%5;allowed=np.triu(np.ones((5,5),bool),1)
    expected=reference(xyz,box,residue,allowed)
    actual=contacts._frame_contacts(xyz,box,residue,allowed,128*1024**2)
    for left,right in zip(actual,expected):
        np.testing.assert_array_equal(left.view('u8') if left.dtype==np.float64 else left,
                                      right.view('u8') if right.dtype==np.float64 else right)


@pytest.mark.parametrize('failure',[False,True])
def test_ordered_window_joins_owned_threads_on_close_or_error(monkeypatch,failure):
    from concurrent.futures import ThreadPoolExecutor
    pools=[]
    def pool(**options):
        executor=ThreadPoolExecutor(**options);pools.append(executor);return executor
    def frame(xyz,*args):
        if failure:raise RuntimeError('injected contact failure')
        return (np.array([0]),np.array([1]),np.array([1.]))
    monkeypatch.setattr(contacts,'ThreadPoolExecutor',pool)
    monkeypatch.setattr(contacts,'_frame_contacts',frame)
    frames=contacts._ordered_frames(np.zeros((17,5,3),dtype='f4'),np.zeros((17,3,3)),np.arange(5),np.zeros((5,5),bool),128*1024**2)
    if failure:
        with pytest.raises(RuntimeError,match='injected contact failure'):next(frames)
    else:
        assert next(frames)[0]==0
        frames.close()
    assert pools and all(not thread.is_alive() for executor in pools for thread in executor._threads)

@pytest.mark.parametrize('case',['nan','large','budget'])
def test_unsupported_inputs_use_original(case):
    xyz=np.zeros((5,3),dtype='f4');box=np.diag([10.,10.,10.]).astype('f4');budget=128*1024**2
    if case=='triclinic':box[1,0]=2
    if case=='nan':xyz[0,0]=np.nan
    if case=='large':xyz[0,0]=1e7
    if case=='budget':budget=1
    with pytest.raises(contacts.UseOriginal):contacts._frame_contacts(xyz,box,np.arange(5),np.triu(np.ones((5,5),bool),1),budget)
