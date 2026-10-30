"""Bounded canonical Rama columns shared by producer and pinned reader."""
import numpy as np
import pyarrow as pa

class RamaBatch:
    def __init__(self,residues,schema):
        self.residues=np.asarray(residues,dtype='i8');self.schema=schema
        self.frames=[];self.times=[];self.values=[];self.reasons=[]
    def append(self,frame,time,angles,why):
        self.frames.append(frame);self.times.append(time)
        self.values.append(np.column_stack((angles['phi'],angles['psi'])).ravel())
        self.reasons.extend(np.column_stack((why['phi'],why['psi'])).ravel().tolist())
    def table(self):
        width=len(self.residues)*2;count=len(self.frames)
        values=np.concatenate(self.values);valid=np.isfinite(values)
        values=np.where(valid,values,0.)
        return pa.Table.from_arrays([
            pa.array(np.repeat(np.asarray(self.frames,dtype='i8'),width)),
            pa.array(np.repeat(np.asarray(self.times,dtype='f8'),width)),
            pa.array(np.tile(np.repeat(self.residues,2),count),type=pa.int64()),
            pa.array(['phi','psi']*(len(self.residues)*count),type=pa.string()),
            pa.array(values,mask=~valid,type=pa.float64()),
            pa.array(self.reasons,type=pa.string())],schema=self.schema)
