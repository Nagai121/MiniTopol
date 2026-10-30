"""Reuse n12-10 exact byte-owned unpack for SCT3 binary planes after inverse Gray."""
import numpy as np
from numba import njit
@njit(nogil=True,cache=True)
def _unpacked_byte_counts(packed,n,m):
 result=np.empty((m,n),dtype=np.uint16);total=n*m
 for column in range(packed.shape[1]):
  b0=packed[0,column]
  b1=packed[1,column]
  b2=packed[2,column]
  b3=packed[3,column]
  b4=packed[4,column]
  b5=packed[5,column]
  b6=packed[6,column]
  b7=packed[7,column]
  b8=packed[8,column]
  b9=packed[9,column]
  frame=(column*8)%n;atom=(column*8)//n
  for shift in range(8):
   if column*8+shift>=total:break
   value=0
   value|=((b0>>shift)&1)<<0
   value|=((b1>>shift)&1)<<1
   value|=((b2>>shift)&1)<<2
   value|=((b3>>shift)&1)<<3
   value|=((b4>>shift)&1)<<4
   value|=((b5>>shift)&1)<<5
   value|=((b6>>shift)&1)<<6
   value|=((b7>>shift)&1)<<7
   value|=((b8>>shift)&1)<<8
   value|=((b9>>shift)&1)<<9
   result[atom,frame]=value
   frame+=1
   if frame==n:frame=0;atom+=1
 return result.T
