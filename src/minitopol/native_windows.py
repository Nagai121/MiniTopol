"""One verified native unpack per physical part, bounded scientific expansion."""
import hashlib,numpy as np
from .direct_counts import extract
from .sasa_counts import context,expanded_table
from .descriptors.count_store import native_hash

def tables(raw,meta,part):
 if hashlib.sha256(raw).hexdigest()!=part['sha256']:raise IOError('Encoded part identity')
 counts,frames,times=extract(raw)
 width=len(meta['residue_index'])+len(set(meta['residue_index']))+1
 if counts.shape[1]!=len(meta['residue_index']) or len(frames)*width!=part['rows'] or native_hash(counts,frames,times,meta)!=part['payload_hash']:raise IOError('Native science identity')
 fac=np.frombuffer(bytes.fromhex(meta['factors_f8_hex']),dtype='<f8');ctx=context(meta)
 window=max(1,min(256,(64*1024**2)//max(1,width*256)))
 for first in range(0,len(frames),window):
  last=min(first+window,len(frames));values=counts[first:last].astype('f8')/960*fac
  yield expanded_table(ctx,frames[first:last],times[first:last],values).replace_schema_metadata({b'minitopol':meta['scientific_metadata'].encode()})
