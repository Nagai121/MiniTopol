"""Lossless physical catalog rebuild; caller must be outside catalog sessions.

Locks the same catalog lock as all normal accesses. The old database is
attached READ_ONLY; only a fresh same-directory file is written. Every user
table's DDL, ordered Arrow IPC bytes and explicit index DDL are compared
before fsync and atomic replacement. No trajectory/descriptor files are read
or changed. This is an explicit maintenance operation, not automatic policy.
"""
import hashlib,os,uuid
from pathlib import Path
from filelock import FileLock

def _quote(value):return '"'+value.replace('"','""')+'"'
def _literal(value):return "'"+str(value).replace("'","''")+"'"
def _signature(connection,catalog):
 import pyarrow as pa
 rows=connection.execute("SELECT schema_name,table_name,sql FROM duckdb_tables() WHERE database_name=? ORDER BY schema_name,table_name",[catalog]).fetchall()
 signatures=[];logical_bytes=0
 for schema,name,ddl in rows:
  qualified='.'.join(map(_quote,(catalog,schema,name)))
  description=connection.execute('DESCRIBE '+qualified).fetchall()
  if any(row[1] not in ('VARCHAR','INTEGER','BIGINT','BOOLEAN','DOUBLE') for row in description):raise ValueError('Unsupported catalog type; retain original catalog')
  expressions=['count(*)*'+str(len(description)*16)]+['coalesce(sum(octet_length(encode('+_quote(row[0])+'))),0)' for row in description if row[1]=='VARCHAR']
  logical_bytes+=sum(connection.execute('SELECT '+','.join(expressions)+' FROM '+qualified).fetchone())
  if logical_bytes>8*1024**2:raise ValueError('Catalog logical budget exceeded; retain original catalog')
  table=connection.execute('SELECT * FROM '+qualified+' ORDER BY ALL').to_arrow_table().combine_chunks()
  sink=pa.BufferOutputStream()
  with pa.ipc.new_stream(sink,table.schema) as writer:writer.write_table(table,max_chunksize=65536)
  signatures.append(dict(schema=schema,name=name,ddl=ddl,rows=len(table),arrow_sha256=hashlib.sha256(sink.getvalue()).hexdigest()))
 indexes=connection.execute("SELECT schema_name,index_name,sql FROM duckdb_indexes() WHERE database_name=? ORDER BY schema_name,index_name",[catalog]).fetchall()
 return dict(tables=signatures,indexes=indexes)

def compact_catalog(path):
 import duckdb
 from .storage import _replace
 root=Path(path).resolve();target=root/'integrated.duckdb'
 if not target.is_file():raise FileNotFoundError(target)
 if target.stat().st_size>16*1024**2:raise ValueError('Catalog physical budget exceeded; retain original catalog')
 if target.with_suffix('.duckdb.wal').exists():raise ValueError('Close/checkpoint catalog WAL before compaction')
 locks=root/'locks';locks.mkdir(exist_ok=True)
 with FileLock(str(locks/'catalog.lock')):
  temporary=root/('.catalog-compact-'+uuid.uuid4().hex+'.duckdb')
  before=target.stat().st_size
  try:
   with duckdb.connect() as con:
    con.execute('ATTACH '+_literal(target)+' AS original (READ_ONLY)')
    prior=_signature(con,'original')
    con.execute('ATTACH '+_literal(temporary)+' AS rebuilt')
    con.execute('COPY FROM DATABASE original TO rebuilt')
    after=_signature(con,'rebuilt')
    if prior!=after:raise IOError('Catalog DDL/typedcell/index copy mismatch')
    con.execute('DETACH rebuilt');con.execute('DETACH original')
   # Reopen the actual closed artifact before publishing, not just its buffer.
   with duckdb.connect() as con:
    con.execute('ATTACH '+_literal(temporary)+' AS reopened (READ_ONLY)')
    if _signature(con,'reopened')!=prior:raise IOError('Closed catalog verification mismatch')
   with temporary.open('rb+') as f:os.fsync(f.fileno())
   _replace(temporary,target)
   return dict(before_bytes=before,after_bytes=target.stat().st_size,signature=prior)
  finally:
   if temporary.exists():temporary.unlink()
