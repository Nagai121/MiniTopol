"""Progress v4 journal; completed datasets retain schema 3.

New readers accept legacy progress v3 without migration. Old readers cannot
resume v4 journals.
"""
import json, os
from ..storage import atomic_json, canonical, digest


def start(path, state):
    atomic_json(path, {"schema": 4, "info": state["info"], "layout": state["layout"]})


def append(path, index, part):
    payload={"index":index,"part":part}
    record={**payload,"sha256":digest(payload)}
    with path.open('ab') as stream:
        stream.write(('\n'+canonical(record)+'\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())


def load(path):
    with path.open('rb+') as stream:
        first=stream.readline()
        try:
            header=json.loads(first)
        except json.JSONDecodeError:
            # Legacy JSON is not required to be minified by its reader.
            stream.seek(0)
            header=json.load(stream)
        if header.get('schema') != 4:
            return header, False
        state={"schema":3,"info":header["info"],"layout":header["layout"],"parts":[]}
        good=stream.tell()
        while line:=stream.readline():
            if not line.endswith(b'\n'):
                # A torn last append is uncommitted. Never ignore a bad full line.
                stream.truncate(good)
                stream.flush();os.fsync(stream.fileno())
                break
            if line.strip():
                record=json.loads(line)
                payload={"index":record["index"],"part":record["part"]}
                if record['sha256'] != digest(payload) or record['index'] != len(state['parts']):
                    raise IOError('Progress journal checksum/order mismatch')
                state['parts'].append(record['part'])
            good=stream.tell()
        return state, True
