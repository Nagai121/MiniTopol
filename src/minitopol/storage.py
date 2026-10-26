"""Atomic publication helpers. Locks are shared by all cooperating processes."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from contextlib import contextmanager

import pyarrow.parquet as pq
from filelock import FileLock


_WINDOWS = os.name == "nt"


def _replace(source, destination):
    """Keep atomic publication; tolerate brief Windows handle contention."""
    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            # Access denied can also be permanent: bound retries to 0.63 s,
            # then propagate the original OS error. Never unlink the target.
            if not _WINDOWS or getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 6:
                raise
            time.sleep(0.01 * 2 ** attempt)


@contextmanager
def _json_lock(path):
    """Serialize short JSON reads/replacements on Windows, across processes.

    Workspace locks use its existing locks directory, including across user
    accounts. Standalone JSON uses the current account's temporary directory.
    """
    if not _WINDOWS:
        yield
        return
    resolved = Path(os.path.realpath(path))
    directory = next((parent / "locks" for parent in resolved.parents
                      if (parent / "integrated.duckdb").is_file() and (parent / "locks").is_dir()), None)
    if directory is None:
        directory = Path(tempfile.gettempdir()) / "minitopol-json-locks"
        directory.mkdir(exist_ok=True)
    key = digest(os.path.normcase(str(resolved)))
    with FileLock(str(directory / (key + ".lock"))):
        yield


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Identifiers must be 1–128 ASCII letters, digits, dots, underscores or hyphens")
    return value


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def fingerprint(path) -> dict:
    path = Path(path).resolve(strict=True)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            hasher.update(block)
    return {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size}


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical(data))
            stream.flush()
            os.fsync(stream.fileno())
        with _json_lock(path):
            _replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)



_BUILTIN_RESULTS = frozenset({'contacts','dynamic_correlation','gyration','msd','ramachandran','rmsd','rmsf','sasa','secondary_structure','time_correlation'})
_RESULT_COLUMNS = frozenset({'trajectory','representation','descriptor','definition_id'})

def _basic_result_schema(schema, *, metadata_dictionary=False):
    import pyarrow as pa
    for field in schema:
        kind=field.type
        if metadata_dictionary and field.name in _RESULT_COLUMNS and pa.types.is_dictionary(kind):
            if not pa.types.is_string(kind.value_type):return False
            kind=kind.value_type
        if field.metadata is not None or not field.nullable or not (pa.types.is_int64(kind) or pa.types.is_float64(kind) or pa.types.is_string(kind)):
            return False
    return True

def _parquet_options(table, *, _metadata_dictionary=False):
    # Standard Parquet encodings only; preserve schema, rows, nulls and values.
    # Restrict the policy to built-in result schemas; other callers keep defaults.
    import pyarrow as pa
    if b"minitopol" not in (table.schema.metadata or {}):
        return {"compression": "zstd"}
    import json
    try:
        info = json.loads(table.schema.metadata[b"minitopol"])
        descriptor = info["definition"]["name"]
    except (ValueError, TypeError, KeyError, UnicodeError):
        return {"compression": "zstd"}
    if not isinstance(descriptor, str):
        return {"compression": "zstd"}
    if any(not (pa.types.is_int64(field.type) or pa.types.is_float64(field.type)
                or pa.types.is_string(field.type)
                or (_metadata_dictionary and field.name in _RESULT_COLUMNS and pa.types.is_dictionary(field.type) and pa.types.is_string(field.type.value_type))) for field in table.schema):
        return {"compression": "zstd"}
    if descriptor not in {"contacts", "dynamic_correlation", "gyration", "msd", "ramachandran", "rmsd", "rmsf", "sasa", "secondary_structure", "time_correlation"}:
        return {"compression": "zstd"}
    dictionary = [field.name for field in table.schema if pa.types.is_string(field.type)
                  or (_metadata_dictionary and field.name in _RESULT_COLUMNS and pa.types.is_dictionary(field.type))]
    if descriptor == "sasa":
        dictionary += [name for name in ("value", "time_ps") if name in table.column_names]
    encoding = {field.name: ("DELTA_BINARY_PACKED" if pa.types.is_integer(field.type)
                            else "BYTE_STREAM_SPLIT" if pa.types.is_floating(field.type) else "PLAIN")
                for field in table.schema if field.name not in dictionary}
    return {"compression": "zstd", "compression_level": 3,
            "use_dictionary": dictionary, "column_encoding": encoding}

def _publish_parquet(path: Path, table, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    os.close(fd)
    try:
        write(table, name)
        if pq.read_metadata(name).num_rows != len(table):
            raise IOError("Parquet row count verification failed")
        with open(name, "rb+") as stream:
            os.fsync(stream.fileno())
        _replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

def atomic_parquet(path: Path, table) -> None:
    def write(value,name):pq.write_table(value,name,**_parquet_options(value))
    _publish_parquet(path,table,write)

def _write_compact_result(table, name):
    # store_schema=False also omits application metadata in Arrow 26. Add it
    # explicitly: the scientific metadata is mandatory, not a saving target.
    with pq.ParquetWriter(name,table.schema,store_schema=False,
                          **_parquet_options(table,_metadata_dictionary=True)) as writer:
        writer.add_key_value_metadata(table.schema.metadata)
        writer.write_table(table)

def _atomic_result_parquet(path: Path, table) -> None:
    metadata=table.schema.metadata or {}
    try:
        name=json.loads(metadata[b'minitopol'])['definition']['name']
    except (KeyError,TypeError,ValueError,UnicodeError):
        return atomic_parquet(path,table)
    if set(metadata)!={b'minitopol'} or not isinstance(name,str) or name not in _BUILTIN_RESULTS or not _basic_result_schema(table.schema,metadata_dictionary=True):
        return atomic_parquet(path,table)
    _publish_parquet(path,table,_write_compact_result)


def read_json(path: Path):
    with _json_lock(path):
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
