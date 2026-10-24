"""Regression checks for the four reported storage/lifecycle defects."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from minitopol import Registry, Workspace, convert
from minitopol.descriptors import DescriptorDefinition, ResultStore
from minitopol.descriptors.calculators import calculate_frames


def imported(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "ws")
    tid = convert(workspace, *trajectory[:2], solvent_resnames=["SOL"], chunk_frames=4)
    return workspace, tid


def test_missing_published_file_cannot_be_reported_complete(tmp_path, trajectory):
    workspace, tid = imported(tmp_path, trajectory)
    definition = DescriptorDefinition("sample", "1", "A", "regression only", "dimensionless")

    def calc(block):
        return pa.table({"frame": list(range(block.start, block.stop)),
                         "value": [1.] * (block.stop - block.start)})

    calculate_frames(workspace, tid, definition, calc)
    next((workspace.path / "descriptors").rglob("*.parquet")).unlink()
    with pytest.raises(IOError, match="file is missing"):
        calculate_frames(workspace, tid, definition, calc)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("tampered", [False, True])
def test_recovery_checks_values_not_only_row_count(tmp_path, trajectory, monkeypatch, legacy, tampered):
    workspace, tid = imported(tmp_path, trajectory)
    definition = DescriptorDefinition("sample", "1", "A", "regression only", "dimensionless")
    store = ResultStore(workspace)
    monkeypatch.setattr(store, "_register", lambda *args: None)
    store.publish(tid, definition, 0, 2, pa.table({"frame": [0, 1], "value": [1., 2.]}))
    path = next((workspace.path / "descriptors").rglob("*.parquet"))
    if legacy:
        marker = path.with_suffix(".json")
        completed = json.loads(marker.read_text())
        completed["schema"] = 1
        del completed["file_sha256"]
        marker.write_text(json.dumps(completed))
    if tampered:
        table = pq.ParquetFile(path).read()
        table = table.set_column(table.schema.get_field_index("value"), "value", pa.array([900., 901.]))
        pq.write_table(table, path)
        with pytest.raises(IOError, match="checksum mismatch"):
            ResultStore(workspace).recover()
        with pytest.raises(LookupError):
            workspace.query_results()
    else:
        assert ResultStore(workspace).recover() == 1
        assert workspace.query_results()["value"].to_pylist() == [1., 2.]


@pytest.mark.parametrize("legacy", [False, True])
def test_optional_descriptor_parameters_normalized(tmp_path, trajectory, legacy):
    workspace, tid = imported(tmp_path, trajectory)
    definition = DescriptorDefinition("sample", "1", "C1", "regression only", "dimensionless")
    abbreviated = definition.as_dict()
    del abbreviated["parameters"]
    registry = Registry(workspace)
    kwargs = dict(trajectory=tid, version="1",
                  groups=[{"name": "bead", "atoms": [0, 1], "weights": [.5, .5]}],
                  descriptor_definitions={"sample": abbreviated})
    registry.register("C1", **kwargs)
    if legacy:
        with workspace.catalog() as con:
            value = json.loads(con.execute("SELECT definition FROM representations WHERE id='C1'").fetchone()[0])
            del value["descriptor_definitions"]["sample"]["parameters"]
            con.execute("UPDATE representations SET definition=? WHERE id='C1'", [json.dumps(value)])
    registry.register("C1", **dict(kwargs, descriptor_definitions={"sample": definition.as_dict()}))
    ResultStore(workspace).publish(tid, definition, 0, 1, pa.table({"frame": [0], "value": [1.]}))
    assert workspace.query_results(representation="C1").num_rows == 1


def test_default_workspace_does_not_depend_on_cwd(tmp_path, monkeypatch):
    import minitopol.workspace as module
    project = tmp_path / "MiniTopol"
    package = project / "src" / "minitopol"
    package.mkdir(parents=True)
    (project / "pyproject.toml").touch()
    monkeypatch.setattr(module, "__file__", str(package / "workspace.py"))
    monkeypatch.delenv("MINITOPOL_WORKSPACE", raising=False)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    first = Workspace()
    assert first.path == project / "BaseTraj_Workspace"
    sentinel = first.path / "keep.txt"
    sentinel.write_text("existing user data")
    monkeypatch.chdir(project)
    assert Workspace().path == first.path
    assert sentinel.read_text() == "existing user data"
    external = tmp_path / "external"
    monkeypatch.setenv("MINITOPOL_WORKSPACE", str(external))
    assert Workspace().path == external
    assert Workspace(tmp_path / "explicit").path == tmp_path / "explicit"



def test_direct_result_publication_rejects_nonfinite_numbers(tmp_path, trajectory):
    workspace, tid = imported(tmp_path, trajectory)
    definition = DescriptorDefinition("example", "1", "A", "test", "dimensionless")
    with pytest.raises(ValueError, match="Non-finite"):
        ResultStore(workspace).publish(
            tid, definition, 0, 1,
            pa.table({"frame": [0], "value": [float("nan")]}))
    assert not list((workspace.path / "descriptors").rglob("*.parquet"))
