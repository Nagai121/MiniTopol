import json
from pathlib import Path

import pytest

from minitopol.batch import _validate_jobs, run_pipelines
from minitopol.cli import main
from minitopol import Registry, Workspace
from minitopol.storage import fingerprint


def job(tmp_path, trajectory, name="job"):
    gro, xtc, _ = trajectory
    return {"workspace": str(tmp_path / name), "gro": str(gro), "xtc": str(xtc),
            "solvent_resnames": ["SOL"], "trajectory_id": name,
            "analysis": {"coordinate_mode": "nonperiodic", "descriptors": ["sasa"], "memory_mb": 128}}


def test_independent_processes_and_completed_rerun_preserve_parts(tmp_path, trajectory):
    pytest.importorskip("psutil")
    jobs = [job(tmp_path, trajectory, "one"), job(tmp_path, trajectory, "two")]
    result = run_pipelines(jobs, workers=2, native_threads=1, gpu_workers=0)
    assert result["status"] == "complete"
    assert all(len(r["analysis"]["results"]) == 1 for r in result["results"])
    ws = Workspace(jobs[0]["workspace"])
    assert Registry(ws).manifest("one")["frames"] == 9
    parts = [p for p in (ws.path / "descriptors").rglob("*") if p.is_file() and p.suffix in (".parquet", ".sct", ".vram")];assert parts
    before = {p: (fingerprint(p)["sha256"], p.stat().st_mtime_ns) for p in parts}
    again = run_pipelines(jobs, workers=2, native_threads=1, gpu_workers=0)
    assert again["status"] == "complete"
    assert {p: (fingerprint(p)["sha256"], p.stat().st_mtime_ns) for p in parts} == before


def test_validation_rejects_shared_workspace_and_missing_science_before_writes(tmp_path, trajectory):
    value = job(tmp_path, trajectory)
    with pytest.raises(ValueError, match="distinct"):
        _validate_jobs([value, value])
    assert not Path(value["workspace"]).exists()
    del value["analysis"]["coordinate_mode"]
    with pytest.raises(ValueError, match="coordinate_mode"):
        _validate_jobs([value])


def test_failed_job_does_not_start_pending_job(tmp_path, trajectory):
    pytest.importorskip("psutil")
    bad, pending = job(tmp_path, trajectory, "bad"), job(tmp_path, trajectory, "pending")
    bad["analysis"]["coordinate_mode"] = "invalid"
    result = run_pipelines([bad, pending], workers=1, gpu_workers=0)
    assert result["status"] == "incomplete"
    assert result["results"][0]["status"] == "failed"
    assert result["results"][1]["status"] == "not_started"
    assert not (Path(pending["workspace"]) / ".pipeline").exists()


def test_rss_guard_stops_owned_worker_and_releases_locks(tmp_path, trajectory):
    pytest.importorskip("psutil")
    value = job(tmp_path, trajectory)
    with pytest.raises(MemoryError, match="RSS"):
        run_pipelines([value], workers=1, gpu_workers=0, rss_limit_mb=.001)
    # The failed run retained its log; a normal invocation can acquire the lock.
    assert run_pipelines([value], workers=1, gpu_workers=0)["status"] == "complete"


def test_cli_manifest_paths_are_relative_to_file(tmp_path, trajectory, monkeypatch, capsys):
    pytest.importorskip("psutil")
    value = job(tmp_path, trajectory, "relative");value["workspace"] = "relative"
    path = tmp_path / "jobs.json";path.write_text(json.dumps([value]), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere";elsewhere.mkdir();monkeypatch.chdir(elsewhere)
    main(["run", str(path), "--workers", "1", "--gpu-workers", "0", "--native-threads", "1"])
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert (tmp_path / "relative" / "integrated.duckdb").exists()


def test_mcp_entrypoint_can_launch_independent_worker(tmp_path, trajectory):
    pytest.importorskip("psutil")
    pytest.importorskip("mcp")
    import asyncio
    import datetime
    import os
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    value = job(tmp_path, trajectory, "mcp-job")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"),
               MINITOPOL_WORKSPACE=str(tmp_path / "mcp-default"))

    async def run():
        with (tmp_path / "mcp.log").open("w") as log:
            async with stdio_client(StdioServerParameters(command=sys.executable,
                    args=["-m", "minitopol.mcp.server"], env=env), errlog=log) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=datetime.timedelta(minutes=2)) as client:
                    await client.initialize()
                    result = await client.call_tool("run_trajectory_pipelines", {
                        "jobs": [value], "workers": 1, "gpu_workers": 0, "native_threads": 1})
                    assert not result.isError
                    report = result.structuredContent
                    if report is None:
                        report = json.loads(next(item.text for item in result.content if item.type == "text"))
                    if set(report) == {"result"}:
                        report = report["result"]
                    assert report["status"] == "complete"
    asyncio.run(run())
