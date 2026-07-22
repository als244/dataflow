"""Webapp export gate: exported programs are accepted by the actual webapp backend.

Runs the FastAPI app in-process (TestClient) — the same code path as
the deployed webapp — and posts a schema-source simulation.

Tests:
- test_simulate_schema_upload: posting an exported program to /api/simulate returns a plan with positive makespan, peak under the requested cap, matching task count, and both compute and transfer tracks.
- test_preview_schema_upload: posting an exported program to /api/workloads/preview returns a task chain matching the program's task count.
"""
from dataclasses import asdict

import pytest

from dataflow.core.convert import to_webapp_program
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, build_shaped_llama3

pytest.importorskip("fastapi")
pytest.importorskip("dataflow_sim")

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from dataflow_sim.app.server.main import app

    return TestClient(app)


def _hardware_params() -> dict:
    from dataflow_sim.app.server.main import HardwareParams
    from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS

    preset_name = sorted(HARDWARE_PRESETS)[0]
    spec = asdict(HARDWARE_PRESETS[preset_name])
    fields = set(HardwareParams.model_fields)
    params = {k: v for k, v in spec.items() if k in fields}
    params["preset"] = preset_name
    return params


@pytest.mark.filterwarnings("ignore:Using .httpx. with .starlette")
def test_simulate_schema_upload(client):
    program = build_shaped_llama3(ShapedLlamaConfig.tiny())
    payload = to_webapp_program(program)

    resp = client.post(
        "/api/simulate",
        json={
            "workload": {"source": "schema", "schema": payload},
            "hardware": _hardware_params(),
            "planner": {"policy": "pressurefit", "fast_memory_capacity_gb": 0.0006},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["makespan_us"] > 0
    assert body["summary"]["peak_fast_memory_gb"] <= 0.0006
    assert len(body["chain"]["tasks"]) == len(program.tasks)
    # timeline carries compute + transfer tracks
    tracks = {iv["track"] for iv in body["log"]["task_intervals"]}
    assert "compute" in tracks and ("from_slow" in tracks or "to_slow" in tracks)


def test_preview_schema_upload(client):
    program = build_shaped_llama3(ShapedLlamaConfig.tiny())
    payload = to_webapp_program(program)
    resp = client.post(
        "/api/workloads/preview",
        json={"workload": {"source": "schema", "schema": payload}, "hardware": _hardware_params()},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["chain"]["tasks"]) == len(program.tasks)
