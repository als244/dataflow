"""CI gates for examples/rl_training: five families, the custom RL
Program on the real engine vs the isolated autograd trainer.

Tests:
- test_rl_training_parity_ppo: for each family, the PPO example's engine run matches the isolated autograd trainer and emits program.json and plan.json.
- test_rl_training_parity_reinforce: the glm52 REINFORCE example's engine run matches the isolated autograd trainer and emits the program and plan artifacts.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from dataflow_training.distributed.topology import repo_root

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("dataflow_sim")
pytest.importorskip("cuda.bindings")

REPO = repo_root()
FAMILIES = ["llama3", "qwen35", "qwen3moe", "dsv32", "glm52"]

pytestmark = [pytest.mark.gpu, pytest.mark.sim, pytest.mark.vram(gib=8)]


def _run(family: str, loss: str, tmp_path: Path):
    r = subprocess.run(
        [sys.executable, str(REPO / f"examples/rl_training/{family}/run.py"),
         "--loss", loss, "--steps", "3", "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "PASS: engine == isolated autograd" in r.stdout
    assert (tmp_path / "program.json").exists()
    assert (tmp_path / "plan.json").exists()


@pytest.mark.parametrize("family", FAMILIES)
def test_rl_training_parity_ppo(tmp_path, family):
    _run(family, "ppo", tmp_path)


def test_rl_training_parity_reinforce(tmp_path):
    _run("glm52", "reinforce", tmp_path)
