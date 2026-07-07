"""CI gate for examples/rl_trainer: the custom RL Program on the real
engine matches the isolated autograd trainer — both objectives."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "examples/rl_trainer/run.py"


@pytest.mark.parametrize("loss", ["ppo", "reinforce"])
def test_rl_trainer_parity(tmp_path, loss):
    art = tmp_path / "rollout.pt"
    r = subprocess.run(
        [sys.executable, str(RUN), "--loss", loss, "--steps", "3",
         "--artifacts", str(art)],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "PASS: engine == isolated autograd" in r.stdout
