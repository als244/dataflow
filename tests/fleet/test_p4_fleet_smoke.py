"""Fleet smoke: the conductor-v1 driver end to end at 125M scale —
daemons launched from topology.toml (skipped when absent or without a
remote host), warm-up dance, weighted 6:2 round split, 12 lockstep DP
steps; asserts the loss drops and the curve stays finite. The 1B
flagship runs the same code path via tools/train_fleet.py."""
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("fleet smoke needs a topology.toml with a remote host",
                allow_module_level=True)

pytestmark = pytest.mark.fleet


def quiet(*a, **k):
    pass


def test_fleet_dp_125m_smoke():
    from dataflow_training.data.fineweb import make_stream
    from dataflow_training.distributed.fleet import run_fleet_dp
    from dataflow_training.run.presets import preset
    from dataflow_training.run.recipe import Recipe

    cfg = preset("l3_125m")
    steps = 12
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=steps)
    res = run_fleet_dp(cfg, recipe, make_stream(cfg.tokens), steps,
                       rank_rounds=(6, 2), budgets=(4.0, 4.0),
                       slabs=(12.0, 10.0), topology=TOPO, log=quiet)
    assert len(res.losses) == steps
    assert all(math.isfinite(x) for x in res.losses)
    assert res.losses[0] > 10.5           # ~ln(50304) at init
    assert res.losses[-1] < res.losses[0] - 1.5   # actually learned
