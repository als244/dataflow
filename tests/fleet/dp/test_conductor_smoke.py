"""Fleet smoke: the conductor driver end to end at 125M scale —
daemons launched from topology.toml (skipped when absent or without a
remote host), warm-up dance, an arbitrary uneven round split across
hosts (exercising weighted rounds, not a claim about relative GPU
speeds), 12 lockstep DP steps; asserts the loss drops and the curve
stays finite. The 1B flagship runs the same code path via
tools/train/train.py.

Tests:
- test_fleet_dp_125m_smoke: twelve lockstep DP steps at 125M across two hosts on an uneven (6, 2) round split return twelve finite losses that start near ln(vocab) and drop by more than 1.5.
"""
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

pytestmark = [pytest.mark.fleet, pytest.mark.gpu,
              pytest.mark.topology_remote, pytest.mark.corpus]


def quiet(*a, **k):
    pass


def test_fleet_dp_125m_smoke():
    from dataflow_training.data.pipeline import legacy_block_pipeline
    from dataflow_training.distributed.fleet import ParallelismScheme, run
    from dataflow_training.run.presets import preset
    from dataflow_training.run.recipe import Recipe

    cfg = preset("l3_125m")
    steps = 12
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=steps)
    # (6, 2) is an arbitrary uneven round split across the two hosts: it
    # exercises the weighted-round path, not a claim about relative GPU
    # speeds — correctness does not depend on the ratio.
    res = run(cfg, recipe, legacy_block_pipeline(cfg), steps,
              scheme=ParallelismScheme.data_parallel((6, 2)),
              budgets=(4.0, 4.0),
              backing=(12.0, 10.0), topology=TOPO, log=quiet)
    assert len(res.losses) == steps
    assert all(math.isfinite(x) for x in res.losses)
    assert res.losses[0] > 10.5           # ~ln(50304) at init
    assert res.losses[-1] < res.losses[0] - 1.5   # actually learned
