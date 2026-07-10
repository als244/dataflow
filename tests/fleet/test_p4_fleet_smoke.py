"""P4 fleet smoke (DATAFLOW_TUBINGEN=1): the conductor-v1 driver end
to end at 125M scale — ssh-launched remote daemon, warm-up dance,
weighted 6:2 round split, 12 lockstep DP steps; asserts the loss
drops and the curve stays finite. The 1B flagship runs the same code
path via tools/pretrain_dp.py."""
import math
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
if not os.environ.get("DATAFLOW_TUBINGEN"):
    pytest.skip("fleet smoke needs DATAFLOW_TUBINGEN=1",
                allow_module_level=True)

pytestmark = pytest.mark.fleet


def quiet(*a, **k):
    pass


def test_fleet_dp_125m_smoke():
    from dataflow.pretrain.fineweb import make_stream
    from dataflow.pretrain.fleet import run_fleet_dp
    from dataflow.pretrain.presets import preset
    from dataflow.pretrain.recipe import Recipe

    cfg = preset("l3_125m")
    steps = 12
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=steps)
    res = run_fleet_dp(cfg, recipe, make_stream(cfg.tokens), steps,
                       rank_rounds=(6, 2), budgets=(4.0, 4.0),
                       slabs=(12.0, 10.0), log=quiet)
    assert len(res.losses) == steps
    assert all(math.isfinite(x) for x in res.losses)
    assert res.losses[0] > 10.5           # ~ln(50304) at init
    assert res.losses[-1] < res.losses[0] - 1.5   # actually learned
