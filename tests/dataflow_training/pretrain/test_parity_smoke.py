"""The infra gate: a tiny real-vocab model trained via BOTH the pytorch
reference and the service daemon from a byte-identical init on the identical
fineweb feed must produce aligned loss curves. Boots an in-process daemon;
GPU + a few seconds.

Tests:
- test_reference_vs_service_parity_smoke: the reference and the service daemon, from a byte-identical init on the same fineweb feed, match at step 0, track along the trajectory, and both learn.
"""
import pytest

torch = pytest.importorskip("torch")
if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
    pytest.skip("bf16 smoke models need compute capability >= (8, 0)",
                allow_module_level=True)

pytestmark = [pytest.mark.corpus, pytest.mark.vram(gib=6)]


@pytest.mark.gpu
def test_reference_vs_service_parity_smoke():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run import parity, presets as P
    from dataflow_training.run.driver import engine_client, run_engine, run_reference
    from dataflow_training.data.pipeline import legacy_block_pipeline
    from dataflow_training.run.recipe import Recipe

    cfg = P.smoke_preset()
    steps = 15
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=3, total_steps=steps)
    feed = legacy_block_pipeline(cfg)
    quiet = lambda *_a, **_k: None

    ref = run_reference(cfg, recipe, feed, steps, seed=11, log=quiet)
    with engine_client(backing_gib=4.0, log=quiet) as client:
        eng = run_engine(client, cfg, recipe, feed, steps, budget_gib=4.0,
                         seed=11, log=quiet)

    rep = parity.compare(ref.losses, eng.losses)
    # forward on a byte-identical init must match tightly; the trajectory
    # must track within bf16 noise
    assert rep.step0_abs < 0.02, rep.summary()
    assert rep.passed, rep.summary()
    # both actually learned (start near ln(V), real drop)
    import math
    ok, msg = parity.curves_healthy(eng.losses,
                                    expect_start=math.log(cfg.vocab_size),
                                    min_drop=0.1)
    assert ok, msg
