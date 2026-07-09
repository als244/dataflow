"""The infra gate: a tiny real-vocab model trained via BOTH the pytorch
reference and the service daemon from a byte-identical init on the identical
fineweb stream must produce aligned loss curves. Boots an in-process daemon;
GPU + a few seconds."""
import pytest
import torch


@pytest.mark.gpu
def test_reference_vs_service_parity_smoke():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.pretrain import parity, presets as P
    from dataflow.pretrain.driver import daemon_client, run_engine, run_reference
    from dataflow.pretrain.fineweb import make_stream
    from dataflow.pretrain.recipe import Recipe

    cfg = P.smoke_preset()
    steps = 15
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=3, total_steps=steps)
    stream = make_stream(cfg.tokens)
    quiet = lambda *_a, **_k: None

    ref = run_reference(cfg, recipe, stream, steps, seed=11, log=quiet)
    with daemon_client(slab_gib=4.0, log=quiet) as client:
        eng = run_engine(client, cfg, recipe, stream, steps, budget_gib=4.0,
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
