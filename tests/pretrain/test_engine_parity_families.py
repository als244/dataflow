"""Per-family ENGINE-SERVICE-vs-reference few-step parity: the same seeded
init (``family_init_all`` server-side; the reference bridges the same bytes)
and the same deterministic fineweb stream through BOTH the pytorch reference
and an in-process dataflowd — the loss curves must align. This is the direct
engine leg of each family's cross-check triangle (reference == golden ==
engine); llama3's original version is test_parity_smoke.py, and this file
extends the gate family-by-family as bridges land. Boots one in-process
daemon per family; GPU + a few seconds each."""
import math

import pytest
import torch


def quiet_log(*args, **kwargs):
    pass


def run_family_parity(cfg, *, steps: int = 15, slab_gib: float = 6.0,
                      budget_gib: float = 4.0) -> None:
    from dataflow.pretrain import parity
    from dataflow.pretrain.driver import daemon_client, run_engine, run_reference
    from dataflow.pretrain.fineweb import make_stream
    from dataflow.pretrain.recipe import Recipe

    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=3, total_steps=steps)
    stream = make_stream(cfg.tokens)

    ref = run_reference(cfg, recipe, stream, steps, seed=11, log=quiet_log)
    with daemon_client(slab_gib=slab_gib, log=quiet_log) as client:
        eng = run_engine(client, cfg, recipe, stream, steps,
                         budget_gib=budget_gib, seed=11, log=quiet_log)

    rep = parity.compare(ref.losses, eng.losses)
    # forward on a byte-identical init must match tightly; the trajectory
    # must track within bf16 noise
    assert rep.step0_abs < 0.02, rep.summary()
    assert rep.passed, rep.summary()
    # the engine actually learned (starts near ln(V), real drop)
    ok, msg = parity.curves_healthy(eng.losses,
                                    expect_start=math.log(cfg.vocab_size),
                                    min_drop=0.1)
    assert ok, msg


@pytest.mark.gpu
def test_qwen3_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.pretrain.presets import qwen3_smoke_preset

    run_family_parity(qwen3_smoke_preset())


@pytest.mark.gpu
def test_olmoe_engine_vs_reference():
    """LBL-OFF (aux_coef=0): the reference trains pure CE, so the engine
    must too."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.pretrain.presets import olmoe_smoke_preset

    run_family_parity(olmoe_smoke_preset())


@pytest.mark.gpu
def test_qwen3moe_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.pretrain.presets import qwen3moe_smoke_preset

    run_family_parity(qwen3moe_smoke_preset())


@pytest.mark.gpu
def test_qwen35moe_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.pretrain.presets import qwen35moe_smoke_preset

    run_family_parity(qwen35moe_smoke_preset())


@pytest.mark.gpu
def test_dsv3_engine_vs_reference():
    """LBL-OFF (aux 0, bias speed 0): pure CE + AdamW on both sides."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.pretrain.presets import dsv3_smoke_preset

    run_family_parity(dsv3_smoke_preset())


@pytest.mark.gpu
def test_dsv32_engine_vs_reference():
    """LBL-OFF + train_indexer=False: the indexer stays at init on both
    sides while its selection drives the sparse attention."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.pretrain.presets import dsv32_smoke_preset

    run_family_parity(dsv32_smoke_preset())


@pytest.mark.gpu
def test_glm52_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.pretrain.presets import glm52_smoke_preset

    run_family_parity(glm52_smoke_preset())
