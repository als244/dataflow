"""Per-family ENGINE-SERVICE-vs-reference few-step parity: the same seeded
init (the daemon's init program; the reference bridges the same bytes)
and the same deterministic fineweb feed through BOTH the pytorch reference
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
                      budget_gib: float = 4.0, data: str = "block") -> None:
    from dataflow_training.run import parity
    from dataflow_training.run.driver import daemon_client, run_engine, run_reference
    from dataflow_training.data.fineweb import make_doc_feed, make_feed
    from dataflow_training.run.recipe import Recipe

    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=3, total_steps=steps)
    feed = (make_doc_feed(cfg.tokens, cfg.seq_len) if data == "doc"
              else make_feed(cfg.tokens))

    ref = run_reference(cfg, recipe, feed, steps, seed=11, log=quiet_log)
    with daemon_client(slab_gib=slab_gib, log=quiet_log) as client:
        eng = run_engine(client, cfg, recipe, feed, steps,
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
def test_gpt2_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import gpt2_smoke_preset

    run_family_parity(gpt2_smoke_preset())


@pytest.mark.gpu
def test_gpt2_docaware_engine_vs_reference():
    """The doc-aware varlen pipeline end to end: EOT-split rounds, the
    reference's packed mode vs the engine's run_args seq_lens — the
    standing gate for the doc-aware data path."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import gpt2_smoke_preset

    run_family_parity(gpt2_smoke_preset(), data="doc")


@pytest.mark.gpu
def test_qwen3_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import qwen3_smoke_preset

    run_family_parity(qwen3_smoke_preset())


@pytest.mark.gpu
def test_olmoe_engine_vs_reference():
    """LBL-OFF (aux_coef=0): the reference trains pure CE, so the engine
    must too."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import olmoe_smoke_preset

    run_family_parity(olmoe_smoke_preset())


@pytest.mark.gpu
def test_olmoe_engine_vs_reference_lbl_on():
    """LBL-ON (aux_coef=0.01, per-round default): the engine injects the
    analytic per-round LBL gradient; the reference autograds the per-round
    composite CE + alpha*LBL. Same per-round semantics at any ga, so the
    CE channels (the pinned scalar convention on BOTH sides) must track.
    The retained-inputs mode has no naive autograd comparator — it is
    router-only by construction (upstream aux dropped) — and is gated at
    the gradient level in tests/dataflow_training/training/test_lbl_modes.py instead."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataclasses import replace

    from dataflow_training.run.presets import olmoe_smoke_preset

    run_family_parity(replace(olmoe_smoke_preset(), aux_coef=0.01))


@pytest.mark.gpu
def test_qwen3moe_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import qwen3moe_smoke_preset

    run_family_parity(qwen3moe_smoke_preset())


@pytest.mark.gpu
def test_qwen35moe_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import qwen35moe_smoke_preset

    run_family_parity(qwen35moe_smoke_preset())


@pytest.mark.gpu
def test_dsv3_engine_vs_reference():
    """LBL-OFF (aux 0, bias speed 0): pure CE + AdamW on both sides."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import dsv3_smoke_preset

    run_family_parity(dsv3_smoke_preset())


@pytest.mark.gpu
def test_dsv32_engine_vs_reference():
    """LBL-OFF + train_indexer=False: the indexer stays at init on both
    sides while its selection drives the sparse attention."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import dsv32_smoke_preset

    run_family_parity(dsv32_smoke_preset())


@pytest.mark.gpu
def test_glm52_engine_vs_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run.presets import glm52_smoke_preset

    run_family_parity(glm52_smoke_preset())
