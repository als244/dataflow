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


def underfull_pipeline(cfg, *, steps_probe: int = 3):
    """A deterministic pipeline whose rounds CANNOT all fill exactly:
    the real shard corpus in per-document mode under no-split ffd
    packing — variable doc lengths under-fill the tiny geometries,
    and the data is LEARNABLE, so the curve-health min_drop check
    stays meaningful (uniform-random synthetic tokens have nothing to
    learn and can never drop). Probes the first steps and asserts
    real under-fill — a gate that accidentally feeds full rounds
    would pass vacuously."""
    from dataflow_training.data.pipeline import pipeline_from_args

    probe = pipeline_from_args(cfg, "shards:")(None)
    fills = [rnd.fill_ratio
             for _ in range(steps_probe)
             for rnd in probe.next_step().rounds]
    assert min(fills) < 0.999, (
        f"underfull gate is vacuous: fills {fills} — the corpus packs "
        f"these rounds exactly; shrink the preset geometry")
    return pipeline_from_args(cfg, "shards:")


def run_family_parity(cfg, *, steps: int = 15, slab_gib: float = 6.0,
                      budget_gib: float = 4.0, data: str = "block") -> None:
    from dataflow_training.run import parity
    from dataflow_training.run.driver import daemon_client, run_engine, run_reference
    from dataflow_training.data.pipeline import (
        legacy_block_pipeline,
        legacy_doc_pipeline,
    )
    from dataflow_training.run.recipe import Recipe

    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=3, total_steps=steps)
    feed = (legacy_doc_pipeline(cfg) if data == "doc"
            else underfull_pipeline(cfg) if data == "underfull"
            else legacy_block_pipeline(cfg))

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


# --- under-full rounds: content re-view engine vs content reference -------
#
# The data-feed no-split default leaves rounds under-full; the engine
# executes ONLY content rows (per-launch [:rows] views), the reference
# slices the same content. Any family whose blocks mis-derive a row
# count (layout constant, stale-tail read, non-token-major ctx field)
# diverges here — 15 steps of optimizer compounding amplify a wrong dW
# into the loss curve. The pipeline helper asserts real under-fill, so
# the gate cannot pass vacuously.


@pytest.mark.gpu
@pytest.mark.parametrize("preset_name", [
    "smoke_preset",            # llama3
    "gpt2_smoke_preset",
    "qwen3_smoke_preset",
    "qwen35_smoke_preset",
    "olmoe_smoke_preset",
    "qwen3moe_smoke_preset",
    "qwen35moe_smoke_preset",
    "dsv3_smoke_preset",
    "dsv32_smoke_preset",
    "glm52_smoke_preset",
])
def test_underfull_engine_vs_reference(preset_name):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run import presets

    cfg = getattr(presets, preset_name)()
    run_family_parity(cfg, data="underfull")


@pytest.mark.gpu
def test_underfull_execute_padding_equivalence():
    """The two under-full execution modes must agree with each other:
    content re-view (default) vs --execute-padding (masked tail runs,
    contributing exactly zero). Same feed, same seed, engine vs engine —
    the loss channel must track within kernel-shape noise. Pins that
    the flip is a pure execution-strategy choice, not a semantics one."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.run import parity
    from dataflow_training.run.driver import daemon_client, run_engine
    from dataflow_training.run.presets import gpt2_smoke_preset
    from dataflow_training.run.recipe import Recipe

    cfg = gpt2_smoke_preset()
    steps = 15
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=3,
                    total_steps=steps)
    feed = underfull_pipeline(cfg)
    with daemon_client(slab_gib=6.0, log=quiet_log) as client:
        content = run_engine(client, cfg, recipe, feed, steps,
                             budget_gib=4.0, seed=11, log=quiet_log)
    with daemon_client(slab_gib=6.0, log=quiet_log) as client:
        padded = run_engine(client, cfg, recipe, feed, steps,
                            budget_gib=4.0, seed=11, log=quiet_log,
                            execute_padding=True)
    rep = parity.compare(content.losses, padded.losses)
    assert rep.step0_abs < 5e-3, rep.summary()
    assert rep.passed, rep.summary()


@pytest.mark.gpu
def test_underfull_poisoned_tail_is_dead_bytes():
    """The sharp content re-view instrument: feed under-full rounds
    whose TAIL BYTES are illegal — token ids past the vocab (an
    embedding gather on them device-asserts) and absurd targets. If
    every task truly computes only content rows, the curve is
    byte-for-byte the clean run's; any read past content dies loudly
    or moves the loss."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataclasses import replace as dc_replace

    from dataflow_training.data.pipeline import PrepackedPipeline
    from dataflow_training.run.driver import daemon_client, run_engine
    from dataflow_training.run.presets import gpt2_smoke_preset
    from dataflow_training.run.recipe import Recipe

    cfg = gpt2_smoke_preset()
    steps = 6
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=steps)

    stepper = underfull_pipeline(cfg)(None)
    packed_steps = [stepper.next_step() for _ in range(steps)]

    def poisoned(step):
        rounds = []
        for rnd in step.rounds:
            tokens = rnd.tokens.copy()
            targets = rnd.targets.copy()
            n = rnd.content
            tokens[n:] = cfg.vocab_size + 12345   # gather-asserts if read
            targets[n:] = -(2 ** 30)
            rounds.append(dc_replace(rnd, tokens=tokens, targets=targets))
        return dc_replace(step, rounds=tuple(rounds))

    assert any(rnd.content < rnd.tokens.shape[0]
               for s in packed_steps for rnd in s.rounds),         "poison gate is vacuous: no under-full round in the window"

    with daemon_client(slab_gib=6.0, log=quiet_log) as client:
        clean = run_engine(client, cfg, recipe,
                           PrepackedPipeline(packed_steps), steps,
                           budget_gib=4.0, seed=11, log=quiet_log)
    with daemon_client(slab_gib=6.0, log=quiet_log) as client:
        dirty = run_engine(client, cfg, recipe,
                           PrepackedPipeline([poisoned(s)
                                              for s in packed_steps]),
                           steps, budget_gib=4.0, seed=11, log=quiet_log)
    assert clean.losses == dirty.losses, (clean.losses, dirty.losses)
