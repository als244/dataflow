"""Grad-accum rounds are a MEMORY OPTIMIZATION, not a semantics knob.

Under the global-denominator convention (run_args ``valid_rows`` carries
the STEP's valid-token total; every round's CE normalizes by it), the
same step tokens produce the same trained weights regardless of how they
are partitioned into rounds — engine(ga=k) == engine(ga=1) up to bf16
summation-order noise. Before the convention this comparison was
ILL-POSED: per-round mean losses summed, so the CE gradient scaled with
the round count.

Legs:
- SGD (linear): the sharp witness — measured envelope 2.4e-06 (llama3,
  3 steps) / 3.8e-07 (olmoe, 2 steps); asserted with ~10x margin.
- AdamW: invariance holds within its own noise (measured 8.8e-04) —
  NOTE: Adam's per-parameter scale invariance also absorbs a WRONG
  (per-round) denominator at few steps, so AdamW weights cannot serve
  as the negative control.
- The TRIPWIRE for a path that forgets the denominator: with
  ``valid_rows`` absent, the legacy per-round normalization makes the
  round-0 loss scalar EXACTLY R times the step-normalized one (same
  forward, same nll sum, different denominator) — asserted to catch the
  forgot-the-denominator class loudly at the VALUE level; under SGD the
  weight trajectories also separate by ~40x over the invariance noise.

Tests:
- test_sgd_rounds_are_memory_optimization: under SGD, ga=4 weights match ga=1 within a tiny envelope while the no-denominator negative separates by more than 10x.
- test_adamw_rounds_within_band: under AdamW, ga=4 and ga=1 weights agree within the optimizer's own noise band.
- test_moe_rounds_are_memory_optimization: for an MoE family, ga=4 and ga=1 weights match within a tight envelope.
- test_missing_denominator_trips_loss_scale: dropping valid_rows makes the round-0 loss scalar exactly R times the step-normalized value.
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
if torch.cuda.get_device_capability() < (8, 0):
    pytest.skip("bf16 triton kernels need compute capability >= (8, 0)",
                allow_module_level=True)

from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.fake import FakeBackend  # noqa: E402
from dataflow_training.data.segments import uniform_segments  # noqa: E402
from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view  # noqa: E402
from dataflow_training.model_families.families import resolve_family  # noqa: E402
from dataflow_training.lowering.planning import plan_program  # noqa: E402
from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

T_STEP = 128
SEQ = 32


def llama_cfg(ga: int, opt: str):
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    return replace(ShapedLlamaConfig.tiny(), seq_len=SEQ,
                   batch=T_STEP // (SEQ * ga), grad_accum_rounds=ga,
                   opt_policy=opt)


def olmoe_cfg(ga: int):
    from dataflow_training.model_families.olmoe import ShapedOlmoeConfig

    return replace(ShapedOlmoeConfig.tiny(), seq_len=SEQ,
                   batch=T_STEP // (SEQ * ga), grad_accum_rounds=ga,
                   aux_coef=0.0, opt_policy="sgd")


def family_layers(cfg):
    if type(cfg).__name__ == "ShapedOlmoeConfig":
        from dataflow_training.model_families.olmoe import family_layouts
    else:
        from dataflow_training.model_families.llama3 import family_layouts
    return family_layouts(cfg)[1].layers


def run_partitioned(cfg, valid_rows, *, steps: int, seed: int = 7):
    """One engine execution; the SAME master token arrays are written
    across however many rounds the config partitions them into. Returns
    ({W_i: {field: cpu tensor}}, loss_0_0 scalar)."""
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    program = fam.lower(replace(cfg, num_steps=steps))
    program = replace(program, final_locations={
        **program.final_locations, "loss_0_0": "backing"})
    planned = plan_program(program, fast_memory_capacity=96 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=seed)
    master = torch.Generator().manual_seed(99)
    for s in range(steps):
        tok = torch.randint(0, cfg.vocab_size, (T_STEP,), generator=master,
                            dtype=torch.int32)
        tgt = torch.randint(0, cfg.vocab_size, (T_STEP,), generator=master,
                            dtype=torch.int32)
        per = T_STEP // cfg.grad_accum_rounds
        for r in range(cfg.grad_accum_rounds):
            torch_view(values[f"tokens_{s}_{r}"], (per,),
                       torch.int32).copy_(tok[r * per:(r + 1) * per])
            torch_view(values[f"targets_{s}_{r}"], (per,),
                       torch.int32).copy_(tgt[r * per:(r + 1) * per])
    run_args = {"segments": uniform_segments(dims, planned.program),
                "step": 0}
    if valid_rows is not None:
        run_args["valid_rows"] = valid_rows
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args=run_args)
    weights: dict = {}
    for i, layer in enumerate(family_layers(cfg)):
        rec = result.objects.get(f"W_{i}")
        slot = rec.backing or rec.fast
        weights[f"W_{i}"] = {
            f.name: torch_view(slot.buffer, f.shape,
                               TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes).clone().cpu()
            for f in layer.weights.fields}
    lrec = result.objects.get("loss_0_0")
    lslot = lrec.backing or lrec.fast
    loss00 = float(torch_view(lslot.buffer, (1,), torch.float32)[0])
    result.close()
    return weights, loss00


def worst_field_rel(a, b) -> float:
    return max(rel_l2(a[k][f], b[k][f]) for k in a for f in a[k])


@pytest.mark.gpu
def test_sgd_rounds_are_memory_optimization():
    a, _ = run_partitioned(llama_cfg(1, "sgd"), T_STEP, steps=3)
    b, _ = run_partitioned(llama_cfg(4, "sgd"), T_STEP, steps=3)
    d = worst_field_rel(a, b)
    assert d < 3e-5, f"ga=4 vs ga=1 (SGD) rel_l2 {d:.3e} (envelope 2.4e-06)"
    # the negative separates: legacy per-round normalization at ga=4 is a
    # DIFFERENT trajectory (4x CE gradient), well above invariance noise
    c, _ = run_partitioned(llama_cfg(4, "sgd"), None, steps=3)
    d_neg = worst_field_rel(a, c)
    assert d_neg > 10 * max(d, 1e-7), (d, d_neg)


@pytest.mark.gpu
def test_adamw_rounds_within_band():
    a, _ = run_partitioned(llama_cfg(1, "adamw"), T_STEP, steps=3)
    b, _ = run_partitioned(llama_cfg(4, "adamw"), T_STEP, steps=3)
    d = worst_field_rel(a, b)
    assert d < 5e-3, f"ga=4 vs ga=1 (AdamW) rel_l2 {d:.3e} (envelope 8.8e-04)"


@pytest.mark.gpu
def test_moe_rounds_are_memory_optimization():
    a, _ = run_partitioned(olmoe_cfg(1), T_STEP, steps=2)
    b, _ = run_partitioned(olmoe_cfg(4), T_STEP, steps=2)
    d = worst_field_rel(a, b)
    assert d < 5e-6, f"ga=4 vs ga=1 (olmoe) rel_l2 {d:.3e} (envelope 3.8e-07)"


@pytest.mark.gpu
def test_missing_denominator_trips_loss_scale():
    """A run that forgets valid_rows reverts to per-round normalization:
    the round-0 forward is identical, so its loss scalar is EXACTLY R
    times the step-normalized value — the loud, value-level tripwire."""
    _, loss_new = run_partitioned(llama_cfg(4, "sgd"), T_STEP, steps=1)
    _, loss_legacy = run_partitioned(llama_cfg(4, "sgd"), None, steps=1)
    ratio = loss_legacy / loss_new
    assert abs(ratio - 4.0) < 1e-3, ratio
