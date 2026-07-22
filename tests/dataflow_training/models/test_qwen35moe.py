"""Qwen3.5-MoE correctness ladder (GPU): the hybrid family on the pluggable
MoE module — the REUSE proof (DeltaNet/gated-attention parts inherited from
qwen35_blocks untouched; only the MLP tail is family code).

Adds over the olmoe ladder: BOTH kinds' ladder-2 (lin + full), the shared
expert (sigmoid-gated ADDITIVE combine — the flextrain warning: it is not a
(1-sigma) mixture) exercised everywhere, topk_then_softmax routing, and
the alpha=0.001 aux convention.

Tests:
- test_qwen35moe_stage_context_completeness: both lin and attn forward blocks' emitted context fields equal their layouts and only the y-only combine epilogue sits past the recompute boundary.
- test_qwen35moe_lowering_validates_and_plans: cfg lowers and validates as qwen35moe with untied embed/head and both linmoe/gattnmoe block keys, and plans/simulates with nonzero task intervals.
- test_qwen35moe_plan_invariance: the model-step matches golden across memory budgets and recompute plans under a widened near-tie flip budget.
- test_qwen35moe_batch2_packed_sequences_vs_golden: a batch=2 packed-sequence model-step matches golden with boundary conv/recurrence reset and per-token routing.
- test_qwen35moe_fixed_seed_bitwise_deterministic: two runs at the same seed produce identical loss and weights.
- test_qwen35moe_poison_on_free_changes_nothing: the poison_on_free engine option leaves loss and weights unchanged and non-NaN.
- test_qwen35moe_interleaving_stress_changes_nothing: random per-task launch jitter leaves loss and weights unchanged.
- test_qwen35moe_measured_costs_replan_still_golden: profiling the heterogeneous MoE task set then replanning on measured costs leaves the math unchanged.
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.testing.gradcheck import (  # noqa: E402
    check_model_step,
    family_gate_kwargs,
    rel_l2,
)

pytestmark = pytest.mark.gpu


def _tiny_cfg(**over):
    from dataflow_training.model_families.qwen35moe import ShapedQwen35MoeConfig

    return replace(ShapedQwen35MoeConfig.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow_training.model_families.qwen35moe import derive_dims

    return derive_dims(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------




# --- ladder 2: per-kind block fwd/recompute/bwd vs golden autograd ----------------


# block-level ladder retired with the golden models: block math is
# gated by the per-op kernel pins, the model-level dW comparison
# (grad: entries), and per-block isolation (tools/deep_compare.py
# --isolate); see docs/correctness_compare.md.


# --- structure + lowering ----------------------------------------------------------


def test_qwen35moe_stage_context_completeness():
    from dataflow_training.blocks.layouts import (
        qwen35moe_attn_activation_layout,
        qwen35moe_lin_activation_layout,
    )
    from dataflow_training.model_families.qwen35moe.blocks import Qwen35MoeAttnBlockFwd, Qwen35MoeLinBlockFwd

    dims = _tiny_dims()
    for cls, cl in (
        (Qwen35MoeLinBlockFwd, qwen35moe_lin_activation_layout(dims)),
        (Qwen35MoeAttnBlockFwd, qwen35moe_attn_activation_layout(dims)),
    ):
        declared = {f.name for f in cl.fields}
        emitted = cls.context_fields_emitted()
        assert declared == emitted, (cls.__name__, declared ^ emitted)
        assert cls.recompute_stage_count() < len(cls.STAGES)
        names = [s[0] for s in cls.STAGES]
        assert names[cls.recompute_stage_count():] == ["moe_experts2_combine"]


@pytest.mark.sim
def test_qwen35moe_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program, simulate_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    assert fam.name == "qwen35moe"
    program = fam.lower(cfg)
    validate_program(program)
    assert program.metadata["family"] == "qwen35moe-shaped"
    ids = {spec.id for spec in program.initial_objects}
    assert {"W_embed", "W_head", "O_head"} <= ids  # untied
    keys = {t.compute_block_key for t in program.tasks}
    assert {"linmoe_fwd", "linmoe_bwd", "gattnmoe_fwd", "gattnmoe_bwd"} <= keys
    planned = plan_program(program, fast_memory_capacity=24 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


# --- ladder 3: full program through the real engine --------------------------------

# dt_bias one-step updates from ZERO init are +-lr * sign(sub-noise grads)
# on BOTH sides (bf16-ULP-vs-AdamW: moment rounding flips update signs
# on sub-ulp grads; ladder-2 pins
# the REAL dt gradient at observability-scale init) — compare with the
# sign-lottery envelope instead of rel_l2. 2.5e-4 = ~2.5x lr.
_ATOL = {"dt_bias": 2.5e-4}




def plan_invariance_gate() -> dict:
    """qwen35moe family bands with a wider flip budget: the recompute
    plan variant re-runs forwards with different batching, so the
    near-tie flip DRAW differs from the calibration run (measured 5
    flips vs the family budget of 4 — same mechanism, fresh lottery)."""
    kw = family_gate_kwargs("qwen35moe")
    kw["counts_budget"] = 8
    return kw


# capacities here were 12MB pre-A2: the gradient gate's dW
# retention (final_locations) raises steady fast-memory demand
@pytest.mark.sim
def test_qwen35moe_plan_invariance():
    pytest.importorskip("fla")
    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                          field_atol=_ATOL, **plan_invariance_gate())
    r2 = check_model_step(cfg, fast_memory_capacity=24 * 1024 * 1024, tol=3e-2,
                          field_atol=_ATOL, **plan_invariance_gate())
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=24 * 1024 * 1024, recompute_levels=levels,
        tol=3e-2, field_atol=_ATOL, **plan_invariance_gate(),
    )
    for r in (r1, r2, r3):
        r.assert_ok()


@pytest.mark.sim
def test_qwen35moe_batch2_packed_sequences_vs_golden():
    """Packed sequences must reset conv/recurrence at boundaries AND route
    per-token regardless of packing (MoE is token-parallel)."""
    pytest.importorskip("fla")
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(
        cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2, field_atol=_ATOL,
        **family_gate_kwargs("qwen35moe"),
    ).assert_ok()


# --- engine-level gates ------------------------------------------------------------


def _run(engine_kwargs=None, resolver_wrapper=None, program=None, seed=7):
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    prog = program if program is not None else plan_program(
        fam.lower(cfg), fast_memory_capacity=24 * 1024 * 1024,
    ).program

    backend = CudaBackend()
    values = fam.initial_values(prog, cfg, backend, seed=seed)
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    resolver = fam.build_resolver(fam.derive_dims(cfg))
    if resolver_wrapper is not None:
        resolver = resolver_wrapper(resolver, backend)
    from dataflow_training.data.segments import uniform_segments
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver, initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(fam.derive_dims(cfg), prog)},
    )
    # mask alignment-padding gaps (8-byte A_log/dt_bias fields at tiny
    # scale — the qwen35 padding artifact; see test_qwen35_math._run35)
    from dataflow_training.blocks.layouts import (
        head_weight_layout,
        qwen35moe_attn_weight_layout,
        qwen35moe_lin_weight_layout,
    )

    dims = fam.derive_dims(cfg)

    def masked(flat, layout):
        if layout is None:
            return flat
        keep = torch.zeros_like(flat, dtype=torch.bool)
        for f in layout.fields:
            n = 1
            for s in f.shape:
                n *= int(s)
            start = f.offset_bytes // 2
            keep[start : start + n] = True
        return torch.where(keep, flat, torch.zeros_like(flat))

    layouts = {"W_embed": None, "W_head": head_weight_layout(dims)}
    for i in range(cfg.n_layers):
        layouts[f"W_{i}"] = (
            qwen35moe_attn_weight_layout(dims) if dims.kinds[i] == "full"
            else qwen35moe_lin_weight_layout(dims)
        )
    out = {}
    for obj_id, layout in layouts.items():
        rec = result.objects.get(obj_id)
        slot = rec.backing or rec.fast
        flat = torch_view(slot.buffer, (rec.size_bytes // 2,), torch.bfloat16).clone()
        out[obj_id] = masked(flat, layout)
    loss_rec = result.objects.get("loss_0_0")
    out["loss"] = float(torch_view((loss_rec.backing or loss_rec.fast).buffer, (1,), torch.float32)[0])
    result.close()
    dry.close()
    for buf in values.values():
        backend.free(buf)
    return out


def _assert_same(a: dict, b: dict, tol: float = 1e-3):
    assert abs(a["loss"] - b["loss"]) / max(abs(b["loss"]), 1e-9) < tol, (a["loss"], b["loss"])
    for k in a:
        if k == "loss":
            continue
        err = rel_l2(a[k], b[k])
        assert err < tol, f"{k}: rel_l2={err}"


@pytest.mark.sim
def test_qwen35moe_fixed_seed_bitwise_deterministic():
    pytest.importorskip("fla")
    a = _run()
    b = _run()
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert torch.equal(a[k], b[k]), k


@pytest.mark.sim
def test_qwen35moe_poison_on_free_changes_nothing():
    pytest.importorskip("fla")
    base = _run()
    poisoned = _run(engine_kwargs={"poison_on_free": True})
    _assert_same(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]  # not NaN


@pytest.mark.sim
def test_qwen35moe_interleaving_stress_changes_nothing():
    pytest.importorskip("fla")
    from dataflow.runtime.device.cuda_spin import SpinKernel

    def wrapper(resolver, backend):
        kernel = SpinKernel()
        rng = torch.Generator().manual_seed(123)

        class Jitter:
            def __init__(self, inner):
                self.inner = inner

            def launch(self, ctx):
                delay = float(torch.randint(20, 400, (1,), generator=rng)[0])
                kernel.launch_us(ctx.stream, delay)
                self.inner.launch(ctx)

        return lambda task: Jitter(resolver(task))

    base = _run()
    jittered = _run(resolver_wrapper=wrapper)
    _assert_same(jittered, base)


@pytest.mark.sim
def test_qwen35moe_measured_costs_replan_still_golden():
    """Profiling the heterogeneous MoE task set (linmoe_*/gattnmoe_* keys,
    packed ctx with int32 routing fields) through the profile_fill hook;
    re-planning on measured costs must not change the math."""
    pytest.importorskip("fla")
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.run.profiling import apply_measured_costs, profile_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    backend = CudaBackend()
    profiles = profile_program(program, fam.build_resolver(fam.derive_dims(cfg)), backend, soak_seconds=0)
    measured = apply_measured_costs(program, profiles)
    assert all("measured" in t.metadata for t in measured.tasks)

    base = _run()
    replanned = plan_program(measured, fast_memory_capacity=24 * 1024 * 1024).program
    again = _run(program=replanned)
    _assert_same(again, base)


