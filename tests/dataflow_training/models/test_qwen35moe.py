"""Qwen3.5-MoE correctness ladder (GPU): the hybrid family on the pluggable
MoE module — the REUSE proof (DeltaNet/gated-attention parts inherited from
qwen35_blocks untouched; only the MLP tail is family code).

Adds over the olmoe ladder: BOTH kinds' ladder-2 (lin + full), the shared
expert (sigmoid-gated ADDITIVE combine — the flextrain warning: it is not a
(1-sigma) mixture) exercised everywhere, topk_then_softmax routing, and
the alpha=0.001 aux convention.

Tests:
- test_qwen35moe_stage_context_completeness: both lin and attn forward blocks' emitted context fields equal their layouts and only the y-only combine epilogue sits past the recompute boundary.
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

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

