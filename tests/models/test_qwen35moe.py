"""Qwen3.5-MoE correctness ladder (GPU): the hybrid family on the pluggable
MoE module — the REUSE proof (DeltaNet/gated-attention parts inherited from
qwen35_blocks untouched; only the MLP tail is family code).

Adds over the olmoe ladder: BOTH kinds' ladder-2 (lin + full), the shared
expert (sigmoid-gated ADDITIVE combine — the flextrain warning: it is not a
(1-sigma) mixture) exercised everywhere, topk_then_softmax routing, and
the alpha=0.001 aux convention.
"""
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.training.testing.gradcheck import check_model_step, rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu


def _tiny_cfg(**over):
    from dataflow.training.models.qwen35moe import ShapedQwen35MoeConfig

    return replace(ShapedQwen35MoeConfig.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow.training.models.qwen35moe import dims_of_qwen35moe

    return dims_of_qwen35moe(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------


def test_golden_qwen35moe_trains():
    from dataflow.models.qwen35moe_reference import GoldenQwen35Moe
    from dataflow.tasks.layouts import (
        head_weight_layout,
        qwen35moe_attn_weight_layout,
        qwen35moe_lin_weight_layout,
    )

    cfg = _tiny_cfg()
    dims = _tiny_dims(cfg)
    gen = torch.Generator().manual_seed(0)

    def packed(layout):
        flat = (torch.randn(layout.total_bytes // 2, generator=gen) * 0.02).to(torch.bfloat16)
        for f in layout.fields:
            start = f.offset_bytes // 2
            n = int(torch.tensor(f.shape).prod())
            if f.name.endswith("_norm_w"):
                flat[start : start + n] = 1.0
            elif f.name == "A_log":
                flat[start : start + n] = (
                    torch.empty(n).uniform_(1.0, 16.0, generator=gen).log().to(torch.bfloat16)
                )
            elif f.name == "dt_bias":
                flat[start : start + n] = 0.0
        return flat.view(torch.uint8)

    def table():
        n = dims.vocab_size * dims.d_model
        return (torch.randn(n, generator=gen) * 0.02).to(torch.bfloat16).view(torch.uint8)

    golden = GoldenQwen35Moe.from_packed_bytes(
        dims, cfg.n_layers, table(),
        [
            packed(
                qwen35moe_attn_weight_layout(dims) if dims.kind_of(i) == "full"
                else qwen35moe_lin_weight_layout(dims)
            )
            for i in range(cfg.n_layers)
        ],
        packed(head_weight_layout(dims)),
    )
    toks = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen).cuda()
    tgts = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen).cuda()
    ce0, aux0 = golden.loss_terms(toks, tgts)
    assert torch.isfinite(aux0) and float(aux0.detach()) > 0.0
    losses = [golden.train_step(toks, tgts) for _ in range(3)]
    assert all(x == x for x in losses)
    assert abs(losses[0] - math.log(dims.vocab_size)) < 0.5
    assert losses[-1] < losses[0]


# --- ladder 2: per-kind block fwd/recompute/bwd vs golden autograd ----------------


def _block_state(dims, wl, seed):
    # 0.06 init: the DeltaNet gate-gradient observability floor (see
    # test_qwen35_math._block_state note) AND spread router logits
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME

    gen = torch.Generator(device="cuda").manual_seed(seed)
    views = {}
    for f in wl.fields:
        n = int(torch.tensor(f.shape).prod())
        dt = TORCH_DTYPE_BY_NAME[f.dtype]
        if f.name.endswith("_norm_w"):
            views[f.name] = torch.ones(f.shape, device="cuda", dtype=dt)
        elif f.name == "A_log":
            views[f.name] = (
                torch.empty(n, device="cuda").uniform_(1.0, 16.0, generator=gen)
                .log().to(dt).view(f.shape)
            )
        elif f.name == "dt_bias":
            views[f.name] = torch.zeros(f.shape, device="cuda", dtype=dt)
        else:
            views[f.name] = (
                torch.randn(n, generator=gen, device="cuda") * 0.06
            ).to(dt).view(f.shape)
    x = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    dy = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    return views, x, dy


_INT_CTX_FIELDS = ("route_ids", "route_order", "route_offsets")


def _ladder2(kind: str, tol: float = 4e-2):
    from dataflow.models.qwen35moe_reference import GoldenQwen35Moe
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME
    from dataflow.tasks.kernels import KernelCtx, resolve_kernels
    from dataflow.tasks.layouts import grad_layout
    from dataflow.tasks.models.qwen35moe_blocks import (
        Qwen35MoeAttnBlockBwd,
        Qwen35MoeAttnBlockFwd,
        Qwen35MoeAttnBlockRecompute,
        Qwen35MoeLinBlockBwd,
        Qwen35MoeLinBlockFwd,
        Qwen35MoeLinBlockRecompute,
    )

    cfg = _tiny_cfg()
    dims = _tiny_dims(cfg)
    if kind == "lin":
        fwd_cls, rc_cls, bwd_cls = (
            Qwen35MoeLinBlockFwd, Qwen35MoeLinBlockRecompute, Qwen35MoeLinBlockBwd,
        )
    else:
        fwd_cls, rc_cls, bwd_cls = (
            Qwen35MoeAttnBlockFwd, Qwen35MoeAttnBlockRecompute, Qwen35MoeAttnBlockBwd,
        )
    kernels = resolve_kernels()
    kctx = KernelCtx()
    fwd = fwd_cls(dims, kernels)
    bwd = bwd_cls(dims, kernels)
    wl, cl = fwd.wl, fwd.cl

    w, x, dy = _block_state(dims, wl, seed=31 if kind == "lin" else 32)
    a = {
        f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
        for f in cl.fields
    }
    y = torch.empty_like(x)
    from dataflow.tasks.modules.moe.spec import moe_meta_layout

    m_l = moe_meta_layout(dims, dims.moe)
    meta_views = {f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype],
                                      device="cuda") for f in m_l.fields}
    from dataflow.tasks.ops import Segments

    seg = Segments.of_dims(dims).on("cuda")
    fwd._forward(kctx, x, w, y, a, extras={"meta": dict(meta_views), "seg": seg})

    a2 = {
        f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
        for f in cl.fields
    }
    rc_cls(dims, kernels)._run_stages(
        kctx, x, w, a2, count=rc_cls.recompute_stage_count(),
        extras={"meta": dict(meta_views), "meta_ready": True, "seg": seg},
    )
    torch.cuda.synchronize()
    errors = {}
    for name in a:
        if name in _INT_CTX_FIELDS:
            assert torch.equal(a2[name], a[name]), f"recompute int field {name}"
        else:
            errors[f"recompute:{name}"] = rel_l2(a2[name], a[name])

    gl = grad_layout(wl, dims.dtypes)
    dwv = {
        f.name: torch.zeros(f.shape, device="cuda", dtype=TORCH_DTYPE_BY_NAME[f.dtype])
        for f in gl.fields
    }
    dx = torch.empty_like(x)
    a["_seg"] = seg
    bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=False,
                  meta={"meta": meta_views})

    golden = GoldenQwen35Moe(dims=dims)
    leaves = {n: t.detach().clone().requires_grad_() for n, t in w.items()}
    x_ref = x.clone().requires_grad_()
    fwd_ref = golden.lin_block_forward if kind == "lin" else golden.full_block_forward
    # selection pinned to the runtime's (see moe-design.md: comparison
    # methodology — flips are model sensitivity, not gradient error)
    y_ref, aux_ref = fwd_ref(x_ref, leaves, route_ids=meta_views["route_ids"], segments=seg)
    y_ref.backward(dy, retain_graph=True)
    aux_ref.backward()

    errors["fwd:y"] = rel_l2(y, y_ref)
    errors["bwd:dx"] = rel_l2(dx, x_ref.grad)
    for name in dwv:
        errors[f"bwd:d{name}"] = rel_l2(dwv[name], leaves[name].grad)

    bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=True,
                  meta={"meta": meta_views})
    for name in dwv:
        errors[f"accum:2x:{name}"] = rel_l2(dwv[name], 2.0 * leaves[name].grad)

    bad = {k: round(v, 4) for k, v in errors.items() if v > tol}
    assert not bad, bad


def test_qwen35moe_lin_block_ladder2():
    _ladder2("lin")


def test_qwen35moe_attn_block_ladder2():
    _ladder2("full")


# --- structure + lowering ----------------------------------------------------------


def test_qwen35moe_stage_context_completeness():
    from dataflow.tasks.layouts import (
        qwen35moe_attn_context_layout,
        qwen35moe_lin_context_layout,
    )
    from dataflow.tasks.models.qwen35moe_blocks import Qwen35MoeAttnBlockFwd, Qwen35MoeLinBlockFwd

    dims = _tiny_dims()
    for cls, cl in (
        (Qwen35MoeLinBlockFwd, qwen35moe_lin_context_layout(dims)),
        (Qwen35MoeAttnBlockFwd, qwen35moe_attn_context_layout(dims)),
    ):
        declared = {f.name for f in cl.fields}
        emitted = cls.context_fields_emitted()
        assert declared == emitted, (cls.__name__, declared ^ emitted)
        assert cls.recompute_stage_count() < len(cls.STAGES)
        names = [s[0] for s in cls.STAGES]
        assert names[cls.recompute_stage_count():] == ["moe_experts2_combine"]


def test_qwen35moe_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program, simulate_program

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
    planned = plan_program(program, fast_memory_capacity=12 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


# --- ladder 3: full program through the real engine --------------------------------

# dt_bias one-step updates from ZERO init are +-lr * sign(sub-noise grads)
# on BOTH sides (bf16-ULP-vs-AdamW: moment rounding flips update signs
# on sub-ulp grads; ladder-2 pins
# the REAL dt gradient at observability-scale init) — compare with the
# sign-lottery envelope instead of rel_l2. 2.5e-4 = ~2.5x lr.
_ATOL = {"dt_bias": 2.5e-4}


def test_qwen35moe_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2, field_atol=_ATOL,
    ).assert_ok()


def test_qwen35moe_plan_invariance():
    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2, field_atol=_ATOL)
    r2 = check_model_step(cfg, fast_memory_capacity=12 * 1024 * 1024, tol=3e-2, field_atol=_ATOL)
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=12 * 1024 * 1024, recompute_levels=levels,
        tol=3e-2, field_atol=_ATOL,
    )
    for r in (r1, r2, r3):
        r.assert_ok()


def test_qwen35moe_batch2_packed_sequences_vs_golden():
    """Packed sequences must reset conv/recurrence at boundaries AND route
    per-token regardless of packing (MoE is token-parallel)."""
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(
        cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2, field_atol=_ATOL,
    ).assert_ok()


# --- engine-level gates ------------------------------------------------------------


def _run(engine_kwargs=None, resolver_wrapper=None, program=None, seed=7):
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    prog = program if program is not None else plan_program(
        fam.lower(cfg), fast_memory_capacity=12 * 1024 * 1024,
    ).program

    backend = CudaBackend()
    values = fam.initial_values(prog, cfg, backend, seed=seed)
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    resolver = fam.build_resolver(fam.dims_of(cfg))
    if resolver_wrapper is not None:
        resolver = resolver_wrapper(resolver, backend)
    from dataflow.runtime.engine import uniform_segments
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver, initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(fam.dims_of(cfg), prog)},
    )
    # mask alignment-padding gaps (8-byte A_log/dt_bias fields at tiny
    # scale — the qwen35 padding artifact; see test_qwen35_math._run35)
    from dataflow.tasks.layouts import (
        head_weight_layout,
        qwen35moe_attn_weight_layout,
        qwen35moe_lin_weight_layout,
    )

    dims = fam.dims_of(cfg)

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
            qwen35moe_attn_weight_layout(dims) if dims.kind_of(i) == "full"
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


def test_qwen35moe_fixed_seed_bitwise_deterministic():
    a = _run()
    b = _run()
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert torch.equal(a[k], b[k]), k


def test_qwen35moe_poison_on_free_changes_nothing():
    base = _run()
    poisoned = _run(engine_kwargs={"poison_on_free": True})
    _assert_same(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]  # not NaN


def test_qwen35moe_interleaving_stress_changes_nothing():
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


def test_qwen35moe_measured_costs_replan_still_golden():
    """Profiling the heterogeneous MoE task set (linmoe_*/gattnmoe_* keys,
    packed ctx with int32 routing fields) through the profile_fill hook;
    re-planning on measured costs must not change the math."""
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program
    from dataflow.training.profiling import apply_measured_costs, profile_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    backend = CudaBackend()
    profiles = profile_program(program, fam.build_resolver(fam.dims_of(cfg)), backend, soak_seconds=0)
    measured = apply_measured_costs(program, profiles)
    assert all("measured" in t.metadata for t in measured.tasks)

    base = _run()
    replanned = plan_program(measured, fast_memory_capacity=12 * 1024 * 1024).program
    again = _run(program=replanned)
    _assert_same(again, base)


def test_qwen35moe_multistep_matches_golden_and_loss_decreases():
    from dataflow.models.qwen35moe_reference import GoldenQwen35Moe
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program
    from dataflow.training.train_loop import train

    STEPS = 3
    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=12 * 1024 * 1024)
    backend = CudaBackend()

    gen = torch.Generator().manual_seed(99)
    one_batch = (
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
    )
    batches = [one_batch] * STEPS

    snapshot = fam.initial_values(planned.program, cfg, backend, seed=5)

    def pinned(name):
        buf = snapshot[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    golden = GoldenQwen35Moe.from_packed_bytes(
        dims, cfg.n_layers, pinned("W_embed"),
        [pinned(f"W_{i}") for i in range(cfg.n_layers)], pinned("W_head"),
    )
    golden_losses = [
        golden.train_step(t.long().cuda(), g.long().cuda()) for t, g in batches
    ]

    report = train(
        planned.program, cfg, backend,
        steps=STEPS, seed=5, token_stream=lambda s: batches[s],
    )
    for ours, ref in zip(report.losses, golden_losses):
        assert abs(ours - ref) / max(abs(ref), 1e-9) < 3e-2, (report.losses, golden_losses)
    assert report.losses[-1] < report.losses[0]
    assert all(n == 0 for n in report.step_slab_overflows[1:]), report.step_slab_overflows
