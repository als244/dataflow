"""OLMoE correctness ladder (GPU): the first family on the pluggable MoE
module, mirrored on tests/models/test_qwen35.py.

Family-specific pins: full-row qk-norm (one rstd per token), the MoE tail
spliced after resid1_norm2, aux load-balance gradient injection (the
golden's autograd objective is CE + aux while its reported loss is CE),
recompute reproducing the ROUTING DECISION bit-exactly (int ctx fields
compared with torch.equal), and end-to-end engine determinism (fixed seed
twice -> identical bytes).
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
    from dataflow.training.olmoe import ShapedOlmoeConfig

    return replace(ShapedOlmoeConfig.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow.training.olmoe import dims_of_olmoe

    return dims_of_olmoe(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------


def test_golden_olmoe_trains():
    from dataflow.models.olmoe_reference import GoldenOlmoe
    from dataflow.tasks.layouts import head_weight_layout, olmoe_weight_layout

    cfg = _tiny_cfg()
    dims = _tiny_dims(cfg)
    gen = torch.Generator().manual_seed(0)

    def packed(layout):
        flat = (torch.randn(layout.total_bytes // 2, generator=gen) * 0.02).to(torch.bfloat16)
        for f in layout.fields:
            if f.name.endswith("_norm_w"):
                start = f.offset_bytes // 2
                n = int(torch.tensor(f.shape).prod())
                flat[start : start + n] = 1.0
        return flat.view(torch.uint8)

    def table():
        n = dims.vocab_size * dims.d_model
        return (torch.randn(n, generator=gen) * 0.02).to(torch.bfloat16).view(torch.uint8)

    golden = GoldenOlmoe.from_packed_bytes(
        dims, cfg.n_layers, table(),
        [packed(olmoe_weight_layout(dims)) for _ in range(cfg.n_layers)],
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


# --- ladder 2: block fwd/recompute/bwd vs golden autograd (with aux) --------------


def _block_state(dims, wl, seed):
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME

    gen = torch.Generator(device="cuda").manual_seed(seed)
    views = {}
    for f in wl.fields:
        n = int(torch.tensor(f.shape).prod())
        dt = TORCH_DTYPE_BY_NAME[f.dtype]
        if f.name.endswith("_norm_w"):
            views[f.name] = torch.ones(f.shape, device="cuda", dtype=dt)
        else:
            views[f.name] = (
                torch.randn(n, generator=gen, device="cuda") * 0.06
            ).to(dt).view(f.shape)
    x = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    dy = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    return views, x, dy


_INT_CTX_FIELDS = ("route_ids", "route_order", "route_offsets")


def test_olmoe_block_ladder2():
    from dataflow.models.olmoe_reference import GoldenOlmoe
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME
    from dataflow.tasks.kernels import KernelCtx, resolve_kernels
    from dataflow.tasks.layouts import grad_layout
    from dataflow.tasks.models.olmoe_blocks import OlmoeBlockBwd, OlmoeBlockFwd, OlmoeBlockRecompute

    cfg = _tiny_cfg()
    dims = _tiny_dims(cfg)
    kernels = resolve_kernels()
    kctx = KernelCtx()
    fwd = OlmoeBlockFwd(dims, kernels)
    bwd = OlmoeBlockBwd(dims, kernels)
    wl, cl = fwd.wl, fwd.cl

    w, x, dy = _block_state(dims, wl, seed=21)
    a = {
        f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
        for f in cl.fields
    }
    y = torch.empty_like(x)
    from dataflow.tasks.modules.moe.spec import moe_meta_layout

    m_l = moe_meta_layout(dims, dims.moe)
    meta_views = {f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype],
                                      device="cuda") for f in m_l.fields}
    fwd._forward(kctx, x, w, y, a, extras={"meta": dict(meta_views)})

    # recompute equivalence: the ROUTING DECISION must reproduce bit-exactly
    a2 = {
        f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
        for f in cl.fields
    }
    OlmoeBlockRecompute(dims, kernels)._run_stages(
        kctx, x, w, a2, count=OlmoeBlockRecompute.recompute_stage_count(),
        extras={"meta": dict(meta_views), "meta_ready": True},
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
    bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=False,
                  meta={"meta": meta_views})

    golden = GoldenOlmoe(dims=dims, n_layers=cfg.n_layers)
    leaves = {n: t.detach().clone().requires_grad_() for n, t in w.items()}
    x_ref = x.clone().requires_grad_()
    # pin the discrete selection to the runtime's (near-tie top-k flips
    # between the kernel and reference attention paths are model
    # sensitivity, not gradient error — both sides differentiate the same
    # conditional function; see moe_mlp_reference docstring)
    y_ref, aux_ref = golden.block_forward(x_ref, leaves, route_ids=meta_views["route_ids"])
    y_ref.backward(dy, retain_graph=True)
    aux_ref.backward()  # the runtime injects this gradient analytically

    errors["fwd:y"] = rel_l2(y, y_ref)
    errors["bwd:dx"] = rel_l2(dx, x_ref.grad)
    for name in dwv:
        errors[f"bwd:d{name}"] = rel_l2(dwv[name], leaves[name].grad)

    bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=True,
                  meta={"meta": meta_views})
    for name in dwv:
        errors[f"accum:2x:{name}"] = rel_l2(dwv[name], 2.0 * leaves[name].grad)

    bad = {k: round(v, 4) for k, v in errors.items() if v > 4e-2}
    assert not bad, bad


# --- structure + lowering ----------------------------------------------------------


def test_olmoe_stage_context_completeness():
    from dataflow.tasks.layouts import olmoe_context_layout
    from dataflow.tasks.models.olmoe_blocks import OlmoeBlockFwd

    cl = olmoe_context_layout(_tiny_dims())
    declared = {f.name for f in cl.fields}
    emitted = OlmoeBlockFwd.context_fields_emitted()
    assert declared == emitted, declared ^ emitted
    assert OlmoeBlockFwd.recompute_stage_count() < len(OlmoeBlockFwd.STAGES)
    # combine (the y-only epilogue) must sit past the recompute boundary
    names = [s[0] for s in OlmoeBlockFwd.STAGES]
    assert names[OlmoeBlockFwd.recompute_stage_count():] == ["moe_experts2_combine"]


def test_olmoe_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program, simulate_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    assert fam.name == "olmoe"
    program = fam.lower(cfg)
    validate_program(program)
    assert program.metadata["family"] == "olmoe-shaped"
    ids = {spec.id for spec in program.initial_objects}
    assert {"W_embed", "W_head", "O_head"} <= ids  # untied
    keys = {t.compute_block_key for t in program.tasks}
    assert {"moeattn_fwd", "moeattn_bwd", "head_loss"} <= keys
    planned = plan_program(program, fast_memory_capacity=8 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


def test_olmoe_partial_ownership_lowering_rejected():
    """Accounting for partial expert ownership is plumbed and unit-tested
    (tests/modules/test_moe.py); the PROGRAM path refuses until a multi-rank
    runtime exists."""
    import dataclasses
    import unittest.mock as mock

    from dataflow.training.olmoe import dims_of_olmoe, lower_olmoe

    cfg = _tiny_cfg()
    part = dataclasses.replace(dims_of_olmoe(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow.training.olmoe.moe_spec_of", return_value=part):
            lower_olmoe(cfg)


# --- ladder 3: full program through the real engine --------------------------------


def test_olmoe_model_step_vs_golden():
    check_model_step(_tiny_cfg(), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()


def test_olmoe_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
    ).assert_ok()


def test_olmoe_plan_invariance():
    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2)
    r2 = check_model_step(cfg, fast_memory_capacity=8 * 1024 * 1024, tol=3e-2)
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=8 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
    )
    for r in (r1, r2, r3):
        r.assert_ok()


def test_olmoe_batch2_packed_sequences_vs_golden():
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()


def test_olmoe_ga2_matches_golden():
    """Grad accumulation with the aux objective: per-round CE + per-round
    aux (per-round counts/T_r) summed across rounds — golden backwards once
    on the total; the runtime accumulates injected gradients per round."""
    from dataflow.models.olmoe_reference import GoldenOlmoe
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _tiny_cfg(grad_accum_rounds=2)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=16 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=3)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    golden = GoldenOlmoe.from_packed_bytes(
        dims, cfg.n_layers, pinned("W_embed"),
        [pinned(f"W_{i}") for i in range(cfg.n_layers)], pinned("W_head"),
    )
    total = None
    for r in range(cfg.grad_accum_rounds):
        toks = torch_view(values[f"tokens_0_{r}"], (dims.tokens,), torch.int32).long().cuda()
        tgts = torch_view(values[f"targets_0_{r}"], (dims.tokens,), torch.int32).long().cuda()
        ce_r, aux_r = golden.loss_terms(toks, tgts)
        term = ce_r + aux_r
        total = term if total is None else total + term
    total.backward()
    golden.step_count = 1
    golden._adamw_obj("embed", golden.w_embed)
    for i, leaves in enumerate(golden.w_blocks):
        golden._adamw_obj(f"block_{i}", leaves)
    golden._adamw_obj("head", golden.w_head)

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
    )

    def worst_field_err(object_id):
        rec = result.objects.get(object_id)
        slot = rec.backing or rec.fast
        layout, leaves = golden.final_leaves(object_id)
        return max(
            rel_l2(
                torch_view(slot.buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                           offset_bytes=f.offset_bytes),
                leaves[f.name],
            )
            for f in layout.fields
        )

    assert worst_field_err("W_embed") < 3e-2
    for i in range(cfg.n_layers):
        assert worst_field_err(f"W_{i}") < 3e-2, f"W_{i}"
    assert worst_field_err("W_head") < 3e-2
    result.close()
    dry.close()
    for buf in values.values():
        backend.free(buf)


# --- engine-level gates: poison / interleave / measured-replan / determinism -------


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
        fam.lower(cfg), fast_memory_capacity=8 * 1024 * 1024,
    ).program

    backend = CudaBackend()
    values = fam.initial_values(prog, cfg, backend, seed=seed)
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    resolver = fam.build_resolver(fam.dims_of(cfg))
    if resolver_wrapper is not None:
        resolver = resolver_wrapper(resolver, backend)
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver, initial_buffers=values, pool_prewarm=dry.pool_demand,
    )
    out = {}
    for obj_id in ["W_embed", "W_head"] + [f"W_{i}" for i in range(cfg.n_layers)]:
        rec = result.objects.get(obj_id)
        slot = rec.backing or rec.fast
        out[obj_id] = torch_view(slot.buffer, (rec.size_bytes // 2,), torch.bfloat16).clone()
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


def test_olmoe_fixed_seed_bitwise_deterministic():
    """Same seed, same plan, twice -> identical LOSS BYTES and weights
    (routing ties, sort, grouped GEMMs, combine: all deterministic)."""
    a = _run()
    b = _run()
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert torch.equal(a[k], b[k]), k


def test_olmoe_poison_on_free_changes_nothing():
    base = _run()
    poisoned = _run(engine_kwargs={"poison_on_free": True})
    _assert_same(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]  # not NaN


def test_olmoe_interleaving_stress_changes_nothing():
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


def test_olmoe_measured_costs_replan_still_golden():
    """The end-to-end profiling gate: every signature (incl. moeattn_bwd,
    whose packed ctx carries int32 routing fields) must profile through the
    profile_fill hook without garbage-index crashes, and re-planning on
    measured costs must not change the math."""
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
    replanned = plan_program(measured, fast_memory_capacity=8 * 1024 * 1024).program
    again = _run(program=replanned)
    _assert_same(again, base)


def test_olmoe_multistep_matches_golden_and_loss_decreases():
    from dataflow.models.olmoe_reference import GoldenOlmoe
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program
    from dataflow.training.train_loop import train

    STEPS = 3
    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=8 * 1024 * 1024)
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

    golden = GoldenOlmoe.from_packed_bytes(
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
