"""Optimizer abstraction gates (tasks/optim.py).

Levels: per-optimizer step math vs independent inline formulas; muon
orthogonalization properties; policy dispatch + state-layout slots;
and the end-to-end gate — a MIXED per-field policy through the REAL
engine vs a hand replica applying reference steps to the engine's own
retained dW gradient slabs (exactly the bytes the optimizer kernels
consumed). Default policy byte-stability is covered by the lowering
tripwires + every existing family ladder (all-adamw unchanged).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dataflow_training.blocks.optim import (
    OPTIMIZERS,
    OptPolicy,
    _ns_orthogonalize,
    resolve_opt_policy,
)

# ---------------------------------------------------------------- units


class _Hyper:
    lr, beta1, beta2, eps, weight_decay, momentum = \
        1e-2, 0.9, 0.95, 1e-8, 0.1, 0.9


def _mk(n=257, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, generator=g, device="cuda") * 0.1).to(dtype)
    grad = (torch.randn(n, generator=g, device="cuda") * 0.01).to(dtype)
    return w, grad


def test_sgd_step_matches_inline_formula():
    w, g = _mk()
    w0 = w.clone()
    OPTIMIZERS["sgd"].step(None, None, _Hyper, 1, w, g, {}, (w.numel(),))
    expect = (w0.float() * (1 - _Hyper.lr * _Hyper.weight_decay)
              - _Hyper.lr * g.float()).to(w.dtype)
    assert torch.equal(w, expect)


def test_sgdm_step_matches_inline_formula():
    w, g = _mk(seed=1)
    m = (torch.randn_like(w.float()) * 0.02).to(torch.bfloat16)
    w0, m0 = w.clone(), m.clone()
    OPTIMIZERS["sgdm"].step(None, None, _Hyper, 1, w, g,
                            {"m": m}, (w.numel(),))
    m_expect32 = m0.float() * _Hyper.momentum + g.float()
    assert torch.equal(m, m_expect32.to(m.dtype))
    w_expect = (w0.float() * (1 - _Hyper.lr * _Hyper.weight_decay)
                - _Hyper.lr * m_expect32).to(w.dtype)
    assert torch.equal(w, w_expect)


def test_muon_orthogonalizes_2d_and_falls_back_1d():
    g = torch.Generator(device="cuda").manual_seed(2)
    m = torch.randn(64, 128, generator=g, device="cuda")
    o = _ns_orthogonalize(m).float()
    # NS5 with the standard (3.4445, -4.775, 2.0315) coefficients drives
    # singular values into a band around ~0.9, NOT exactly 1 (speed over
    # exactness — the Muon convention). Property: well-conditioned band.
    sv = torch.linalg.svdvals(o)
    assert 0.5 < sv.min() and sv.max() < 1.3, (sv.min(), sv.max())
    # and it preserves direction: positive alignment with the input
    assert (o * m).sum() > 0
    # 1D field: muon = NESTEROV momentum step (no NS), momentum-dtype
    # arithmetic (the flextrain convention)
    w1, g1 = _mk(seed=3)
    m1 = torch.zeros_like(w1)
    w0 = w1.clone()
    OPTIMIZERS["muon"].step(None, None, _Hyper, 1, w1, g1,
                            {"m": m1}, (w1.numel(),))
    gm = g1.to(torch.bfloat16)
    eff = gm.add(gm, alpha=_Hyper.momentum).float()
    expect = (w0.float() * (1 - _Hyper.lr * _Hyper.weight_decay)
              - _Hyper.lr * eff).to(w1.dtype)
    assert torch.equal(m1, gm) and torch.equal(w1, expect)


def test_muon_recipe_classification_and_3d():
    from dataflow_training.kernels.muon import (
        ns_orthogonalize_batched as _ns_orthogonalize_batched,
    )
    from dataflow_training.blocks.optim import resolve_opt_policy

    r = resolve_opt_policy("muon")
    assert r.for_field("wq", None, (256, 256)) == "muon"
    assert r.for_field("w13_experts", None, (8, 128, 256)) == "muon"
    for key, shape in (("attn_norm_w", (128,)), ("embed.w", (512, 128)),
                       ("head.w", (512, 128)), ("w_router", (128, 8)),
                       ("w_idx_q", (64, 256)), ("dt_bias", (32,))):
        assert r.for_field(key, None, shape) == "adamw", key
    # overrides beat the rules
    r2 = resolve_opt_policy("muon").__class__(overrides=(("w_router", "muon"),))
    assert r2.for_field("w_router", None, (128, 8)) == "muon"
    # batched NS: every expert slice lands in the singular-value band
    g = torch.Generator(device="cuda").manual_seed(5)
    stack = torch.randn(4, 48, 96, generator=g, device="cuda")
    o = _ns_orthogonalize_batched(stack).float()
    for b in range(4):
        sv = torch.linalg.svdvals(o[b])
        assert 0.5 < sv.min() and sv.max() < 1.3


# ------------------------------------------------------ policy + layout


def test_policy_dispatch_and_validation():
    p = OptPolicy(default="adamw",
                  overrides=(("w?", "muon"), ("embed.*", "sgd")))
    assert p.for_field("wq") == "muon"
    assert p.for_field("embed.w") == "sgd"
    assert p.for_field("attn_norm_w") == "adamw"
    assert resolve_opt_policy("sgdm").default == "sgdm"
    with pytest.raises(ValueError):
        resolve_opt_policy("adamw2")


def test_opt_state_layout_slots_follow_policy():
    from dataflow_training.blocks.layouts import opt_state_layout, weight_layout
    from dataflow_training.model_families.families import resolve_family as _rf
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    dims = _rf(ShapedLlamaConfig.tiny()).dims_of(ShapedLlamaConfig.tiny())
    wl = weight_layout(dims)
    base = opt_state_layout(wl, dims.dtypes)                # all adamw
    assert {f.name.split("_", 1)[0] for f in base.fields} == {"m", "v"}
    none = opt_state_layout(wl, dims.dtypes, opt_policy="sgd")
    assert none.total_bytes == 0 and not none.fields
    mixed = opt_state_layout(
        wl, dims.dtypes,
        opt_policy=OptPolicy(default="sgd", overrides=(("wq", "adamw"),
                                                       ("wk", "sgdm"))))
    names = [f.name for f in mixed.fields]
    assert names == ["m_wq", "v_wq", "m_wk"]


# --------------------------------------------------------------- E2E
#
# The E2E gates feed each hand replica the ENGINE'S OWN dW slabs — the
# exact gradient bytes the optimizer kernels consumed — so they isolate
# optimizer-step math from forward/backward parity (covered elsewhere).


def retain_grad_slabs(program):
    """Pin every dW object's terminal placement to fast so the gradient
    slabs survive pool recycling for post-run readout (final_locations
    is the designed retention API)."""
    dw_ids = {o.id for t in program.tasks for o in t.outputs
              if o.id.startswith("dW")}
    dw_ids.update(o.id for o in program.initial_objects
                  if o.id.startswith("dW"))
    return replace(program,
                   final_locations={**dict(program.final_locations),
                                    **{oid: "fast" for oid in dw_ids}})


def block_dw_id(records, layer):
    """The block-gradient object id for ``layer`` ("dW_{s}_{i}") from
    the result's object table — embed/head grads have their own
    prefixes and single-step programs have exactly one candidate."""
    cands = [oid for oid in records
             if oid.startswith("dW") and oid.endswith(f"_{layer}")
             and not oid.startswith(("dW_embed", "dW_head"))]
    assert len(cands) == 1, (layer, cands)
    return cands[0]


def engine_grad_fields(result, dw_id, weights, dims, ns=None, layer=None):
    """{field name: tensor view} of one retained dW slab, unpacked with
    the grad layout mirroring ``weights``. Frozen fields carry no
    gradient storage, so they are absent by construction."""
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow_training.blocks.layouts import grad_layout

    gl = grad_layout(weights, dims.dtypes, ns=ns, layer=layer,
                     opt_policy=getattr(dims, "opt_policy", None))
    rec = result.objects.get(dw_id)
    slot = rec.fast or rec.backing
    return {f.name: torch_view(slot.buffer, f.shape,
                               TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes)
            for f in gl.fields}


def test_mixed_policy_model_step_vs_hand_replica():
    """llama3 tiny, one step, per-field policy {wq: muon, w1: sgdm,
    wo: sgd, rest adamw} through the REAL engine — final weights must
    match independent inline reference steps fed the engine's retained
    dW gradient slabs."""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow_training.blocks.base_blocks import AdamWHyper
    from dataflow_training.blocks.layouts import weight_layout
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import rel_l2

    policy = OptPolicy(default="adamw",
                       overrides=(("wq", "muon"), ("w1", "sgdm"),
                                  ("wo", "sgd")))
    cfg = replace(ShapedLlamaConfig.tiny(), opt_policy=policy)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(retain_grad_slabs(fam.lower(cfg)),
                           fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=7)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    leaves = [pinned("W_embed"),
              [pinned(f"W_{i}") for i in range(cfg.n_layers)],
              pinned("W_head")]

    hp = AdamWHyper()

    def ref_step(name, w, g):
        kind = policy.for_field(name)
        w32, g32 = w.detach().float(), g.detach().float()
        if kind == "adamw":
            m = g32 * (1 - hp.beta1)
            v = g32 * g32 * (1 - hp.beta2)
            mhat = m / (1 - hp.beta1)
            vhat = v / (1 - hp.beta2)
            out = (w32 * (1 - hp.lr * hp.weight_decay)
                   - hp.lr * mhat / (vhat.sqrt() + hp.eps))
        elif kind == "sgd":
            out = w32 * (1 - hp.lr * hp.weight_decay) - hp.lr * g32
        elif kind == "sgdm":
            out = w32 * (1 - hp.lr * hp.weight_decay) - hp.lr * g32
        else:  # muon (flextrain port), step 1 from zero state
            gm = g.detach().to(torch.bfloat16)
            eff = gm.add(gm, alpha=hp.momentum).float()
            out = w32 * (1 - hp.lr * hp.weight_decay)
            if w.dim() == 2 and min(w.shape) > 1:
                scale = 0.2 * max(w.shape) ** 0.5
                out = out - hp.lr * scale * _ns_orthogonalize(eff).float()
            else:
                out = out - hp.lr * eff
        return out.to(w.dtype)

    from dataflow_training.data.segments import uniform_segments

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)})

    worst = (0.0, "")
    for i in range(cfg.n_layers):
        layout = weight_layout(dims, layer=i)
        grads = engine_grad_fields(
            result, block_dw_id(result.objects.records, i), layout,
            dims, layer=i)
        rec = result.objects.get(f"W_{i}")
        buf = (rec.backing or rec.fast).buffer
        # compare against ref_step applied to the ORIGINAL packed
        # bytes + the engine's dW slab, field by field
        w0 = leaves[1][i]
        for f in layout.fields:
            if f.name not in grads:      # frozen: no grad, no step
                continue
            dt = TORCH_DTYPE_BY_NAME[f.dtype]
            base = (w0[f.offset_bytes:f.offset_bytes + f.nbytes]
                    .view(dt).view(*f.shape).cuda())
            expect = ref_step(f.name, base, grads[f.name])
            got = torch_view(buf, f.shape, dt,
                             offset_bytes=f.offset_bytes)
            r = rel_l2(got, expect)
            if r > worst[0]:
                worst = (r, f"W_{i}.{f.name}")
    assert worst[0] <= 3e-2, worst


def test_muon_recipe_string_model_step_vs_hand_replica():
    """opt_policy="muon" (THE RECIPE) on llama3 tiny through the real
    engine: 2D projections take nesterov-NS muon, embed/head/norms take
    adamw — verified against the engine's retained dW grads + inline
    reference steps for BOTH rules."""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow_training.blocks.base_blocks import AdamWHyper
    from dataflow_training.blocks.layouts import weight_layout
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import rel_l2
    from dataflow_training.blocks.optim import _ns_orthogonalize, resolve_opt_policy

    cfg = replace(ShapedLlamaConfig.tiny(), opt_policy="muon")
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    assert resolve_opt_policy(dims.opt_policy).for_field(
        "wq", None, (2, 2)) == "muon"
    planned = plan_program(retain_grad_slabs(fam.lower(cfg)),
                           fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=9)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    leaves = [pinned("W_embed"),
              [pinned(f"W_{i}") for i in range(cfg.n_layers)],
              pinned("W_head")]

    hp = AdamWHyper()
    policy = resolve_opt_policy("muon")

    def ref_step(name, w, g):
        w32, g32 = w.detach().float(), g.detach().float()
        if policy.for_field(name, None, tuple(w.shape)) == "adamw":
            m = g32 * (1 - hp.beta1)
            v = g32 * g32 * (1 - hp.beta2)
            out = (w32 * (1 - hp.lr * hp.weight_decay)
                   - hp.lr * (m / (1 - hp.beta1))
                   / ((v / (1 - hp.beta2)).sqrt() + hp.eps))
        else:  # muon (flextrain port), step 1 from zero state
            gm = g.detach().to(torch.bfloat16)
            eff = gm.add(gm, alpha=hp.momentum).float()
            out = w32 * (1 - hp.lr * hp.weight_decay)
            scale = 0.2 * max(w.shape) ** 0.5
            out = out - hp.lr * scale * _ns_orthogonalize(eff).float()
        return out.to(w.dtype)

    from dataflow_training.data.segments import uniform_segments

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)})

    worst = (0.0, "")
    for i in range(cfg.n_layers):
        layout = weight_layout(dims, layer=i)
        grads = engine_grad_fields(
            result, block_dw_id(result.objects.records, i), layout,
            dims, layer=i)
        rec = result.objects.get(f"W_{i}")
        buf = (rec.backing or rec.fast).buffer
        w0 = leaves[1][i]
        for f in layout.fields:
            if f.name not in grads:      # frozen: no grad, no step
                continue
            dt = TORCH_DTYPE_BY_NAME[f.dtype]
            base = (w0[f.offset_bytes:f.offset_bytes + f.nbytes]
                    .view(dt).view(*f.shape).cuda())
            expect = ref_step(f.name, base, grads[f.name])
            got = torch_view(buf, f.shape, dt, offset_bytes=f.offset_bytes)
            r = rel_l2(got, expect)
            if r > worst[0]:
                worst = (r, f"W_{i}.{f.name}")
    assert worst[0] <= 3e-2, worst


def test_layer_indexed_policy_sizes_and_model_step():
    """(layer index, param name) addressing: layer 0 -> sgd (O_0 sizes
    to ZERO bytes), other layers -> adamw. Structural sizing through
    lowering + the real-engine step vs a per-layer hand replica."""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow_training.blocks.base_blocks import AdamWHyper
    from dataflow_training.blocks.layouts import weight_layout
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import rel_l2

    policy = OptPolicy(default="adamw",
                       layer_overrides=(((0,), "sgd"),))
    cfg = replace(ShapedLlamaConfig.tiny(), opt_policy=policy)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    prog = fam.lower(cfg)
    o_sizes = {o.id: o.size_bytes for o in prog.initial_objects
               if o.id.startswith("O_") and o.id[2:].isdigit()}
    assert "O_0" not in o_sizes, o_sizes   # sgd: stateless -> O DROPPED
    assert o_sizes["O_1"] > 0               # adamw: m+v

    planned = plan_program(retain_grad_slabs(prog),
                           fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=13)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    leaves = [pinned("W_embed"),
              [pinned(f"W_{i}") for i in range(cfg.n_layers)],
              pinned("W_head")]

    hp = AdamWHyper()

    def ref_step(layer, w, g):
        w32, g32 = w.detach().float(), g.detach().float()
        if policy.for_field("any", layer, tuple(w.shape)) == "sgd":
            out = w32 * (1 - hp.lr * hp.weight_decay) - hp.lr * g32
        else:
            m = g32 * (1 - hp.beta1)
            v = g32 * g32 * (1 - hp.beta2)
            out = (w32 * (1 - hp.lr * hp.weight_decay)
                   - hp.lr * (m / (1 - hp.beta1))
                   / ((v / (1 - hp.beta2)).sqrt() + hp.eps))
        return out.to(w.dtype)

    from dataflow_training.data.segments import uniform_segments

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)})

    worst = (0.0, "")
    for i in range(cfg.n_layers):
        layout = weight_layout(dims, layer=i)
        grads = engine_grad_fields(
            result, block_dw_id(result.objects.records, i), layout,
            dims, layer=i)
        rec = result.objects.get(f"W_{i}")
        buf = (rec.backing or rec.fast).buffer
        w0 = leaves[1][i]
        for f in layout.fields:
            if f.name not in grads:      # frozen: no grad, no step
                continue
            dt = TORCH_DTYPE_BY_NAME[f.dtype]
            base = (w0[f.offset_bytes:f.offset_bytes + f.nbytes]
                    .view(dt).view(*f.shape).cuda())
            expect = ref_step(i, base, grads[f.name])
            got = torch_view(buf, f.shape, dt, offset_bytes=f.offset_bytes)
            r = rel_l2(got, expect)
            if r > worst[0]:
                worst = (r, f"W_{i}.{f.name}")
    assert worst[0] <= 3e-2, worst


def test_lr_schedules_shapes():
    from dataflow_training.blocks.optim import LRSchedule

    w = LRSchedule("wsd", warmup_steps=10, total_steps=100,
                   decay_frac=0.2, min_lr_frac=0.1)
    assert abs(w.scale(5) - 0.5) < 1e-9          # warmup ramp
    assert w.scale(10) == 1.0 and w.scale(80) == 1.0   # stable
    assert abs(w.scale(90) - 0.55) < 1e-9        # mid-decay
    assert abs(w.scale(100) - 0.1) < 1e-9        # floor
    c = LRSchedule("cosine", total_steps=100, min_lr_frac=0.1)
    assert abs(c.scale(50) - 0.55) < 1e-9
    assert abs(c.scale(100) - 0.1) < 1e-9
    assert LRSchedule("constant", total_steps=50).scale(30) == 1.0
    # the DEFAULT degenerates to 1.0 until total_steps is declared
    assert LRSchedule().scale(123) == 1.0


def test_hyper_overrides_and_schedule_model_step():
    """Baseline hyper + per-field overrides (norms: wd=0; embed: lr/10)
    + a WSD warmup schedule (step 1 of warmup 2 => scale 0.5), through
    the REAL engine vs a hand replica."""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow_training.blocks.base_blocks import AdamWHyper
    from dataflow_training.blocks.layouts import embed_weight_layout, weight_layout
    from dataflow_training.model_families.llama3.blocks import build_resolver
    from dataflow_training.blocks.optim import LRSchedule
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import rel_l2

    base_lr, wd = 1e-2, 0.1
    policy = OptPolicy(default="adamw", hyper_overrides=(
        ("*norm*", {"weight_decay": 0.0}),
        ("embed.*", {"lr": base_lr / 10}),
    ))
    hyper = AdamWHyper(lr=base_lr, weight_decay=wd,
                       schedule=LRSchedule("wsd", warmup_steps=2,
                                           total_steps=100))
    cfg = replace(ShapedLlamaConfig.tiny(), opt_policy=policy)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(retain_grad_slabs(fam.lower(cfg)),
                           fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=17)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    leaves = [pinned("W_embed"),
              [pinned(f"W_{i}") for i in range(cfg.n_layers)],
              pinned("W_head")]

    sched_scale = 0.5     # step 1 of warmup 2

    def ref_step(key, w, g):
        lr = (base_lr / 10 if key.startswith("embed.") else base_lr)
        w_decay = 0.0 if "norm" in key else wd
        lr = lr * sched_scale
        w32, g32 = w.detach().float(), g.detach().float()
        m = g32 * (1 - 0.9)
        v = g32 * g32 * (1 - 0.95)
        return (w32 * (1 - lr * w_decay)
                - lr * (m / (1 - 0.9))
                / ((v / (1 - 0.95)).sqrt() + 1e-8)).to(w.dtype)

    from dataflow_training.data.segments import uniform_segments

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program,
        resolver=build_resolver(dims, hyper=hyper),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)})

    worst = (0.0, "")
    checks = [("W_embed", "embed", None, embed_weight_layout(dims),
               leaves[0], "dW_embed_0")] + [
        (f"W_{i}", None, i, weight_layout(dims, layer=i), leaves[1][i],
         block_dw_id(result.objects.records, i))
        for i in range(cfg.n_layers)]
    for oid, ns, layer, layout, w0, dw_id in checks:
        grads = engine_grad_fields(result, dw_id, layout, dims,
                                   ns=ns, layer=layer)
        rec = result.objects.get(oid)
        buf = (rec.backing or rec.fast).buffer
        for f in layout.fields:
            key = f"{ns}.{f.name}" if ns else f.name
            if f.name not in grads:      # frozen: no grad, no step
                continue
            dt = TORCH_DTYPE_BY_NAME[f.dtype]
            base = (w0[f.offset_bytes:f.offset_bytes + f.nbytes]
                    .view(dt).view(*f.shape).cuda())
            expect = ref_step(key, base, grads[f.name])
            got = torch_view(buf, f.shape, dt, offset_bytes=f.offset_bytes)
            r = rel_l2(got, expect)
            if r > worst[0]:
                worst = (r, key)
    assert worst[0] <= 3e-2, worst
