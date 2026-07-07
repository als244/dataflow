"""Optimizer abstraction gates (tasks/optim.py).

Levels: per-optimizer step math vs independent inline formulas; muon
orthogonalization properties; policy dispatch + state-layout slots;
and the end-to-end gate — a MIXED per-field policy through the REAL
engine vs a hand replica applying reference steps to golden autograd
gradients. Default policy byte-stability is covered by the lowering
tripwires + every existing family ladder (all-adamw unchanged).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dataflow.tasks.optim import (
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
    # 1D field: muon == sgdm (same momentum step, no NS)
    w1, g1 = _mk(seed=3)
    m1 = torch.zeros_like(w1)
    w2, m2 = w1.clone(), m1.clone()
    OPTIMIZERS["muon"].step(None, None, _Hyper, 1, w1, g1,
                            {"m": m1}, (w1.numel(),))
    OPTIMIZERS["sgdm"].step(None, None, _Hyper, 1, w2, g1,
                            {"m": m2}, (w2.numel(),))
    assert torch.equal(w1, w2) and torch.equal(m1, m2)


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
    from dataflow.tasks.layouts import opt_state_layout, weight_layout
    from dataflow.training.families import resolve_family as _rf
    from dataflow.training.llama3 import ShapedLlamaConfig

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


def test_mixed_policy_model_step_vs_hand_replica():
    """llama3 tiny, one step, per-field policy {wq: muon, w1: sgdm,
    wo: sgd, rest adamw} through the REAL engine — final weights must
    match golden-autograd grads + independent inline reference steps."""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow.tasks.llama3_blocks import AdamWHyper
    from dataflow.training.families import resolve_family
    from dataflow.training.llama3 import ShapedLlamaConfig
    from dataflow.training.planning import plan_program
    from dataflow.training.testing.gradcheck import rel_l2

    policy = OptPolicy(default="adamw",
                       overrides=(("wq", "muon"), ("w1", "sgdm"),
                                  ("wo", "sgd")))
    cfg = replace(ShapedLlamaConfig.tiny(), opt_policy=policy)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=7)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    leaves = [pinned("W_embed"),
              [pinned(f"W_{i}") for i in range(cfg.n_layers)],
              pinned("W_head")]
    golden = fam.golden().from_packed_bytes(dims, cfg.n_layers, *leaves)
    tokens = torch_view(values["tokens_0_0"], (dims.tokens,),
                        torch.int32).long().cuda()
    targets = torch_view(values["targets_0_0"], (dims.tokens,),
                         torch.int32).long().cuda()
    for p in golden.parameters():
        p.grad = None
    golden.loss(tokens, targets).backward()

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
        else:  # muon, step 1: m = g
            out = w32 * (1 - hp.lr * hp.weight_decay)
            if w.dim() == 2 and min(w.shape) > 1:
                scale = max(1.0, w.shape[0] / w.shape[1]) ** 0.5
                out = out - hp.lr * scale * _ns_orthogonalize(g32).float()
            else:
                out = out - hp.lr * g32
        return out.to(w.dtype)

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand)

    worst = (0.0, "")
    for i in range(cfg.n_layers):
        layout, gleaves = golden.final_leaves(f"W_{i}")
        rec = result.objects.get(f"W_{i}")
        buf = (rec.backing or rec.fast).buffer
        # compare against ref_step applied to the ORIGINAL packed
        # bytes + golden autograd grads, field by field
        w0 = leaves[1][i]
        for f, (name, param) in zip(layout.fields,
                                    golden.w_blocks[i].items()):
            assert f.name == name
            dt = TORCH_DTYPE_BY_NAME[f.dtype]
            nbytes = param.numel() * dt.itemsize
            base = (w0[f.offset_bytes:f.offset_bytes + nbytes]
                    .view(dt).view(*f.shape).cuda())
            expect = ref_step(name, base.reshape(param.shape),
                              param.grad).reshape(f.shape)
            got = torch_view(buf, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                             offset_bytes=f.offset_bytes)
            r = rel_l2(got, expect)
            if r > worst[0]:
                worst = (r, f"W_{i}.{name}")
    assert worst[0] <= 3e-2, worst
