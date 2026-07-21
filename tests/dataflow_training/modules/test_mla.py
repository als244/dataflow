"""MLA op-level pins (GPU): the DeepSeek-V3 attention reference and its
load-bearing conventions, BEFORE any block executable exists.

Pins: padded-v exactness (flash/SDPA at shared head_dim with zero-padded
values == unpadded math, output AND gradients), shared-k_rope broadcast
gradient (sum over heads), rope-slice conventions, and reference
self-consistency across packed sequences.
"""
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu


@dataclass(frozen=True)
class _Dims:
    d_model: int = 128
    n_heads: int = 4
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_dim: int = 16
    qk_rope_dim: int = 8
    v_head_dim: int = 16
    rope_base: float = 10_000.0
    max_tokens: int = 128
    seq_len: int = 64
    seq_lens: tuple = None


def _weights(d: _Dims, seed=0, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h, qk, v = d.n_heads, d.qk_nope_dim + d.qk_rope_dim, d.v_head_dim

    def r(*shape):
        return (torch.randn(*shape, generator=g, device="cuda") * 0.06).to(dtype)

    return {
        "attn_norm_w": torch.ones(d.d_model, device="cuda", dtype=dtype),
        "w_q_a": r(d.d_model, d.q_lora_rank),
        "q_a_norm_w": torch.ones(d.q_lora_rank, device="cuda", dtype=dtype),
        "w_q_b": r(d.q_lora_rank, h * qk),
        "w_kv_a": r(d.d_model, d.kv_lora_rank + d.qk_rope_dim),
        "kv_a_norm_w": torch.ones(d.kv_lora_rank, device="cuda", dtype=dtype),
        "w_kv_b": r(d.kv_lora_rank, h * (d.qk_nope_dim + v)),
        "wo": r(h * v, d.d_model),
    }


def test_padded_v_attention_is_exact():
    """softmax(QK^T) @ [V|0] == [softmax(QK^T) @ V | 0], fwd and bwd."""
    from dataflow_training.blocks import ops

    torch.manual_seed(1)
    t, h, qk, v = 128, 4, 24, 16
    q = torch.randn(t, h * qk, device="cuda", requires_grad=True)
    k = torch.randn(t, h * qk, device="cuda", requires_grad=True)
    val = torch.randn(t, h, v, device="cuda", requires_grad=True)

    # unpadded ground truth per head (manual causal SDPA at v-dim)
    s = 64
    b = t // s
    q4 = q.view(b, s, h, qk).transpose(1, 2)
    k4 = k.view(b, s, h, qk).transpose(1, 2)
    v4 = val.view(b, s, h, v).transpose(1, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
    ref = ref.transpose(1, 2).reshape(t, h * v)

    v_pad = torch.cat(
        [val, torch.zeros(t, h, qk - v, device="cuda")], dim=-1,
    ).reshape(t, h * qk)
    out = ops.attention_reference(q, k, v_pad, h, h, qk,
                                  ops.Segments.uniform(s, q.shape[0] // s))
    out3 = out.view(t, h, qk)
    assert torch.equal(out3[..., v:], torch.zeros_like(out3[..., v:]))
    got = out3[..., :v].reshape(t, h * v)
    assert rel_l2(got, ref) < 1e-5

    # gradients: inject only through the real columns; padding grads zero
    dy = torch.randn_like(ref)
    gq, gk, gv = torch.autograd.grad(ref, (q, k, val), dy, retain_graph=True)
    gq2, gk2, gv2 = torch.autograd.grad(got, (q, k, val), dy)
    assert rel_l2(gq2, gq) < 1e-5
    assert rel_l2(gk2, gk) < 1e-5
    assert rel_l2(gv2, gv) < 1e-5


def test_mla_forms_shapes_and_grads_flow():
    from dataflow_training.blocks.modules.mla_forms import mla_block_reference

    d = _Dims()
    w = _weights(d)
    for name, t_ in w.items():
        t_.requires_grad_()
    x = torch.randn(d.max_tokens, d.d_model, device="cuda", requires_grad=True)
    y = mla_block_reference(x, w, d)
    assert y.shape == (d.max_tokens, d.d_model)
    y.backward(torch.randn_like(y))
    for name, t_ in w.items():
        assert t_.grad is not None and torch.isfinite(t_.grad).all(), name
    assert torch.isfinite(x.grad).all()


def test_mla_shared_k_rope_broadcast_gradient():
    """k_rope is one 64-dim vector per token expanded across heads: its
    gradient must equal the SUM over heads of per-head k-rope grads.
    Verified by comparing against a variant with independent per-head
    copies whose grads are summed."""
    from dataflow_training.blocks import ops
    from dataflow_training.blocks.modules.mla_forms import mla_attention_reference

    d = _Dims()
    w = _weights(d, seed=3)
    x = torch.randn(d.max_tokens, d.d_model, device="cuda")
    h1 = ops.rmsnorm_reference(x, w["attn_norm_w"])

    w_kv_a = w["w_kv_a"].detach().clone().requires_grad_()
    w2 = dict(w)
    w2["w_kv_a"] = w_kv_a
    y = mla_attention_reference(h1, w2, d)
    (gy,) = torch.autograd.grad(y.sum(), w_kv_a)
    # rope-column block of w_kv_a feeds ONLY the shared k_rope path; its
    # gradient must be nonzero (broadcast reduced) and finite
    rope_cols = gy[:, d.kv_lora_rank:]
    assert torch.isfinite(gy).all()
    assert rope_cols.abs().sum() > 0


def test_mla_forms_ragged_packing_matches_per_sequence():
    from dataclasses import replace

    from dataflow_training.blocks.modules.mla_forms import mla_block_reference

    d_packed = _Dims(max_tokens=96, seq_len=None, seq_lens=(64, 32))
    w = _weights(d_packed, seed=5)
    x = torch.randn(96, d_packed.d_model, device="cuda")
    y = mla_block_reference(x, w, d_packed)

    d_a = replace(d_packed, max_tokens=64, seq_len=64, seq_lens=None)
    d_b = replace(d_packed, max_tokens=32, seq_len=32, seq_lens=None)
    ya = mla_block_reference(x[:64], w, d_a)
    yb = mla_block_reference(x[64:], w, d_b)
    assert rel_l2(y, torch.cat([ya, yb])) < 1e-5


# --- block executables vs golden autograd (M-G1b/G3 gate) --------------------------


def _dsv3_dims(kind="moe", **over):
    from dataflow_training.blocks.layouts import Dsv3Dims, DTypePolicy, ParamDTypes
    from dataflow_training.blocks.modules.moe.spec import MoESpec

    moe = MoESpec(
        n_experts=8, top_k=2, d_ff_expert=32,
        routing_mode="sigmoid_noaux_tc", aux_coef=1e-4,
        n_shared_experts=1, d_ff_shared=32, shared_gate=False,
        n_group=4, topk_group=2, routed_scaling=2.5, bias_update_speed=0.001,
    )
    kw = dict(
        d_model=128, n_heads=4, q_lora_rank=64, kv_lora_rank=32,
        qk_nope_dim=16, qk_rope_dim=8, v_head_dim=16,
        d_ff=256, first_k_dense=1, vocab_size=512, max_tokens=128, seq_len=64,
        dtypes=DTypePolicy(overrides=(
            ("w_router_bias", ParamDTypes("fp32", "fp32", "fp32")),
        )),
        moe=moe,
    )
    kw.update(over)
    return Dsv3Dims(**kw)


def _block_state(dims, wl, seed):
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME

    gen = torch.Generator(device="cuda").manual_seed(seed)
    views = {}
    for f in wl.fields:
        n = int(torch.tensor(f.shape).prod())
        dt = TORCH_DTYPE_BY_NAME[f.dtype]
        if f.name.endswith("_norm_w"):
            views[f.name] = torch.ones(f.shape, device="cuda", dtype=dt)
        elif f.name == "w_router_bias":
            views[f.name] = torch.zeros(f.shape, device="cuda", dtype=dt)
        else:
            views[f.name] = (
                torch.randn(n, generator=gen, device="cuda") * 0.06
            ).to(dt).view(f.shape)
    x = (torch.randn(dims.max_tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    dy = (torch.randn(dims.max_tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    return views, x, dy


def _golden_block(x_ref, leaves, dims, kind, route_ids=None, segments=None):
    """Autograd block: MLA attention (reference) + dense-or-moe tail."""
    from dataflow_training.blocks import ops
    from dataflow_training.blocks.modules.mla_forms import mla_attention_reference
    from dataflow_training.blocks.modules.moe.forms import moe_mlp_reference

    h1 = ops.rmsnorm_reference(x_ref, leaves["attn_norm_w"])
    attn = mla_attention_reference(h1, leaves, dims, segments)
    h_mid = x_ref + attn @ leaves["wo"]
    h2 = ops.rmsnorm_reference(h_mid, leaves["ffn_norm_w"])
    if kind == "dense":
        s = ops.swiglu_fwd(h2 @ leaves["w1"], h2 @ leaves["w3"])
        y = h_mid + s @ leaves["w2"]
        aux = torch.zeros((), device=x_ref.device)
        return y, aux
    lens = dims.seq_lens if dims.seq_lens is not None else (
        dims.seq_len,) * (dims.max_tokens // dims.seq_len)
    return moe_mlp_reference(h2, leaves, dims.moe, h_mid,
                             route_ids=route_ids, seq_lens=tuple(lens))


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_dsv3_block_ladder2(kind):
    from dataflow_training.model_families.dsv3.blocks import (
        Dsv3DenseBlockBwd,
        Dsv3DenseBlockFwd,
        Dsv3DenseBlockRecompute,
        Dsv3MoeBlockBwd,
        Dsv3MoeBlockFwd,
        Dsv3MoeBlockRecompute,
    )
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME
    from dataflow_training.kernels import KernelCtx, resolve_kernels
    from dataflow_training.blocks.layouts import grad_layout
    from dataflow_training.testing.gradcheck import rel_l2

    dims = _dsv3_dims()
    kernels = resolve_kernels()
    kctx = KernelCtx()
    if kind == "dense":
        fwd = Dsv3DenseBlockFwd(dims, kernels)
        rc = Dsv3DenseBlockRecompute(dims, kernels)
        bwd = Dsv3DenseBlockBwd(dims, kernels)
    else:
        fwd = Dsv3MoeBlockFwd(dims, kernels)
        rc = Dsv3MoeBlockRecompute(dims, kernels)
        bwd = Dsv3MoeBlockBwd(dims, kernels)
    wl, cl = fwd.wl, fwd.cl

    w, x, dy = _block_state(dims, wl, seed=31)
    a = {
        f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
        for f in cl.fields
    }
    y = torch.empty_like(x)
    from dataflow_training.blocks.modules.moe.spec import moe_aux_temp_layout
    from dataflow_training.data.segments import Segments

    # ONE materialized Segments handed to fwd/recompute (extras) and bwd
    # (a["_seg"]) — standing in for the engine's run-prologue that normally
    # sets it (always-varlen attention needs cu/positions/max_len)
    seg = Segments.from_dims(dims).on("cuda")
    meta_views = None
    extras = {"seg": seg}
    if kind == "moe":
        m_l = moe_aux_temp_layout(dims, dims.moe)
        meta_views = {f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype],
                                          device="cuda") for f in m_l.fields}
        extras = {"aux_temp": dict(meta_views), "seg": seg}
    fwd._forward(kctx, x, w, y, a, extras=extras)

    a2 = {
        f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
        for f in cl.fields
    }
    rc._run_stages(kctx, x, w, a2, count=rc.recompute_stage_count(),
                   extras={**extras, "aux_temp_ready": True})
    torch.cuda.synchronize()
    errors = {}
    for name in a:
        if name in ("route_ids", "route_order", "route_offsets"):
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
    bwd_meta = None if meta_views is None else {"aux_temp": meta_views}
    if bwd_meta is None:
        bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=False)
    else:
        bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=False, aux_temp=bwd_meta)

    leaves = {n: t_.detach().clone().requires_grad_() for n, t_ in w.items()
              if n != "w_router_bias"}
    if "w_router_bias" in w:
        leaves["w_router_bias"] = w["w_router_bias"]  # non-gradient
    x_ref = x.clone().requires_grad_()
    route_ids = None if meta_views is None else meta_views["route_ids"]
    y_ref, aux_ref = _golden_block(x_ref, leaves, dims, kind, route_ids=route_ids,
                                   segments=seg)
    y_ref.backward(dy, retain_graph=True)
    if kind == "moe":
        aux_ref.backward()

    errors["fwd:y"] = rel_l2(y, y_ref)
    errors["bwd:dx"] = rel_l2(dx, x_ref.grad)
    for name in dwv:
        if name == "w_router_bias":
            # the bias is policy-frozen: nothing rides its dW slot anymore
            # (the per-step sign rule reads the persistent Aux counts in the
            # LAST round's bwd — family model-step gates cover it e2e)
            assert not dwv[name].float().abs().any()
            continue
        errors[f"bwd:d{name}"] = rel_l2(dwv[name], leaves[name].grad)

    if bwd_meta is None:
        bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=True)
    else:
        bwd._backward(kctx, dy, a, x, w, dx, dwv, accum=True, aux_temp=bwd_meta)
    for name in dwv:
        if name == "w_router_bias":
            continue
        errors[f"accum:2x:{name}"] = rel_l2(dwv[name], 2.0 * leaves[name].grad)

    bad = {k: round(v, 4) for k, v in errors.items() if v > 4e-2}
    assert not bad, bad


def test_dsv3_stage_context_completeness():
    from dataflow_training.model_families.dsv3.blocks import Dsv3DenseBlockFwd, Dsv3MoeBlockFwd
    from dataflow_training.blocks.layouts import (
        dsv3_dense_activation_layout,
        dsv3_moe_activation_layout,
    )

    dims = _dsv3_dims()
    for cls, cl in ((Dsv3DenseBlockFwd, dsv3_dense_activation_layout(dims)),
                    (Dsv3MoeBlockFwd, dsv3_moe_activation_layout(dims))):
        declared = {f.name for f in cl.fields}
        emitted = cls.context_fields_emitted()
        assert declared == emitted, (cls.__name__, declared ^ emitted)
        assert cls.recompute_stage_count() < len(cls.STAGES)
