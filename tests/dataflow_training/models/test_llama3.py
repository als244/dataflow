"""Correctness ladder (GPU): ops (level 1), block (level 2), model (level 3),
plus plan-invariance — the flagship async/buffer-bug detector.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.blocks import ops  # noqa: E402
from dataflow_training.blocks.layouts import LlamaDims  # noqa: E402
from dataflow_training.model_families.llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow_training.testing.gradcheck import (  # noqa: E402
    check_block_backward,
    check_model_step,
    rel_l2,
)

pytestmark = pytest.mark.gpu

DIMS = LlamaDims(d_model=256, n_heads=8, n_kv_heads=2, d_ff=512, vocab_size=512, max_tokens=128, seq_len=128)
CFG = ShapedLlamaConfig(
    n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
    vocab_size=512, seq_len=128, batch=1,
)


# --- ladder level 1: ops -----------------------------------------------------

def test_rmsnorm_backward():
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    out = torch.empty_like(x)
    rstd = torch.empty(64, device="cuda", dtype=torch.float32)
    ops.rmsnorm_fwd(x, w, out, rstd)
    dx, dw = ops.rmsnorm_bwd(dy, x, rstd, w)

    xr = x.clone().requires_grad_()
    wr = w.clone().requires_grad_()
    yr = ops.rmsnorm_reference(xr, wr)
    assert rel_l2(out, yr) < 2e-2
    yr.backward(dy)
    assert rel_l2(dx, xr.grad) < 2e-2
    assert rel_l2(dw, wr.grad) < 2e-2


def test_rope_backward_is_transpose():
    """<rope(x), y> == <x, rope_bwd(y)> (rotation orthogonality)."""
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    pos = ops.Segments.uniform(128, 1).on(x.device).positions
    rx = ops.rope_fwd(x, pos, 8, 32, 500_000.0)
    ry = ops.rope_bwd(y, pos, 8, 32, 500_000.0)
    lhs = (rx.float() * y.float()).sum()
    rhs = (x.float() * ry.float()).sum()
    # normalize by the inner-product scale (the raw value can be near zero)
    scale = x.float().norm() * y.float().norm()
    assert abs(lhs.item() - rhs.item()) / scale.item() < 1e-3


def test_flash_wrapper_matches_autograd():
    q = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    d_attn = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    seg = ops.Segments.uniform(128, 1).on(q.device)
    out, lse = ops.flash_fwd(q, k, v, 8, 2, 32, cu_seqlens=seg.cu, max_seqlen=seg.max_len)
    dq, dk, dv = ops.flash_bwd(d_attn, q, k, v, out, lse, 8, 2, 32,
                               cu_seqlens=seg.cu, max_seqlen=seg.max_len)

    qr, kr, vr = (t.clone().requires_grad_() for t in (q, k, v))
    outr = ops.attention_reference(qr, kr, vr, 8, 2, 32)  # segments=None: one sequence
    assert rel_l2(out, outr) < 2e-2
    outr.backward(d_attn)
    assert rel_l2(dq, qr.grad) < 3e-2
    assert rel_l2(dk, kr.grad) < 3e-2
    assert rel_l2(dv, vr.grad) < 3e-2


def test_swiglu_backward():
    x1 = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    x3 = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ds = torch.randn_like(x1)
    dx1, dx3 = ops.swiglu_bwd(ds, x1, x3)
    x1r, x3r = x1.clone().requires_grad_(), x3.clone().requires_grad_()
    s = ops.swiglu_fwd(x1r, x3r)
    s.backward(ds)
    assert rel_l2(dx1, x1r.grad) < 2e-2
    assert rel_l2(dx3, x3r.grad) < 2e-2


def test_ce_loss_fused():
    logits = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, 512, (128,), device="cuda", dtype=torch.int32)
    loss = torch.empty(1, device="cuda", dtype=torch.float32)
    dlogits = torch.empty_like(logits)
    ops.ce_loss_fwd_bwd(logits, targets, loss, dlogits)

    lr = logits.clone().requires_grad_()
    ref = ops.ce_loss_reference(lr, targets)
    assert abs(loss.item() - ref.item()) / ref.item() < 1e-3
    ref.backward()
    assert rel_l2(dlogits, lr.grad) < 2e-2


def test_adamw_step_matches_manual():
    w = torch.randn(1000, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(1000, device="cuda", dtype=torch.bfloat16)
    m = torch.zeros_like(w)
    v = torch.zeros_like(w)
    w0 = w.clone()
    ops.adamw_step(w, g, m, v, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.01, step=1)
    # manual reference (identical formula incl. bf16 state round-trip)
    mf = (0.1 * g.float()).to(torch.bfloat16)
    vf = (0.05 * g.float() * g.float()).to(torch.bfloat16)
    mhat = mf.float() / (1 - 0.9)
    vhat = vf.float() / (1 - 0.95)
    expect = (w0.float() - 1e-3 * (mhat / (vhat.sqrt() + 1e-8) + 0.01 * w0.float())).to(torch.bfloat16)
    assert torch.equal(w, expect)
    assert torch.equal(m, mf) and torch.equal(v, vf)


def test_embed_roundtrip():
    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    tokens = torch.randint(0, 512, (128,), device="cuda", dtype=torch.int32)
    y = torch.empty(128, 256, device="cuda", dtype=torch.bfloat16)
    ops.embed_fwd(tokens, w, y)
    assert torch.equal(y, w[tokens.long()])
    dy = torch.randn_like(y)
    dw = torch.empty_like(w)
    ops.embed_bwd_accum(tokens, dy, dw, zero_first=True)
    ref = torch.zeros_like(w).index_add_(0, tokens.long(), dy)
    assert rel_l2(dw, ref) < 1e-3


# --- ladder level 2: block ------------------------------------------------------

def test_block_backward_vs_autograd():
    check_block_backward(DIMS, tol=3e-2).assert_ok()


# --- ladder level 3: full model step through the real engine ----------------------

def test_model_step_vs_golden():
    check_model_step(CFG, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()


def test_plan_invariance():
    """Different plans (budgets + recompute) must produce the same math."""
    r1 = check_model_step(CFG, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2)
    r2 = check_model_step(CFG, fast_memory_capacity=8 * 1024 * 1024, tol=3e-2)
    levels = {f"A_0_0_{i}": 1 for i in range(CFG.n_layers)}
    r3 = check_model_step(
        CFG, fast_memory_capacity=8 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
    )
    for r in (r1, r2, r3):
        r.assert_ok()


def test_model_step_muon_policy_golden_parity():
    """opt_policy="muon" through the REAL engine vs the policy-dispatched
    golden: matrix fields (wq/wk/wv/wo/w1/w2/w3) take the registry muon
    step (bf16 momentum, nesterov, NS5, Moonshot scaling — both sides run
    the same aten math), norms and embed/head resolve to adamw via the
    recipe's fragment rules (embed/head through their ns-prefixed keys).
    This is the muon-training golden gate the optimizer abstraction was
    missing."""
    from dataclasses import replace

    cfg = replace(ShapedLlamaConfig.tiny(), opt_policy="muon")
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()
