"""Qwen3.5 correctness ladder, part 1: kernel/spec pinning (GPU).

Before any block exists, the family's math spec (pure-torch reference forms
in tasks/ops.py) is pinned three ways:
  1. our sequential delta-rule recurrence == fla's own naive reference (fp32,
     spec vs spec);
  2. fla's CHUNK kernels (the ones the blocks will call) == our recurrence
     at bf16 tolerances — forward AND backward (the backward is the
     Blackwell check: fla issue #640's Hopper workaround must not be needed
     on sm_120);
  3. the conv + l2norm helpers == their references.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.tasks import ops  # noqa: E402
from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu

T, HK, HV, K, V = 256, 2, 4, 32, 32


def _inputs(seed=0, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, HK, K, device="cuda", generator=g).to(dtype)
    k = torch.randn(T, HK, K, device="cuda", generator=g).to(dtype)
    v = (torch.randn(T, HV, V, device="cuda", generator=g) * 0.5).to(dtype)
    beta = torch.rand(T, HV, device="cuda", generator=g).to(dtype)
    a = torch.randn(T, HV, device="cuda", generator=g).to(dtype)
    A_log = (torch.empty(HV, device="cuda").uniform_(1.0, 16.0, generator=g)).log()
    dt_bias = torch.zeros(HV, device="cuda")
    g_log = ops.gated_delta_gate_reference(a, A_log, dt_bias)
    qn = ops.l2norm_reference(q)
    kn = ops.l2norm_reference(k)
    return qn, kn, v, beta, a, A_log, dt_bias, g_log


def test_reference_recurrence_matches_fla_naive():
    """Spec vs spec at fp32: our recurrence == fla's naive reference."""
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

    qn, kn, v, beta, _a, _Al, _dt, g_log = _inputs(dtype=torch.float32)
    ours = ops.gated_delta_rule_reference(qn, kn, v, beta, g_log)
    rep = HV // HK
    theirs, _ = naive_recurrent_gated_delta_rule(
        qn.repeat_interleave(rep, dim=1).unsqueeze(0),
        kn.repeat_interleave(rep, dim=1).unsqueeze(0),
        v.unsqueeze(0), beta.unsqueeze(0), g_log.unsqueeze(0),
        scale=K ** -0.5,
    )
    assert rel_l2(ours, theirs.squeeze(0).to(ours.dtype)) < 1e-5


def test_fla_chunk_fwd_matches_reference():
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    qn, kn, v, beta, a, A_log, dt_bias, g_log = _inputs(dtype=torch.bfloat16)
    ref = ops.gated_delta_rule_reference(qn, kn, v, beta, g_log)
    # fwd contract (fla 0.5.1, from ChunkGatedDeltaRuleFunction):
    # returns (g_post, o, A_int, final_state, initial_state, g_input)
    g_post, o, A_int, _fs, _is, g_input = chunk_gated_delta_rule_fwd(
        qn.unsqueeze(0), kn.unsqueeze(0), v.unsqueeze(0).contiguous(),
        a.unsqueeze(0), beta.unsqueeze(0), scale=K ** -0.5,
        initial_state=None, output_final_state=False,
        cu_seqlens=None, chunk_indices=None,
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
    )
    assert rel_l2(o.squeeze(0), ref) < 3e-2
    # g_post / A_int / g_input are opaque bwd inputs we save verbatim —
    # assert only sanity, not internal structure
    assert torch.isfinite(g_post).all() and g_post.shape[-1] == HV
    assert g_input is not None and torch.isfinite(g_input.float()).all()
    # under use_gate_in_kernel, fwd's g_input return is the RAW gate input
    # (a) passed through — gdn_gate_bwd re-derives softplus grads from it.
    # Our blocks therefore reuse the saved `ba`'s a-slice as g_input; no
    # extra context field needed.
    assert rel_l2(g_input.squeeze(0).float(), a.float()) < 1e-3


def test_fla_chunk_bwd_matches_reference_autograd():
    """THE Blackwell check: fla's chunk bwd on sm_120 vs autograd through
    our recurrence (fla #640 documents a Hopper-only Triton bwd failure;
    verify sm_120 does not need the expand/reduce workaround)."""
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_fwd,
    )

    qn, kn, v, beta, a, A_log, dt_bias, g_log = _inputs(dtype=torch.bfloat16)
    do = (torch.randn_like(v.float()) * 0.5).to(torch.bfloat16)

    # reference grads by autograd through the fp32 recurrence
    q_r = qn.float().requires_grad_()
    k_r = kn.float().requires_grad_()
    v_r = v.float().requires_grad_()
    beta_r = beta.float().requires_grad_()
    a_r = a.float().requires_grad_()
    A_r = A_log.clone().requires_grad_()
    dt_r = dt_bias.clone().requires_grad_()
    g_r = ops.gated_delta_gate_reference(a_r, A_r, dt_r)
    out = ops.gated_delta_rule_reference(q_r, k_r, v_r, beta_r, g_r)
    out.backward(do.float())

    g_post, o, A_int, _fs, _is, g_input = chunk_gated_delta_rule_fwd(
        qn.unsqueeze(0), kn.unsqueeze(0), v.unsqueeze(0).contiguous(),
        a.unsqueeze(0), beta.unsqueeze(0), scale=K ** -0.5,
        initial_state=None, output_final_state=False,
        cu_seqlens=None, chunk_indices=None,
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
    )
    # bwd contract: g = POST-cumsum gate (fwd ret 0), g_input = PRE-cumsum
    # per-token gate (fwd ret 5); returns
    # (dq, dk, dv, dbeta, da, dh0, dA_log, ddt_bias)
    dq, dk, dv, db, da, _dh0, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
        q=qn.unsqueeze(0), k=kn.unsqueeze(0), v=v.unsqueeze(0).contiguous(),
        g=g_post, beta=beta.unsqueeze(0), A=A_int,
        scale=K ** -0.5, initial_state=None, do=do.unsqueeze(0), dht=None,
        cu_seqlens=None, chunk_indices=None,
        use_gate_in_kernel=True, g_input=g_input, A_log=A_log, dt_bias=dt_bias,
    )
    assert rel_l2(dq.squeeze(0), q_r.grad) < 5e-2
    assert rel_l2(dk.squeeze(0), k_r.grad) < 5e-2
    assert rel_l2(dv.squeeze(0), v_r.grad) < 5e-2
    assert rel_l2(db.squeeze(0), beta_r.grad) < 5e-2
    assert rel_l2(da.squeeze(0), a_r.grad) < 5e-2
    assert dA_log is not None and rel_l2(dA_log, A_r.grad) < 5e-2
    assert ddt_bias is not None and rel_l2(ddt_bias, dt_r.grad) < 8e-2


def test_conv_and_l2norm_helpers_match_references():
    import fla.modules.conv.triton.ops as fops
    from fla.modules.l2norm import l2norm_fwd

    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(512, 192, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(192, 4, device="cuda", generator=g) * 0.2).to(torch.bfloat16)
    y = fops.causal_conv1d_fwd(x.unsqueeze(0), w, None, None, activation="silu")
    y = y[0] if isinstance(y, tuple) else y
    y = y.squeeze(0) if y.dim() == 3 else y
    assert rel_l2(y, ops.causal_conv1d_silu_reference(x, w)) < 2e-2

    q = torch.randn(512, 4, 32, device="cuda", generator=g).to(torch.bfloat16)
    qn, _rstd = l2norm_fwd(q.view(-1, 32))
    assert rel_l2(qn.view_as(q), ops.l2norm_reference(q)) < 2e-2


def test_golden_qwen35_trains():
    """Golden self-consistency: hybrid stack + tied head trains — loss starts
    at ~ln(vocab) (random-init sanity) and decreases on a memorized batch.
    (Runtime-vs-golden pinning is ladder 3, once the blocks exist.)"""
    import math

    from dataflow.models.qwen35_reference import GoldenQwen35
    from dataflow.tasks.layouts import (
        head_weight_layout,
        qwen35_attn_weight_layout,
        qwen35_lin_weight_layout,
    )
    from dataflow.training.shaped_qwen35 import ShapedQwen35Config, dims_of_qwen35

    cfg = ShapedQwen35Config.tiny()
    dims = dims_of_qwen35(cfg)
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

    blocks = [
        packed(
            qwen35_attn_weight_layout(dims) if dims.kind_of(i) == "full"
            else qwen35_lin_weight_layout(dims)
        )
        for i in range(dims.n_layers)
    ]
    golden = GoldenQwen35.from_packed_bytes(
        dims, dims.n_layers, packed(head_weight_layout(dims)), blocks,
    )
    toks = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen).cuda()
    tgts = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen).cuda()
    losses = [golden.train_step(toks, tgts) for _ in range(3)]
    assert all(x == x for x in losses)                       # finite
    assert abs(losses[0] - math.log(dims.vocab_size)) < 0.5  # random-init sanity
    assert losses[-1] < losses[0]
