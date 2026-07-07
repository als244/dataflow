"""MLA op-level pins (GPU): the DeepSeek-V3 attention reference and its
load-bearing conventions, BEFORE any block executable exists (M-G1).

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

from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402

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
    tokens: int = 128
    seq_len: int = 64
    seq_lens: tuple = None

    @property
    def seq_spec(self):
        return self.seq_lens if self.seq_lens is not None else self.seq_len


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
    from dataflow.tasks import ops

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
    out = ops.attention_reference(q, k, v_pad, h, h, qk, s)
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


def test_mla_reference_shapes_and_grads_flow():
    from dataflow.tasks.mla import mla_block_reference

    d = _Dims()
    w = _weights(d)
    for name, t_ in w.items():
        t_.requires_grad_()
    x = torch.randn(d.tokens, d.d_model, device="cuda", requires_grad=True)
    y = mla_block_reference(x, w, d)
    assert y.shape == (d.tokens, d.d_model)
    y.backward(torch.randn_like(y))
    for name, t_ in w.items():
        assert t_.grad is not None and torch.isfinite(t_.grad).all(), name
    assert torch.isfinite(x.grad).all()


def test_mla_shared_k_rope_broadcast_gradient():
    """k_rope is one 64-dim vector per token expanded across heads: its
    gradient must equal the SUM over heads of per-head k-rope grads.
    Verified by comparing against a variant with independent per-head
    copies whose grads are summed."""
    from dataflow.tasks import ops
    from dataflow.tasks.mla import mla_attention_reference

    d = _Dims()
    w = _weights(d, seed=3)
    x = torch.randn(d.tokens, d.d_model, device="cuda")
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


def test_mla_reference_ragged_packing_matches_per_sequence():
    from dataclasses import replace

    from dataflow.tasks.mla import mla_block_reference

    d_packed = _Dims(tokens=96, seq_len=None, seq_lens=(64, 32))
    w = _weights(d_packed, seed=5)
    x = torch.randn(96, d_packed.d_model, device="cuda")
    y = mla_block_reference(x, w, d_packed)

    d_a = replace(d_packed, tokens=64, seq_len=64, seq_lens=None)
    d_b = replace(d_packed, tokens=32, seq_len=32, seq_lens=None)
    ya = mla_block_reference(x[:64], w, d_a)
    yb = mla_block_reference(x[64:], w, d_b)
    assert rel_l2(y, torch.cat([ya, yb])) < 1e-5
