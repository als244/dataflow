"""Ragged-packing (varlen-first) gates: real engine vs golden with a
genuinely non-uniform per-round sequence pattern.

Varlen-first: seq_spec carries ragged bounds end to end. Activations
were already
[total_tokens, d]; these tests pin the SEQUENCE-DEPENDENT semantics under
ragged packing — rope positions resetting per sequence, block-diagonal
causal attention (per-seq flash calls + the (heads, tokens) lse layout),
and the qwen3.5 conv window / delta-rule state resets via cu_seqlens.
"""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.tasks import ops  # noqa: E402
from dataflow.training.llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow.training.testing.gradcheck import check_model_step, rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu

RAGGED = (73, 38, 17)  # sum 128; deliberately unaligned lengths


def test_flash_ragged_matches_reference_autograd():
    """Per-seq flash fwd/bwd (ragged path incl. GQA reduction and the
    (heads, tokens) lse layout) vs autograd through the block-diag
    reference."""
    t, h, kvh, hd = sum(RAGGED), 8, 2, 64
    gen = torch.Generator(device="cuda").manual_seed(0)

    def r(*shape):
        return (torch.randn(*shape, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)

    q, k, v, dy = r(t, h * hd), r(t, kvh * hd), r(t, kvh * hd), r(t, h * hd)
    out, lse = ops.flash_fwd(q, k, v, h, kvh, hd, RAGGED)
    assert lse.shape == (h, t)
    aq = q.detach().clone().requires_grad_()
    ak = k.detach().clone().requires_grad_()
    av = v.detach().clone().requires_grad_()
    ref = ops.attention_reference(aq, ak, av, h, kvh, hd, RAGGED)
    assert rel_l2(out, ref) < 2e-3
    ref.backward(dy)
    dq, dk, dv = ops.flash_bwd(dy, q, k, v, out, lse, h, kvh, hd, RAGGED)
    assert rel_l2(dq, aq.grad) < 3e-2
    assert rel_l2(dk, ak.grad) < 3e-2
    assert rel_l2(dv, av.grad) < 3e-2


def test_llama_model_step_ragged():
    cfg = ShapedLlamaConfig(
        n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
        vocab_size=512, seq_len=128, batch=1, seq_lens=RAGGED,
    )
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()


def test_qwen35_model_step_ragged():
    """Hybrid family under ragged packing: DeltaNet state + conv window
    reset via ragged cu_seqlens, gated attention block-diagonal."""
    from dataclasses import replace

    from dataflow.training.qwen35 import ShapedQwen35Config

    cfg = replace(ShapedQwen35Config.tiny(), seq_lens=RAGGED)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()
