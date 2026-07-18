"""Ragged-packing (varlen-first) gates: real engine vs golden with a
genuinely non-uniform per-round sequence pattern.

Varlen-first: the round's Segments carries ragged bounds end to end.
Activations were already
[total_tokens, d]; these tests pin the SEQUENCE-DEPENDENT semantics under
ragged packing — rope positions resetting per sequence, block-diagonal
causal attention (per-seq flash calls + the (heads, tokens) lse layout),
and the qwen3.5 conv window / delta-rule state resets via cu_seqlens.
"""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.blocks import ops  # noqa: E402
from dataflow_training.model_families.llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow_training.testing.gradcheck import check_model_step, rel_l2  # noqa: E402

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
    sr = ops.Segments(tuple(RAGGED)).on("cuda")
    out, lse = ops.flash_fwd(q, k, v, h, kvh, hd, cu_seqlens=sr.cu, max_seqlen=sr.max_len)
    assert lse.shape == (h, t)
    aq = q.detach().clone().requires_grad_()
    ak = k.detach().clone().requires_grad_()
    av = v.detach().clone().requires_grad_()
    ref = ops.attention_reference(aq, ak, av, h, kvh, hd, sr)
    assert rel_l2(out, ref) < 2e-3
    ref.backward(dy)
    dq, dk, dv = ops.flash_bwd(dy, q, k, v, out, lse, h, kvh, hd, cu_seqlens=sr.cu, max_seqlen=sr.max_len)
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

    from dataflow_training.model_families.qwen35 import ShapedQwen35Config

    cfg = replace(ShapedQwen35Config.tiny(), seq_lens=RAGGED)
    grad_tol, min_cos, budget = _FAMILY_GRAD_GATE["qwen35"]
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_QWEN35_BIAS_ATOL, grad_tol=grad_tol,
                     min_cosine=min_cos,
                     counts_budget=budget).assert_ok()


def test_varlen_single_launch_matches_reference_autograd():
    """The C3 single-launch cu_seqlens path vs NAIVE autograd through
    the block-diagonal reference (the C3 kernel gates compared
    flash-vs-flash; this is flash-vs-naive), incl. a pad-tail-like
    short final segment."""
    segs = (73, 38, 12, 5)                    # sum 128, short tail
    t, h, kvh, hd = sum(segs), 8, 2, 64
    gen = torch.Generator(device="cuda").manual_seed(1)

    def r(*shape):
        return (torch.randn(*shape, device="cuda", generator=gen)
                * 0.5).to(torch.bfloat16)

    q, k, v, dy = r(t, h * hd), r(t, kvh * hd), r(t, kvh * hd), r(t, h * hd)
    cu = torch.tensor([0, 73, 111, 123, 128], dtype=torch.int32,
                      device="cuda")
    out, lse = ops.flash_fwd(q, k, v, h, kvh, hd,
                             cu_seqlens=cu, max_seqlen=t)
    aq = q.detach().clone().requires_grad_()
    ak = k.detach().clone().requires_grad_()
    av = v.detach().clone().requires_grad_()
    ref = ops.attention_reference(aq, ak, av, h, kvh, hd, ops.Segments(tuple(segs)))
    assert rel_l2(out, ref) < 2e-3
    ref.backward(dy)
    dq, dk, dv = ops.flash_bwd(dy, q, k, v, out, lse, h, kvh, hd,
                               cu_seqlens=cu, max_seqlen=t)
    assert rel_l2(dq, aq.grad) < 3e-2
    assert rel_l2(dk, ak.grad) < 3e-2
    assert rel_l2(dv, av.grad) < 3e-2


def _ragged_partition(cfg):
    """A deliberately unaligned partition of one round's tokens."""
    t = cfg.seq_len * cfg.batch
    a = t // 2 + 3
    b = t // 4 + 1
    return (a, b, t - a - b)


# Sub-noise bias fields (zero-init, ~1e-6 KL/router grads): a one-AdamW-step
# update is +-lr*sign(noise) on BOTH engine and golden, so rel_l2 there is a
# coin flip — the |a-b|<=atol envelope catches real garbage. Same class the
# per-family dsv32/glm52 tests envelope with _BIAS_ATOL.
_DSA_BIAS_ATOL = {"w_router_bias": 2.5e-3, "idx_k_ln_b": 2.5e-4}
# qwen3.5's zero-init decay bias: true grad below the bf16 noise floor,
# one-step update = +-lr * sign(noise). The old GOLDEN shared the
# engine's noise (identical packed order) so rel_l2 agreed by luck of
# construction; the ISOLATED twin computes its own noise, making the
# sign a coin flip — the |a-b| <= ~2lr envelope is the honest gate.
_QWEN35_BIAS_ATOL = {"dt_bias": 2.5e-4}


# Per-family gradient bands live in gradcheck.FAMILY_GRAD_GATE (shared
# by every ladder-3 suite); see docs/correctness_compare.md for the
# calibration and tier mechanisms.
from dataflow_training.testing.gradcheck import FAMILY_GRAD_GATE as _FAMILY_GRAD_GATE


@pytest.mark.parametrize("family", [
    "qwen3", "olmoe", "dsv3", "dsv32", "glm52", "qwen3moe", "qwen35moe",
])
def test_model_step_ragged_all_families(family):
    """Ragged model step vs golden (naive autograd reference) for the
    seven families not covered by the two original ragged gates —
    with llama3 + qwen35 above, all NINE families gate the
    sequence-dependent semantics (positions reset, block-diagonal
    attention / state resets, per-field grads) under packing.

    dsv32/glm52 (DSA) were xfail here under a mislabeled "top-k tie-break"
    reason; the real cause was the index-scores triton store dropping its
    column mask (out-of-bounds cols spilled via the row stride, corrupting
    causal cells for any L%64!=0 — every ragged length). With that fixed the
    selection matches the golden exactly; the only residual is the sub-noise
    bias sign-lottery, enveloped below exactly as the per-family gates do."""
    from dataclasses import replace
    import importlib

    mod = importlib.import_module(f"dataflow_training.model_families.{family}")
    cfg_cls = next(v for k, v in vars(mod).items()
                   if k.startswith("Shaped") and k.endswith("Config"))
    cfg = cfg_cls.tiny()
    cfg = replace(cfg, seq_lens=_ragged_partition(cfg))
    atol = None
    if family in ("dsv32", "glm52"):
        atol = _DSA_BIAS_ATOL
    elif family == "dsv3":
        # noaux bias one-flip envelope: a single near-tie flip moves one
        # count by 1; a count on the exact integer mean then flips its
        # sign term by speed (minimal example: the counts-parity entry +
        # tools/debug_dsv3_bias_repro.py). 1.5x speed admits the
        # one-flip class only.
        atol = {"router_bias": 1.5e-3}   # dsv3 twin buffer name
    elif family == "qwen35moe":
        atol = _QWEN35_BIAS_ATOL
    grad_tol, min_cos, budget = _FAMILY_GRAD_GATE[family]
    check_model_step(cfg, fast_memory_capacity=96 * 1024 * 1024,
                     tol=3e-2, field_atol=atol, grad_tol=grad_tol,
                     min_cosine=min_cos,
                     counts_budget=budget).assert_ok()
