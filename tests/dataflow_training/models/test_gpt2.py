"""GPT-2 correctness ladder (GPU): layernorm/gelu ops (level 1), the
model-step dW gates uniform/ragged/tied (level 3), and the twin's packed
fp32 equivalence — the varlen-native bar for the learned-position family
(positions restart per segment; LayerNorm itself is per-token and has no
sequence axis).

Tests:
- test_layernorm_backward: the fused layernorm fwd/bwd matches the autograd reference for output and dx/dw/db.
- test_layernorm_apply_matches_fwd: layernorm_apply reproduces the fwd output byte-for-byte given the saved mean/rstd.
- test_gelu_backward_matches_autograd: tanh-approx gelu forward and the aten gelu_backward match autograd.
- test_twin_packed_matches_per_sequence_fp32: the fp32 twin's packed-sequence loss equals the token-weighted mean of the per-sequence losses.
- test_twin_rejects_overlong_segment: a segment longer than n_ctx raises ValueError in the twin forward.
- test_model_step_uniform: a uniform tiny-cfg model-step matches golden, bias params under the walk envelope.
- test_model_step_ragged: a ragged sequence-length model-step matches golden.
- test_model_step_tied: the tied embed/head variant's model-step matches golden.
- test_model_step_nobias: the bias-free variant (no b_* fields) model-step matches golden without an envelope.
- test_qkv_bias_grad_sections: c_attn.bias q/v grad sections agree tightly (cos > 0.999) while both engine and twin k sections sit at the structural-zero noise floor.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataclasses import replace  # noqa: E402

from dataflow_training.blocks import ops  # noqa: E402
from dataflow_training.model_families.gpt2 import ShapedGpt2Config  # noqa: E402
from dataflow_training.testing.gradcheck import (  # noqa: E402
    check_model_step,
    family_gate_kwargs,
    rel_l2,
)

pytestmark = pytest.mark.gpu

FAST_MEM = 96 * 1024 * 1024


# --- ladder level 1: ops -----------------------------------------------------

def test_layernorm_backward():
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    out = torch.empty_like(x)
    mean = torch.empty(64, device="cuda", dtype=torch.float32)
    rstd = torch.empty(64, device="cuda", dtype=torch.float32)
    ops.layernorm_fwd(x, w, b, out, mean, rstd)
    dx, dw, db = ops.layernorm_bwd(dy, x, mean, rstd, w)

    xr = x.clone().requires_grad_()
    wr = w.clone().requires_grad_()
    br = b.clone().requires_grad_()
    yr = ops.layernorm_reference(xr, wr, br)
    assert rel_l2(out, yr) < 2e-2
    yr.backward(dy)
    assert rel_l2(dx, xr.grad) < 2e-2
    assert rel_l2(dw, wr.grad) < 2e-2
    assert rel_l2(db, br.grad) < 2e-2


def test_layernorm_apply_matches_fwd():
    x = torch.randn(33, 415, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(415, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(415, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    mean = torch.empty(33, device="cuda", dtype=torch.float32)
    rstd = torch.empty(33, device="cuda", dtype=torch.float32)
    ops.layernorm_fwd(x, w, b, out, mean, rstd)
    assert torch.equal(ops.layernorm_apply(x, mean, rstd, w, b), out)


def test_gelu_backward_matches_autograd():
    x = torch.randn(64, 160, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    y = torch.nn.functional.gelu(x, approximate="tanh")
    dx = torch.ops.aten.gelu_backward(dy, x, approximate="tanh")

    xr = x.clone().requires_grad_()
    yr = torch.nn.functional.gelu(xr, approximate="tanh")
    assert rel_l2(y, yr) < 2e-2
    yr.backward(dy)
    assert rel_l2(dx, xr.grad) < 2e-2


# --- twin: packed fp32 equivalence (the varlen-native bar) -------------------

def test_twin_packed_matches_per_sequence_fp32():
    from reference_models.gpt2 import Gpt2, Gpt2Config

    torch.manual_seed(7)
    cfg = Gpt2Config(n_layers=2, d_model=64, n_heads=4, d_ff=160,
                     vocab_size=97, n_ctx=48)
    m = Gpt2(cfg).float()
    seq_lens = (17, 5, 26)
    t = sum(seq_lens)
    tokens = torch.randint(0, 97, (1, t))
    targets = torch.randint(0, 97, (1, t))
    packed = m.loss(tokens, targets, seq_lens=seq_lens)
    lo, acc = 0, 0.0
    for n in seq_lens:
        acc = acc + m.loss(tokens[:, lo:lo + n], targets[:, lo:lo + n]) * n
        lo += n
    per_seq = acc / t
    assert abs(packed.item() - per_seq.item()) < 1e-6


def test_twin_rejects_overlong_segment():
    from reference_models.gpt2 import Gpt2, Gpt2Config

    m = Gpt2(Gpt2Config(n_layers=1, d_model=64, n_heads=4, d_ff=160,
                        vocab_size=97, n_ctx=16))
    tokens = torch.randint(0, 97, (1, 20))
    with pytest.raises(ValueError):
        m.forward(tokens, seq_lens=(20,))


# --- ladder level 3: one full model step, engine vs reference ----------------
#
# ALL biases are ZERO-INIT, so their one-step PARAM entries are per-
# element sign lotteries wherever an element's true gradient is near
# zero (|p0|=0 removes the dilution that hides sign flips elsewhere;
# update = +-lr per element from zero adamw moments). Two flavors, one
# mechanism: c_attn.bias's k-section gradient is STRUCTURALLY zero
# (softmax is exactly invariant to the key bias — a constant added to
# every key shifts each row's scores uniformly and cancels; measured
# ~1e-6 vs 5e-4/7e-2 real q/v sections), and any real bias can have a
# few near-zero ELEMENTS (the ragged case flips ~1 of 160 in c_fc).
# The param entries gate at the ~2*lr walk scale via param_atol; the
# GRAD entries stay fully live — they are the sharp instrument (rel
# ~3e-3 measured) — plus the per-section gate below. Real training
# bounds the b_k walk via weight decay; the loss is invariant to it.
BIAS_WALK_PARAM_ATOL = {"bias": 3e-2}


@pytest.mark.sim
def test_model_step_uniform():
    check_model_step(ShapedGpt2Config.tiny(), fast_memory_capacity=FAST_MEM,
                     param_atol=BIAS_WALK_PARAM_ATOL,
                     **family_gate_kwargs("gpt2")).assert_ok()


@pytest.mark.sim
def test_model_step_ragged():
    cfg = replace(ShapedGpt2Config.tiny(), seq_lens=(35, 17, 12))
    check_model_step(cfg, fast_memory_capacity=FAST_MEM,
                     param_atol=BIAS_WALK_PARAM_ATOL,
                     **family_gate_kwargs("gpt2")).assert_ok()


@pytest.mark.sim
def test_model_step_tied():
    check_model_step(ShapedGpt2Config.tiny_tied(),
                     fast_memory_capacity=FAST_MEM,
                     param_atol=BIAS_WALK_PARAM_ATOL,
                     **family_gate_kwargs("gpt2")).assert_ok()


@pytest.mark.sim
def test_model_step_nobias():
    """The bias-free variant has no b_* fields at all — no envelope."""
    check_model_step(ShapedGpt2Config.tiny_nobias(),
                     fast_memory_capacity=FAST_MEM,
                     **family_gate_kwargs("gpt2")).assert_ok()


@pytest.mark.sim
def test_qkv_bias_grad_sections():
    """The sharp instrument behind the c_attn.bias envelope: the q/v bias
    grad sections must agree tightly (real signal, tracked at rel ~3e-3),
    and BOTH sides' k sections must sit at the structural-zero noise floor
    (softmax key-bias invariance) — a real k-gradient appearing on either
    side is a bug in that side's attention math."""
    from dataflow_training.testing.gradcheck import cos_sim

    cfg = ShapedGpt2Config.tiny()
    report = check_model_step(cfg, fast_memory_capacity=FAST_MEM,
                              capture=True, param_atol=BIAS_WALK_PARAM_ATOL,
                              **family_gate_kwargs("gpt2"))
    eg = report.states["engine_grads"]
    tg = report.states["twin_grads"]
    d = cfg.d_model
    checked = 0
    for name in sorted(eg):
        if not name.endswith("c_attn.bias"):
            continue
        e, t = eg[name], tg[name]
        for sec, sl in (("q", slice(0, d)), ("v", slice(2 * d, 3 * d))):
            assert rel_l2(e[sl], t[sl]) < 3e-2, (name, sec)
            assert cos_sim(e[sl], t[sl]) > 0.999, (name, sec)
        k_e, k_t = e[d:2 * d], t[d:2 * d]
        live = min(float(e[:d].norm()), float(e[2 * d:].norm()))
        assert float(k_e.norm().detach()) < 0.05 * live, (name, "engine k not ~0")
        assert float(k_t.norm().detach()) < 0.05 * live, (name, "twin k not ~0")
        checked += 1
    assert checked == cfg.n_layers
