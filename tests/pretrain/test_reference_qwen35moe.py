"""Gates for the independent Qwen3.5-MoE reference: fwd/bwd smoke, a tiny
model learns, the scalar-loss convention holds, and byte-identical-init
agreement with GoldenQwen35Moe (forward + LBL-OFF curve + the LBL-ON leg).
The golden is a second witness, not automatic truth (the engine-service
parity smoke closes the loop). Exercises the full hybrid stack — Gated
DeltaNet + gated attention mixers — with the routed MoE + sigmoid-gated
shared expert tail on every layer."""
import pytest
import torch

from reference_models.qwen35moe import Qwen35Moe, Qwen35MoeConfig

TINY = dict(
    n_layers=4, d_model=64, full_attention_interval=4, n_heads=4,
    n_kv_heads=2, head_dim=16, partial_rotary_factor=0.25, lin_k_heads=2,
    lin_v_heads=4, lin_k_head_dim=16, lin_v_head_dim=16, lin_conv_kernel=4,
    n_experts=8, top_k=2, d_ff_expert=64, n_shared_experts=1, d_ff_shared=64,
    vocab_size=512,
)


def test_forward_shapes_and_init_loss():
    torch.manual_seed(0)
    cfg = Qwen35MoeConfig(**TINY)
    assert [cfg.kind_of(i) for i in range(4)] == ["lin", "lin", "lin", "full"]
    m = Qwen35Moe(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = m(tok)
    assert logits.shape == (2, 32, cfg.vocab_size)
    loss = m.loss(tok, torch.randint(0, cfg.vocab_size, (2, 32)))
    assert abs(float(loss.detach()) - torch.log(torch.tensor(512.0)).item()) < 1.0


def test_loss_conventions_and_learns():
    torch.manual_seed(0)
    cfg = Qwen35MoeConfig(**TINY)
    m = Qwen35Moe(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 32))
    tgt = torch.randint(0, cfg.vocab_size, (2, 32))
    alpha = 0.02
    composite = m.loss(tok, tgt, aux_coef=alpha)
    lbl = m.load_balance_loss()
    ce = m.loss(tok, tgt)
    assert float(lbl.detach()) > 0.0
    assert abs(float(composite.detach())
               - (float(ce.detach()) + alpha * float(lbl.detach()))) < 1e-5
    l0 = m.loss(tok, tgt)
    l0.backward()
    gnorm = sum(p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None).sqrt()
    assert torch.isfinite(gnorm) and float(gnorm) > 0
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad()
        l = m.loss(tok, tgt)
        l.backward()
        opt.step()
    assert float(l.detach()) < float(l0.detach()) - 0.5


@pytest.mark.gpu
def test_matches_golden_from_identical_init():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataclasses import replace

    from dataflow.models.qwen35moe_reference import GoldenQwen35Moe
    from dataflow.pretrain.bridges import qwen35moe as qwen35moe_bridge
    from dataflow.pretrain.crosscheck import moe_golden_gate
    from dataflow.training.models.qwen35moe import ShapedQwen35MoeConfig

    # batch=2 exercises the per-sequence reset in the conv + delta-rule
    cfg_off = replace(ShapedQwen35MoeConfig.tiny(), seq_len=64, batch=2, aux_coef=0.0)
    moe_golden_gate(cfg_off, GoldenQwen35Moe, qwen35moe_bridge, alpha=0.02)
