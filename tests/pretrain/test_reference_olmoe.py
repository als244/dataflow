"""Gates for the independent OLMoE reference: fwd/bwd smoke, a tiny model
learns, the scalar-loss convention holds (composite == CE + alpha*LBL, CE
reported), and — the load-bearing one — byte-identical-init agreement with
GoldenOlmoe: forward + LBL-OFF training curve + the LBL-ON leg (CE channel,
alpha-scaled LBL term). The golden is a second witness, not automatic truth
(the engine-service parity smoke closes the loop against the real engine).
Exercises full-row qk-norm + softmax_then_topk routing."""
import pytest
import torch

from reference_models.olmoe import Olmoe, OlmoeConfig

TINY = dict(n_layers=2, d_model=64, n_heads=4, n_kv_heads=4, head_dim=16,
            n_experts=8, top_k=2, d_ff_expert=64, vocab_size=512)


def test_forward_shapes_and_init_loss():
    torch.manual_seed(0)
    cfg = OlmoeConfig(**TINY)
    m = Olmoe(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = m(tok)
    assert logits.shape == (2, 16, cfg.vocab_size)
    loss = m.loss(tok, torch.randint(0, cfg.vocab_size, (2, 16)))
    assert abs(float(loss.detach()) - torch.log(torch.tensor(512.0)).item()) < 1.0


def test_loss_conventions_and_learns():
    """composite == CE + alpha*LBL exactly (the pinned scalar convention:
    CE is the reported channel, the LBL term rides separately), and a tiny
    model learns."""
    torch.manual_seed(0)
    cfg = OlmoeConfig(**TINY)
    m = Olmoe(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))
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

    from dataflow.models.olmoe_reference import GoldenOlmoe
    from dataflow.pretrain.bridges import olmoe as olmoe_bridge
    from dataflow.pretrain.crosscheck import moe_golden_gate
    from dataflow.training.models.olmoe import ShapedOlmoeConfig

    cfg_off = replace(ShapedOlmoeConfig.tiny(), seq_len=64, batch=2, aux_coef=0.0)
    moe_golden_gate(cfg_off, GoldenOlmoe, olmoe_bridge, alpha=0.02)
