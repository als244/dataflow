"""Gates for the independent GLM-5.2 reference: fwd/bwd smoke, a tiny model
learns, the scalar-loss convention holds, and byte-identical-init agreement
with GoldenGlm52 — forward + LBL-OFF curve + the seq-wise-aux LBL-ON leg +
the BIAS-ON leg. train_indexer=False on the golden side keeps the indexer
frozen at init on both sides (the reference has no KL / L^I_multi
objective) while leaders' selections still drive the sparse attention and
FOLLOWERS reuse them (IndexShare). Exercises the leader/follower kind split
— followers carry NO indexer weights, which the byte-identity gate checks
structurally."""
import pytest
import torch

from reference_models.glm52 import Glm52, Glm52Config

TINY = dict(n_layers=4, d_model=64, n_heads=4, q_lora_rank=32,
            kv_lora_rank=16, qk_nope_dim=16, qk_rope_dim=8, v_head_dim=16,
            d_ff=128, first_k_dense=1, n_experts=8, top_k=2, d_ff_expert=32,
            n_group=4, topk_group=2, routed_scaling=2.5, n_shared_experts=1,
            d_ff_shared=32, index_n_heads=4, index_head_dim=16, index_topk=8,
            indexer_types=("full", "full", "shared", "shared"),
            vocab_size=512)


def test_forward_shapes_and_init_loss():
    torch.manual_seed(0)
    cfg = Glm52Config(**TINY)
    assert [cfg.is_leader(i) for i in range(4)] == [True, True, False, False]
    m = Glm52(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))    # T=16 > index_topk=8
    logits = m(tok)
    assert logits.shape == (2, 16, cfg.vocab_size)
    loss = m.loss(tok, torch.randint(0, cfg.vocab_size, (2, 16)))
    assert abs(float(loss.detach()) - torch.log(torch.tensor(512.0)).item()) < 1.0


def test_loss_conventions_and_learns():
    torch.manual_seed(0)
    cfg = Glm52Config(**TINY)
    m = Glm52(cfg)
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

    from dataflow.models.glm52_reference import GoldenGlm52
    from dataflow.pretrain.bridges import glm52 as glm52_bridge
    from dataflow.pretrain.crosscheck import moe_golden_gate
    from dataflow.training.models.glm52 import ShapedGlm52Config

    # tiny: 6 layers over gdl/gml/gmf kinds; index_topk=24 < seq_len=64.
    # lr 2e-3 for the DSA near-tie flip class (see the dsv32 gate).
    cfg_off = replace(ShapedGlm52Config.tiny(), seq_len=64, batch=2,
                      aux_coef=0.0, bias_update_speed=0.0, train_indexer=False)
    moe_golden_gate(cfg_off, GoldenGlm52, glm52_bridge, alpha=0.02,
                    bias_speed=1e-3, lr=2e-3)
