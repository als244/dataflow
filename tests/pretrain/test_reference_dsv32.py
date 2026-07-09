"""Gates for the independent DeepSeek-V3.2 reference: fwd/bwd smoke, a tiny
model learns, the scalar-loss convention holds, and byte-identical-init
agreement with GoldenDsv32 — forward + LBL-OFF curve + the seq-wise-aux
LBL-ON leg + the BIAS-ON leg. The golden runs with train_indexer=False so
BOTH sides keep the indexer frozen at init (the reference carries no KL
objective; CE never reaches the indexer on either side) while its selection
still drives the sparse attention. Exercises MLA + the DSA lightning
indexer (index_topk < seq_len: genuinely sparse selection)."""
import pytest
import torch

from reference_models.dsv32 import Dsv32, Dsv32Config

TINY = dict(n_layers=2, d_model=64, n_heads=4, q_lora_rank=32,
            kv_lora_rank=16, qk_nope_dim=16, qk_rope_dim=8, v_head_dim=16,
            first_k_dense=1, d_ff_dense=128, n_experts=8, top_k=2,
            d_ff_expert=32, n_group=4, topk_group=2, n_shared_experts=1,
            d_ff_shared=32, index_n_heads=4, index_head_dim=16, index_topk=8,
            vocab_size=512)


def test_forward_shapes_and_init_loss():
    torch.manual_seed(0)
    cfg = Dsv32Config(**TINY)
    m = Dsv32(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))    # T=16 > index_topk=8
    logits = m(tok)
    assert logits.shape == (2, 16, cfg.vocab_size)
    loss = m.loss(tok, torch.randint(0, cfg.vocab_size, (2, 16)))
    assert abs(float(loss.detach()) - torch.log(torch.tensor(512.0)).item()) < 1.0


def test_loss_conventions_and_learns():
    torch.manual_seed(0)
    cfg = Dsv32Config(**TINY)
    m = Dsv32(cfg)
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

    from dataflow.models.dsv32_reference import GoldenDsv32
    from dataflow.pretrain.bridges import dsv32 as dsv32_bridge
    from dataflow.pretrain.crosscheck import moe_golden_gate
    from dataflow.training.models.dsv32 import ShapedDsv32Config

    # index_topk=24 < seq_len=64: the selection is genuinely sparse.
    # lr 2e-3 (vs the dense families' 1e-2): DSA adds a second DISCRETE
    # selection mechanism, and at tiny dims bf16-drift near-tie flips are
    # dense at 1e-2 — a real mapping/formula bug still blows the envelope.
    cfg_off = replace(ShapedDsv32Config.tiny(), seq_len=64, batch=2,
                      aux_coef=0.0, bias_update_speed=0.0, train_indexer=False)
    moe_golden_gate(cfg_off, GoldenDsv32, dsv32_bridge, alpha=0.02,
                    bias_speed=1e-3, lr=2e-3)
