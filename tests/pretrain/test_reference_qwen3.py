"""Gates for the independent qwen3-dense reference: fwd/bwd smoke, a tiny
model learns, and — the load-bearing ones — from a byte-identical init it
AGREES with the engine's GoldenQwen3 on the forward AND over a short
training curve (engine-exact AdamW both sides). The golden is a second
WITNESS, not automatic truth: a divergence indicts one of the two
implementations and is investigated in both directions (the engine-service
parity smoke in test_engine_parity_families.py closes the loop against the
real engine). Exercises the qwen3 deltas: per-head qk-norm and head_dim
decoupled from d_model (q_dim != d_model in the tiny config)."""
import pytest
import torch

from reference_models.qwen3 import Qwen3, Qwen3Config

TINY = dict(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=32,
            d_ff=160, vocab_size=512)


def test_forward_shapes_and_init_loss():
    torch.manual_seed(0)
    cfg = Qwen3Config(**TINY)
    assert cfg.q_dim == 128 and cfg.q_dim != cfg.d_model  # decoupling exercised
    m = Qwen3(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = m(tok)
    assert logits.shape == (2, 16, cfg.vocab_size)
    loss = m.loss(tok, torch.randint(0, cfg.vocab_size, (2, 16)))
    # random logits -> ~ln(V); allow init logit variance
    assert abs(float(loss.detach()) - torch.log(torch.tensor(512.0)).item()) < 1.0


def test_backward_finite_and_learns():
    torch.manual_seed(0)
    cfg = Qwen3Config(**TINY)
    m = Qwen3(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))
    l0 = m.loss(tok, tgt)
    l0.backward()
    gnorm = sum(p.grad.pow(2).sum() for p in m.parameters()).sqrt()
    assert torch.isfinite(gnorm) and float(gnorm) > 0
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        l = m.loss(tok, tgt)
        l.backward()
        opt.step()
    assert float(l) < float(l0.detach()) - 0.5


@pytest.mark.gpu
def test_matches_golden_from_identical_init():
    """Byte-identical init into BOTH the reference and the engine's golden;
    forward mean-CE agreement, then a 5-step training-curve agreement
    (engine-exact AdamW on both sides, fresh iid data per step)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.models.qwen3_reference import GoldenQwen3
    from dataflow.pretrain.bridges import (
        assert_state_dict_byte_identical,
        get_bytes_from_values,
    )
    from dataflow.pretrain.bridges import qwen3 as qwen3_bridge
    from dataflow.pretrain.crosscheck import (
        assert_curves_close,
        golden_curve,
        reference_curve,
        step_data,
    )
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.tasks.base_blocks import AdamWHyper
    from dataflow.training.families import resolve_family
    from dataflow.training.models.qwen3 import ShapedQwen3Config

    LR, WD = 1e-2, 0.1
    cfg = ShapedQwen3Config(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
                            head_dim=32, d_ff=160, vocab_size=512,
                            seq_len=32, batch=2)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=11)
    gb = get_bytes_from_values(values)
    try:
        golden = GoldenQwen3.from_packed_bytes(
            dims, cfg.n_layers, gb("W_embed"),
            [gb(f"W_{i}") for i in range(cfg.n_layers)], gb("W_head"),
            hyper=AdamWHyper(lr=LR, weight_decay=WD))
        model = qwen3_bridge.build_reference_model(cfg, device="cuda")
        qwen3_bridge.load_reference_init(model, cfg, dims, gb)
        assert_state_dict_byte_identical(model, qwen3_bridge.to_qwen3_state_dict(cfg, gb))

        # forward agreement on identical weights
        torch.manual_seed(1)
        tok = torch.randint(0, cfg.vocab_size, (dims.tokens,), device="cuda")
        tgt = torch.randint(0, cfg.vocab_size, (dims.tokens,), device="cuda")
        g_loss = float(golden.loss(tok, tgt).detach())
        r_loss = float(model.loss(tok.view(cfg.batch, cfg.seq_len),
                                  tgt.view(cfg.batch, cfg.seq_len)).detach())
        assert abs(g_loss - r_loss) < 0.03, f"golden {g_loss} vs reference {r_loss}"

        # short training-curve agreement (fresh iid data per step)
        data = step_data(cfg.vocab_size, dims.tokens, steps=5, seed=3)
        gc = golden_curve(golden, data)
        rc = reference_curve(model, data, cfg.batch, cfg.seq_len,
                             lr=LR, weight_decay=WD)
        assert_curves_close(gc, rc, atol=0.05)
    finally:
        for buf in values.values():
            backend.free(buf)
