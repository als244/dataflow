"""Gates for the independent qwen3.5-dense reference: fwd/bwd smoke, a tiny
model learns, and — the load-bearing one — it AGREES with the engine's
GoldenQwen35 from a byte-identical init, for both the untied and tied
embedding configs. Exercises the hybrid stack: Gated-DeltaNet linear layers
(causal conv + delta-rule recurrence + gated RMSNorm) and the gated
full-attention layer (per-head qk-norm + partial RoPE + output gate)."""
from dataclasses import replace

import pytest
import torch

from references.qwen35 import Qwen35, Qwen35Config

_TINY = dict(
    n_layers=4, d_model=256, full_attention_interval=4, n_heads=4,
    n_kv_heads=2, head_dim=64, partial_rotary_factor=0.25, lin_k_heads=2,
    lin_v_heads=4, lin_k_head_dim=32, lin_v_head_dim=32, lin_conv_kernel=4,
    d_ff=512, vocab_size=512,
)


def test_forward_shapes_and_learns():
    torch.manual_seed(0)
    cfg = Qwen35Config(**_TINY)
    assert [cfg.kind_of(i) for i in range(4)] == ["lin", "lin", "lin", "full"]
    m = Qwen35(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 32))
    tgt = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = m(tok)
    assert logits.shape == (2, 32, cfg.vocab_size)
    l0 = m.loss(tok, tgt)
    assert abs(float(l0.detach()) - torch.log(torch.tensor(512.0)).item()) < 1.0
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


def _golden_crosscheck(tied: bool):
    from dataflow.models.qwen35_reference import GoldenQwen35
    from dataflow.pretrain import bridge
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.training.families import resolve_family
    from dataflow.training.models.qwen35 import ShapedQwen35Config

    # batch=2 exercises the per-sequence reset in the conv + delta-rule
    base = ShapedQwen35Config.tiny_tied() if tied else ShapedQwen35Config.tiny()
    cfg = replace(base, seq_len=64, batch=2)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=11)
    gb = bridge.get_bytes_from_values(values)
    try:
        golden = GoldenQwen35.from_packed_bytes(
            dims, cfg.n_layers, gb("W_embed"),
            [gb(f"W_{i}") for i in range(cfg.n_layers)],
            None if tied else gb("W_head"))
        model = bridge.build_qwen35_reference(cfg, device="cuda")
        bridge.load_qwen35_init(model, cfg, gb)
        # byte-identity of the loaded init (state_dict, not named_parameters:
        # the tied config shares one tensor for embed + lm_head, which
        # named_parameters deduplicates away)
        sd = bridge.to_qwen35_state_dict(cfg, gb)
        msd = model.state_dict()
        assert set(sd) == set(msd), set(sd) ^ set(msd)
        for k, v in sd.items():
            assert torch.equal(msd[k].detach().cpu(), v.cpu()), k
        torch.manual_seed(1)
        tok = torch.randint(0, cfg.vocab_size, (dims.tokens,), device="cuda")
        tgt = torch.randint(0, cfg.vocab_size, (dims.tokens,), device="cuda")
        g_loss = float(golden.loss(tok, tgt).detach())
        B, T = dims.tokens // dims.seq_len, dims.seq_len
        r_loss = float(model.loss(tok.view(B, T), tgt.view(B, T)).detach())
        assert abs(g_loss - r_loss) < 0.02, f"golden {g_loss} vs reference {r_loss}"
    finally:
        for buf in values.values():
            backend.free(buf)


@pytest.mark.gpu
def test_matches_golden_untied():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    _golden_crosscheck(tied=False)


@pytest.mark.gpu
def test_matches_golden_tied():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    _golden_crosscheck(tied=True)
