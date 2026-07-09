"""Gates for the independent reference model: fwd/bwd smoke, a tiny model
learns, and — the load-bearing one — it AGREES with the engine's golden
llama3 from a byte-identical init (validates the model + the weight bridge)."""
import pytest
import torch

from reference_models import Llama3, Llama3Config


def test_forward_shapes_and_init_loss():
    torch.manual_seed(0)
    cfg = Llama3Config(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
                       d_ff=160, vocab_size=512)
    m = Llama3(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = m(tok)
    assert logits.shape == (2, 16, cfg.vocab_size)
    loss = m.loss(tok, torch.randint(0, cfg.vocab_size, (2, 16)))
    # random logits -> ~ln(V); allow init logit variance
    assert abs(float(loss.detach()) - torch.log(torch.tensor(512.0)).item()) < 1.0


def test_backward_finite_and_learns():
    torch.manual_seed(0)
    cfg = Llama3Config(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
                       d_ff=160, vocab_size=512)
    m = Llama3(cfg)
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
    the forward mean-CE must agree to within bf16 kernel-order noise. This
    is the ground-truth cross-check + the weight-bridge gate."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow.models.llama3_reference import GoldenLlama3
    from dataflow.pretrain import bridge
    from dataflow.training.families import resolve_family
    from dataflow.training.models.llama3 import ShapedLlamaConfig

    cfg = ShapedLlamaConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
                            d_ff=160, vocab_size=512, seq_len=32, batch=2)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    from dataflow.runtime.device.cuda import CudaBackend
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=11)
    gb = bridge.get_bytes_from_values(values)

    # golden from the same packed bytes
    golden = GoldenLlama3.from_packed_bytes(
        dims, cfg.n_layers, gb("W_embed"),
        [gb(f"W_{i}") for i in range(cfg.n_layers)], gb("W_head"))

    # reference from the same packed bytes, + byte-identity gate
    model = bridge.build_reference(cfg, device="cuda")
    bridge.load_engine_init(model, dims, cfg.n_layers, gb)
    bridge.assert_byte_identical(model, dims, cfg.n_layers, gb)

    torch.manual_seed(1)
    tok = torch.randint(0, cfg.vocab_size, (dims.tokens,), device="cuda")
    tgt = torch.randint(0, cfg.vocab_size, (dims.tokens,), device="cuda")
    g_loss = float(golden.loss(tok, tgt))
    B, T = dims.tokens // dims.seq_len, dims.seq_len
    r_loss = float(model.loss(tok.view(B, T), tgt.view(B, T)))
    assert abs(g_loss - r_loss) < 0.03, f"golden {g_loss} vs reference {r_loss}"

    for buf in values.values():
        backend.free(buf)
