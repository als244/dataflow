"""Reference muon (the yardstick side): the hybrid classification
must mirror the engine recipe, the Newton-Schulz core must actually
orthogonalize, and a tiny end-to-end reference run must train."""
import pytest

torch = pytest.importorskip("torch")

from dataflow.pretrain.driver import (  # noqa: E402
    MUON_ADAMW_FRAGMENTS,
    ReferenceMuon,
    ns_orthogonalize,
    reference_optimizer,
)
from dataflow.pretrain.recipe import Recipe  # noqa: E402


def test_fragments_mirror_engine_recipe():
    from dataflow.tasks import optim

    engine = getattr(optim, "_RECIPE_ADAMW_FRAGMENTS")
    assert tuple(MUON_ADAMW_FRAGMENTS) == tuple(engine)


def test_ns_orthogonalize_matches_engine_and_conditions():
    from dataflow.tasks.kernels.muon import ns_orthogonalize_batched

    g = torch.Generator().manual_seed(3)
    for r, c in ((64, 256), (256, 64), (128, 128)):
        x = torch.randn(1, r, c, generator=g)
        o = ns_orthogonalize(x.clone())
        # the reference twin must agree with the engine kernel's NS
        # (same constants/order; fp rounding only)
        eng = ns_orthogonalize_batched(x.clone())
        assert torch.allclose(o, eng, atol=1e-4, rtol=1e-4), (r, c)
        # 5 quintic iterations orthogonalize APPROXIMATELY (that is
        # muon): singular values pulled into a tight band around 1
        # from an unconditioned random matrix
        s = torch.linalg.svdvals(o[0])
        assert float(s.min()) > 0.3 and float(s.max()) < 1.5, (
            s.min(), s.max())
        assert float(s.max() / s.min()) < 5.0, (s.min(), s.max())
        # tall inputs must round-trip the transpose (shape preserved)
        assert o.shape == x.shape


def test_classification_on_reference_llama3():
    from reference_models.llama3 import Llama3Config, Llama3

    cfg = Llama3Config(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
                       d_ff=128, vocab_size=256)
    model = Llama3(cfg).bfloat16()
    opt = ReferenceMuon(model, Recipe(total_steps=10))
    muon_names = {n for n, p in model.named_parameters()
                  if any(p is q for q in opt.muon_params)}
    adamw_names = {n for n, p in model.named_parameters()
                   if any(p is q for q in opt.adamw_params)}
    # seven rank-2 projections per layer take muon
    for i in range(cfg.n_layers):
        for f in ("wq", "wk", "wv", "wo"):
            assert f"blocks.{i}.attn.{f}.weight" in muon_names
        for f in ("w1", "w3", "w2"):
            assert f"blocks.{i}.mlp.{f}.weight" in muon_names
    # tables and every norm gain take adamw
    assert "embed.weight" in adamw_names
    assert "lm_head.weight" in adamw_names
    assert all("norm" in n for n in adamw_names
               if n not in ("embed.weight", "lm_head.weight"))
    assert len(muon_names) == 7 * cfg.n_layers
    assert not (muon_names & adamw_names)


def test_reference_optimizer_dispatch():
    from dataclasses import replace

    from reference_models.llama3 import Llama3Config, Llama3
    from dataflow.training.models.llama3 import ShapedLlamaConfig

    rcfg = Llama3Config(n_layers=1, d_model=64, n_heads=2, n_kv_heads=2,
                        d_ff=128, vocab_size=256)
    model = Llama3(rcfg).bfloat16()
    cfg = ShapedLlamaConfig.tiny()
    r = Recipe(total_steps=10)
    assert type(reference_optimizer(model, cfg, r)).__name__ \
        == "ReferenceAdamW"
    muon_cfg = replace(cfg, opt_policy="muon")
    assert isinstance(reference_optimizer(model, muon_cfg, r),
                      ReferenceMuon)
    bad_cfg = replace(cfg, opt_policy="sgd")
    with pytest.raises(ValueError, match="no reference optimizer"):
        reference_optimizer(model, bad_cfg, r)


@pytest.mark.gpu
def test_tiny_muon_reference_trains():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from reference_models.llama3 import Llama3Config, Llama3

    cfg = Llama3Config(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
                       d_ff=128, vocab_size=256)
    model = Llama3(cfg).bfloat16().cuda().train()
    opt = ReferenceMuon(model, Recipe(peak_lr=1e-3, min_lr=1e-4,
                                      warmup_steps=1, total_steps=8))
    g = torch.Generator().manual_seed(5)
    tok = torch.randint(0, cfg.vocab_size, (4, 8, 32), generator=g).cuda()
    losses = []
    for step in range(8):
        opt.zero_grad()
        loss = model.loss(tok[step % 4], tok[(step + 1) % 4])
        loss.backward()
        opt.step(step)
        losses.append(float(loss))
        assert loss == loss, f"NaN at step {step}"     # NaN guard
    assert losses[-1] < losses[0], losses              # it learns
