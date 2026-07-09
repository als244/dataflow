"""Reference-vs-golden training-curve agreement (the per-family cross-check
harness).

Both witnesses start from the SAME engine packed init bytes and consume the
SAME seeded token stream; each runs an optimizer that replicates the engine's
AdamW exactly (the golden's ``train_step`` per-field policy dispatch; the
reference under the driver's ``ReferenceAdamW``), so their loss curves must
track within bf16 kernel-order noise of two independent forward
implementations. A divergence indicts ONE of the two implementations — the
golden is a second witness, not automatic truth; investigate both directions
(the per-family engine-service parity smoke closes the loop against the real
engine).

Family-neutral at LBL-off: every reference exposes ``loss(tokens, targets)``
whose default is pure mean CE (the MoE families' ``aux_coef`` defaults to 0).
"""
from __future__ import annotations

import torch

from .driver import ReferenceAdamW
from .recipe import Recipe


def flat_recipe(lr: float, weight_decay: float, steps: int) -> Recipe:
    """Constant-LR recipe: warmup 0 and min == peak collapse the cosine, so
    ``lr_at(step) == lr`` for every step — matching a schedule-less
    ``AdamWHyper(lr=lr)`` on the golden side."""
    return Recipe(peak_lr=lr, min_lr=lr, warmup_steps=0,
                  total_steps=max(steps, 1), weight_decay=weight_decay)


def step_data(vocab: int, tokens: int, steps: int, seed: int,
              device: str = "cuda") -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-step ``(tokens, targets)`` — fresh iid rows each step from one
    seeded stream, shared verbatim by both witnesses."""
    g = torch.Generator().manual_seed(seed)
    data = []
    for _ in range(steps):
        tok = torch.randint(0, vocab, (tokens,), generator=g)
        tgt = torch.randint(0, vocab, (tokens,), generator=g)
        data.append((tok.to(device), tgt.to(device)))
    return data


def golden_curve(golden, data) -> list[float]:
    """Per-step CE from the golden's ``train_step`` (loss is computed BEFORE
    the optimizer update, like the reference curve below)."""
    return [golden.train_step(tok, tgt) for tok, tgt in data]


def reference_curve(model, data, batch: int, seq_len: int, *,
                    lr: float, weight_decay: float) -> list[float]:
    """Per-step CE training the reference nn.Module under the driver's
    engine-mirroring ``ReferenceAdamW`` at a constant LR."""
    opt = ReferenceAdamW(model.parameters(), flat_recipe(lr, weight_decay, len(data)))
    losses = []
    for k, (tok, tgt) in enumerate(data):
        opt.zero_grad()
        loss = model.loss(tok.view(batch, seq_len), tgt.view(batch, seq_len))
        loss.backward()
        losses.append(float(loss.detach()))
        opt.step(k)
    return losses


def reference_curve_moe(model, data, batch: int, seq_len: int, *,
                        lr: float, weight_decay: float,
                        aux_coef: float) -> tuple[list[float], list[float]]:
    """(CE curve, LBL-term curve) for a MoE reference. The training objective
    is CE + aux_coef * LBL — matching the goldens' gradient-injected form —
    but the logged scalar is the CE CHANNEL: the pinned reporting convention
    (the engine never folds the LBL term into the ``loss_*`` objects). The
    second list is the per-step LBL term, ALPHA-FREE
    (``model.load_balance_loss()`` from the step's forward)."""
    opt = ReferenceAdamW(model.parameters(), flat_recipe(lr, weight_decay, len(data)))
    ces, lbls = [], []
    for k, (tok, tgt) in enumerate(data):
        opt.zero_grad()
        composite = model.loss(tok.view(batch, seq_len), tgt.view(batch, seq_len),
                               aux_coef=aux_coef)
        lbl = float(model.load_balance_loss().detach())
        composite.backward()
        ces.append(float(composite.detach()) - aux_coef * lbl)
        lbls.append(lbl)
        opt.step(k)
    return ces, lbls


def assert_curves_close(golden_losses, reference_losses, atol: float) -> None:
    deltas = [abs(a - b) for a, b in zip(golden_losses, reference_losses)]
    assert max(deltas) < atol, (
        f"training-curve divergence (worst |d|={max(deltas):.4f} at step "
        f"{deltas.index(max(deltas))}): golden={golden_losses} "
        f"reference={reference_losses}")


def moe_golden_gate(cfg_off, golden_cls, bridge_mod, *, alpha: float,
                    lr: float = 1e-2, weight_decay: float = 0.1,
                    steps: int = 5, seed: int = 3, fwd_atol: float = 0.03,
                    curve_atol: float = 0.05, aux_atol: float = 0.005) -> None:
    """The full MoE reference-vs-golden gate, shared by the MoE family tests.

    LBL-OFF leg (``cfg_off`` must carry ``aux_coef=0`` — no load-balance
    functions on either side): state_dict byte-identity, forward CE
    agreement, multi-step curve agreement. LBL-ON leg (fresh witnesses from
    the SAME packed bytes, ``aux_coef=alpha``): the scalar-loss convention is
    pinned — the REPORTED scalar is pure CE on both sides (golden train_step
    returns CE while differentiating CE+aux; the reference logs
    composite − alpha·LBL) — the alpha-scaled LBL terms agree, and the
    CE-channel training curves agree.
    """
    from dataclasses import replace

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.tasks.base_blocks import AdamWHyper
    from dataflow.training.families import resolve_family

    from .bridges import assert_state_dict_byte_identical, get_bytes_from_values

    assert cfg_off.aux_coef == 0.0, "cfg_off is the LBL-OFF leg"
    fam = resolve_family(cfg_off)
    dims = fam.dims_of(cfg_off)
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg_off), cfg_off, backend, seed=11)
    gb = get_bytes_from_values(values)
    B, T = dims.tokens // dims.seq_len, dims.seq_len
    hyper = AdamWHyper(lr=lr, weight_decay=weight_decay)
    try:
        data = step_data(cfg_off.vocab_size, dims.tokens, steps=steps, seed=seed)

        # -- LBL-OFF leg -------------------------------------------------------
        golden = golden_cls.from_packed_bytes(
            dims, cfg_off.n_layers, gb("W_embed"),
            [gb(f"W_{i}") for i in range(cfg_off.n_layers)], gb("W_head"),
            hyper=hyper)
        model = bridge_mod.build_reference_model(cfg_off, device="cuda")
        bridge_mod.load_reference_init(model, cfg_off, dims, gb)
        assert_state_dict_byte_identical(
            model, bridge_mod.to_reference_state_dict(cfg_off, gb))
        tok, tgt = data[0]
        g_loss = float(golden.loss(tok, tgt).detach())
        r_loss = float(model.loss(tok.view(B, T), tgt.view(B, T)).detach())
        assert abs(g_loss - r_loss) < fwd_atol, (
            f"fwd: golden {g_loss} vs reference {r_loss}")
        assert_curves_close(
            golden_curve(golden, data),
            reference_curve(model, data, B, T, lr=lr, weight_decay=weight_decay),
            atol=curve_atol)

        # -- LBL-ON leg (fresh witnesses from the same bytes) -------------------
        cfg_on = replace(cfg_off, aux_coef=alpha)
        dims_on = resolve_family(cfg_on).dims_of(cfg_on)
        golden_on = golden_cls.from_packed_bytes(
            dims_on, cfg_on.n_layers, gb("W_embed"),
            [gb(f"W_{i}") for i in range(cfg_on.n_layers)], gb("W_head"),
            hyper=hyper)
        model_on = bridge_mod.build_reference_model(cfg_on, device="cuda")
        bridge_mod.load_reference_init(model_on, cfg_on, dims_on, gb)
        ce_g, aux_g = golden_on.loss_terms(tok, tgt)
        composite = model_on.loss(tok.view(B, T), tgt.view(B, T), aux_coef=alpha)
        lbl_r = float(model_on.load_balance_loss().detach())
        ce_r = float(composite.detach()) - alpha * lbl_r
        assert abs(float(ce_g.detach()) - ce_r) < fwd_atol, (
            f"CE channel: golden {float(ce_g.detach())} vs reference {ce_r}")
        assert float(aux_g.detach()) > 0.0
        assert abs(float(aux_g.detach()) - alpha * lbl_r) < aux_atol, (
            f"LBL term: golden {float(aux_g.detach())} vs "
            f"alpha*reference {alpha * lbl_r}")
        ces_r, lbls_r = reference_curve_moe(
            model_on, data, B, T, lr=lr, weight_decay=weight_decay, aux_coef=alpha)
        assert_curves_close(golden_curve(golden_on, data), ces_r, atol=curve_atol)
        assert all(x > 0.0 for x in lbls_r)
    finally:
        for buf in values.values():
            backend.free(buf)
