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


def assert_curves_close(golden_losses, reference_losses, atol: float) -> None:
    deltas = [abs(a - b) for a, b in zip(golden_losses, reference_losses)]
    assert max(deltas) < atol, (
        f"training-curve divergence (worst |d|={max(deltas):.4f} at step "
        f"{deltas.index(max(deltas))}): golden={golden_losses} "
        f"reference={reference_losses}")
