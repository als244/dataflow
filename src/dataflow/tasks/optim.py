"""Optimizer abstraction: per-FIELD optimizer choice, per-optimizer
state slots and step rule.

The optimizer executable (``llama3_blocks.OptimizerStep``, shared by
every family) and the O-object sizing (``layouts.opt_state_layout``)
both dispatch through this registry, so a family — builtin or external
— configures optimizers per parameter FIELD (the finest-grained unit:
one entry of a packed weight layout) without touching either.

An optimizer is (state slots, step rule):

- ``adamw``  — slots ("m", "v"); the historical default, math delegated
  to the registry ``adamw_step`` kernel (bit-identical to before this
  abstraction existed).
- ``sgd``    — no slots; decoupled weight decay.
- ``sgdm``   — slots ("m",); heavy-ball momentum, decoupled decay.
- ``muon``   — slots ("m",); momentum + Newton-Schulz orthogonalization
  for 2D matrix fields (rank-scaled), plain momentum step for 1D
  fields (the standard Muon convention: embeddings/gains/biases keep a
  momentum rule; real deployments usually route them to adamw via the
  policy instead).

Assignment is an ``OptPolicy``: fnmatch patterns over the same
namespaced field keys the dtype policy uses ("wq", "head.w",
"embed.w", ...), first match wins, default "adamw". String shorthand
("sgd") means every field. The per-field ``update_specials`` mechanism
(noaux router bias, frozen fields) stays the HIGHEST-priority override
on top of the policy.

All step math runs in fp32 and round-trips through the field's storage
dtypes (weights) / the dtype policy's opt dtype (state) — the same
convention the AdamW kernel pins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Callable

import torch

# NS5 coefficients (Jordan et al. Muon; quintic Newton-Schulz)
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315
_NS_ITERS = 5


def _ns_orthogonalize(m: torch.Tensor) -> torch.Tensor:
    """Approximate UV^T of the momentum matrix via quintic
    Newton-Schulz (fp32; deterministic; transpose trick keeps the
    iterate wide)."""
    x = m.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(_NS_ITERS):
        a = x @ x.T
        b = _NS_B * a + _NS_C * a @ a
        x = _NS_A * x + b @ x
    return (x.T if transposed else x).to(m.dtype)


@dataclass(frozen=True)
class OptimizerDef:
    name: str
    slots: tuple[str, ...]
    # step(kctx, kernels, hyper, step_i, w_view, g_view, states, shape)
    step: Callable


def _adamw_step(kctx, kernels, hp, step_i, w, g, states, shape):
    kernels.adamw_step(
        kctx, w, g, states["m"], states["v"],
        lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
        weight_decay=hp.weight_decay, step=step_i,
    )


def _sgd_step(kctx, kernels, hp, step_i, w, g, states, shape):
    w32 = w.float()
    w32.mul_(1.0 - hp.lr * hp.weight_decay)
    w32.add_(g.float(), alpha=-hp.lr)
    w.copy_(w32.to(w.dtype))


def _sgdm_step(kctx, kernels, hp, step_i, w, g, states, shape):
    m = states["m"]
    m32 = m.float().mul_(hp.momentum).add_(g.float())
    m.copy_(m32.to(m.dtype))
    w32 = w.float()
    w32.mul_(1.0 - hp.lr * hp.weight_decay)
    w32.add_(m32, alpha=-hp.lr)
    w.copy_(w32.to(w.dtype))


def _muon_step(kctx, kernels, hp, step_i, w, g, states, shape):
    m = states["m"]
    m32 = m.float().mul_(hp.momentum).add_(g.float())
    m.copy_(m32.to(m.dtype))
    w32 = w.float()
    w32.mul_(1.0 - hp.lr * hp.weight_decay)
    if len(shape) == 2 and min(shape) > 1:
        o = _ns_orthogonalize(m32.view(shape))
        scale = max(1.0, shape[0] / shape[1]) ** 0.5
        w32.add_(o.reshape(-1).float(), alpha=-hp.lr * scale)
    else:
        w32.add_(m32, alpha=-hp.lr)
    w.copy_(w32.to(w.dtype))


OPTIMIZERS: dict[str, OptimizerDef] = {
    "adamw": OptimizerDef("adamw", ("m", "v"), _adamw_step),
    "sgd": OptimizerDef("sgd", (), _sgd_step),
    "sgdm": OptimizerDef("sgdm", ("m",), _sgdm_step),
    "muon": OptimizerDef("muon", ("m",), _muon_step),
}


def register_optimizer(d: OptimizerDef) -> None:
    if d.name in OPTIMIZERS:
        raise ValueError(f"optimizer {d.name!r} already registered")
    OPTIMIZERS[d.name] = d


@dataclass(frozen=True)
class OptPolicy:
    """Per-field optimizer assignment. ``overrides`` are (fnmatch
    pattern, optimizer name) pairs over namespaced field keys; first
    match wins; ``default`` covers the rest."""

    default: str = "adamw"
    overrides: tuple = field(default_factory=tuple)

    def for_field(self, key: str, layer: int | None = None) -> str:
        for pat, name in self.overrides:
            if fnmatch(key, pat):
                return name
        return self.default

    def validate(self) -> "OptPolicy":
        for name in [self.default] + [n for _, n in self.overrides]:
            if name not in OPTIMIZERS:
                raise ValueError(f"unknown optimizer {name!r} "
                                 f"(have: {sorted(OPTIMIZERS)})")
        return self


def resolve_opt_policy(p) -> OptPolicy:
    """None | str | OptPolicy -> OptPolicy (validated)."""
    if p is None:
        return OptPolicy()
    if isinstance(p, str):
        return OptPolicy(default=p).validate()
    return p.validate()
