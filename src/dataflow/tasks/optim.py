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
- ``muon``   — slots ("m",); NESTEROV momentum + quintic Newton-Schulz
  (flextrain-aligned coefficients) on rank-2 matrices and rank-3
  expert stacks (batched per-slice NS), rank-scaled; nesterov momentum
  step on anything else. ``hyper.muon_lr`` overrides ``lr`` for muon
  fields.

Assignment is an ``OptPolicy``: fnmatch patterns over the same
namespaced field keys the dtype policy uses ("wq", "head.w",
"embed.w", ...), first match wins, default "adamw". String shorthand
("sgd") means every field — EXCEPT ``"muon"``, which means the HYBRID
RECIPE (``MuonRecipePolicy``): muon for matrix weights, adamw for
embeddings/head/norms/routers/indexer/1D params — because that split
is the only configuration muon is meant to run in. Raw
muon-on-everything stays available as ``OptPolicy(default="muon")``. The per-field ``update_specials`` mechanism
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


def _ns_orthogonalize_batched(m: torch.Tensor) -> torch.Tensor:
    """Approximate UV^T per slice of a (B, r, c) stack via quintic
    Newton-Schulz (fp32; deterministic; transpose trick keeps iterates
    wide; per-slice Frobenius normalization)."""
    x = m.float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1))
    for _ in range(_NS_ITERS):
        a = x @ x.mT
        b = _NS_B * a + _NS_C * a @ a
        x = _NS_A * x + b @ x
    return (x.mT if transposed else x).to(m.dtype)


def _ns_orthogonalize(m: torch.Tensor) -> torch.Tensor:
    """2D convenience wrapper over the batched form."""
    return _ns_orthogonalize_batched(m.unsqueeze(0)).squeeze(0)


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
    """Nesterov momentum + Newton-Schulz (flextrain-aligned: same NS5
    coefficients, nesterov update, per-expert-slice NS on 3D stacks —
    batched here). ``hp.muon_lr`` overrides ``hp.lr`` when set (the two
    rules want very different learning rates)."""
    lr = hp.muon_lr if getattr(hp, "muon_lr", None) else hp.lr
    m = states["m"]
    m32 = m.float().mul_(hp.momentum).add_(g.float())
    m.copy_(m32.to(m.dtype))
    eff = g.float() + hp.momentum * m32          # nesterov
    w32 = w.float()
    w32.mul_(1.0 - lr * hp.weight_decay)
    if len(shape) in (2, 3) and min(shape[-2:]) > 1:
        eff3 = eff.view(shape if len(shape) == 3 else (1, *shape))
        o = _ns_orthogonalize_batched(eff3)
        scale = max(1.0, shape[-2] / shape[-1]) ** 0.5
        w32.add_(o.reshape(-1).float(), alpha=-lr * scale)
    else:
        w32.add_(eff, alpha=-lr)
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

    def for_field(self, key: str, layer: int | None = None,
                  shape: tuple | None = None) -> str:
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


# name fragments that always take adamw under the muon recipe,
# regardless of rank (flextrain's hybrid classification, plus our
# DSA indexer fields — trained conservatively)
_RECIPE_ADAMW_FRAGMENTS = ("norm", "embed", "head", "router", "idx")


@dataclass(frozen=True)
class MuonRecipePolicy:
    """THE meaning of ``opt_policy="muon"``: the standard deployment
    split (flextrain's hybrid rules) —

    - muon for structurally-matrix weights: rank-2 projections and
      rank-3 stacked expert weights (Newton-Schulz per expert slice);
    - adamw for everything else: embeddings, the LM head, norms/gains,
      routers, indexer fields, and every 1D parameter.

    ``overrides`` (fnmatch pattern -> optimizer name) win over the
    rules, so exceptions stay one line. For raw muon-on-everything use
    ``OptPolicy(default="muon")`` explicitly.
    """

    overrides: tuple = ()

    def for_field(self, key: str, layer: int | None = None,
                  shape: tuple | None = None) -> str:
        for pat, name in self.overrides:
            if fnmatch(key, pat):
                return name
        if shape is not None and len(shape) not in (2, 3):
            return "adamw"
        low = key.lower()
        if any(fr in low for fr in _RECIPE_ADAMW_FRAGMENTS):
            return "adamw"
        return "muon"

    def validate(self) -> "MuonRecipePolicy":
        for name in [n for _, n in self.overrides]:
            if name not in OPTIMIZERS:
                raise ValueError(f"unknown optimizer {name!r}")
        return self


MUON_RECIPE = MuonRecipePolicy()


def resolve_opt_policy(p):
    """None | str | policy -> validated policy object.

    Strings: "adamw"/"sgd"/"sgdm" mean that optimizer for EVERY field;
    "muon" means the HYBRID RECIPE (MuonRecipePolicy — muon for matrix
    weights, adamw for embed/head/norm/router/1D), because that is the
    only configuration muon is meant to run in.
    """
    if p is None:
        return OptPolicy()
    if isinstance(p, str):
        if p == "muon":
            return MUON_RECIPE
        return OptPolicy(default=p).validate()
    return p.validate()
