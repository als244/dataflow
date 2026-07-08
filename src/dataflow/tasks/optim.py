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



def _ns_orthogonalize_batched(m: torch.Tensor) -> torch.Tensor:
    """Back-compat alias — the math lives in kernels/muon.py (the
    flextrain port; single source of truth)."""
    from .kernels.muon import ns_orthogonalize_batched

    return ns_orthogonalize_batched(m.float()).to(m.dtype)


def _ns_orthogonalize(m: torch.Tensor) -> torch.Tensor:
    """2D convenience wrapper over the batched form."""
    return _ns_orthogonalize_batched(m.unsqueeze(0)).squeeze(0)


@dataclass(frozen=True)
class LRSchedule:
    """lr(step) as a pure function of the 1-indexed optimizer step —
    deterministic, engine-safe (step rides task.block_params).

    kinds:
    - "constant" (DEFAULT — debugging consistency: identical lr every
      step unless a warmup is declared): linear warmup, then 1.0.
    - "wsd": linear warmup over ``warmup_steps``, stable at 1.0, then
      linear decay over the last ``decay_frac`` of ``total_steps``
      down to ``min_lr_frac`` — the recommended TRAINING schedule.
    - "cosine": linear warmup, then cosine from 1.0 to ``min_lr_frac``
      at ``total_steps``.

    ``total_steps=None`` (the default) additionally DEGENERATES wsd
    and cosine to warmup-then-1.0 — a decaying schedule cannot act
    without knowing the run horizon. ``scale(step)`` multiplies lr AND
    muon_lr, after any per-field hyper overrides.
    """

    kind: str = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    decay_frac: float = 0.1
    min_lr_frac: float = 0.0

    def scale(self, step: int) -> float:
        import math

        if self.warmup_steps and step <= self.warmup_steps:
            return step / self.warmup_steps
        if self.total_steps is None:
            return 1.0
        if self.kind == "constant":
            return 1.0
        if self.kind == "cosine":
            span = max(1, self.total_steps - self.warmup_steps)
            prog = min(1.0, (step - self.warmup_steps) / span)
            lo = self.min_lr_frac
            return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * prog))
        if self.kind == "wsd":
            decay_steps = max(1, int(self.total_steps * self.decay_frac))
            decay_start = self.total_steps - decay_steps
            if step <= decay_start:
                return 1.0
            prog = min(1.0, (step - decay_start) / decay_steps)
            return 1.0 + (self.min_lr_frac - 1.0) * prog
        raise ValueError(f"unknown schedule kind {self.kind!r}")


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
    """Matrix fields (rank 2/3) run the REGISTRY ``muon_step`` kernel —
    the flextrain port (bf16 momentum arithmetic, nesterov, fused NS5,
    Moonshot 0.2*sqrt(max(r,c)) scaling; kernels/muon.py). Non-matrix
    fields (only reachable via raw OptPolicy(default="muon"); the
    recipe routes them to adamw) take a nesterov momentum step here.
    ``hp.muon_lr`` overrides ``hp.lr`` when set."""
    lr = hp.muon_lr if getattr(hp, "muon_lr", None) else hp.lr
    m = states["m"]
    if len(shape) in (2, 3) and min(shape[-2:]) > 1:
        kernels.muon_step(kctx, w, g, m, shape=shape, lr=lr,
                          beta=hp.momentum, eps=hp.eps,
                          weight_decay=hp.weight_decay)
        return
    gm = g.to(m.dtype)
    m.mul_(hp.momentum).add_(gm)
    eff = gm.add(m, alpha=hp.momentum).float()
    w32 = w.float()
    w32.mul_(1.0 - lr * hp.weight_decay)
    w32.add_(eff, alpha=-lr)
    w.copy_(w32.to(w.dtype))


def _frozen_step(kctx, kernels, hp, step_i, w, g, states, shape):
    """Freeze-as-policy: no state, no update. Warm-up phases assign this
    as the default so O layouts collapse to the trainable fields."""


OPTIMIZERS: dict[str, OptimizerDef] = {
    "adamw": OptimizerDef("adamw", ("m", "v"), _adamw_step),
    "sgd": OptimizerDef("sgd", (), _sgd_step),
    "sgdm": OptimizerDef("sgdm", ("m",), _sgdm_step),
    "muon": OptimizerDef("muon", ("m",), _muon_step),
    "frozen": OptimizerDef("frozen", (), _frozen_step),
}


def register_optimizer(d: OptimizerDef) -> None:
    if d.name in OPTIMIZERS:
        raise ValueError(f"optimizer {d.name!r} already registered")
    OPTIMIZERS[d.name] = d


@dataclass(frozen=True)
class OptPolicy:
    """Per-field optimizer assignment.

    ``overrides``: (fnmatch pattern, optimizer name) pairs over
    namespaced field keys; first match wins; ``default`` covers the
    rest. ``layer_overrides`` mirrors DTypePolicy's depth convention
    EXACTLY: ((layers_tuple, sub_policy), ...) — the first layer-set
    containing the layer wins and its SUB-POLICY (an OptPolicy, a
    MuonRecipePolicy, or a string shorthand incl. "muon" = the recipe)
    owns every field decision for that layer; unmatched layers fall
    through to this policy's own rules. So (layer index, param name)
    addressing is:

        OptPolicy(default="adamw", layer_overrides=(
            (tuple(range(4)), "sgd"),                  # layers 0-3
            ((7,), OptPolicy(overrides=(("w?", "muon"),))),
        ))
    """

    default: str = "adamw"
    overrides: tuple = field(default_factory=tuple)
    layer_overrides: tuple = ()
    # (fnmatch pattern, {hyper field: value}) — first match wins; the
    # matched dict REPLACES those fields of the base hyper for that
    # param (e.g. (("*norm*", {"weight_decay": 0.0}),
    #             ("embed.*", {"lr": 1e-5}))). The lr schedule scales
    # lr/muon_lr AFTER these overrides.
    hyper_overrides: tuple = ()

    def for_layer(self, layer: int | None):
        if layer is not None:
            for layers, sub in self.layer_overrides:
                if layer in layers:
                    return resolve_opt_policy(sub)
        return self

    def for_field(self, key: str, layer: int | None = None,
                  shape: tuple | None = None) -> str:
        pol = self.for_layer(layer)
        if pol is not self:
            return pol.for_field(key, None, shape)
        for pat, name in self.overrides:
            if fnmatch(key, pat):
                return name
        return self.default

    def validate(self) -> "OptPolicy":
        for name in [self.default] + [n for _, n in self.overrides]:
            if name not in OPTIMIZERS:
                raise ValueError(f"unknown optimizer {name!r} "
                                 f"(have: {sorted(OPTIMIZERS)})")
        for _, sub in self.layer_overrides:
            resolve_opt_policy(sub)
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
    layer_overrides: tuple = ()   # same depth convention as OptPolicy
    hyper_overrides: tuple = ()   # same semantics as OptPolicy

    def for_layer(self, layer: int | None):
        if layer is not None:
            for layers, sub in self.layer_overrides:
                if layer in layers:
                    return resolve_opt_policy(sub)
        return self

    def for_field(self, key: str, layer: int | None = None,
                  shape: tuple | None = None) -> str:
        pol = self.for_layer(layer)
        if pol is not self:
            return pol.for_field(key, None, shape)
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
        for _, sub in self.layer_overrides:
            resolve_opt_policy(sub)
        return self


MUON_RECIPE = MuonRecipePolicy()


def hyper_for(policy, key: str, layer: int | None, base):
    """Per-(layer, field) effective hyper: route through
    layer_overrides to the owning sub-policy, apply its first-matching
    hyper_overrides dict via dataclasses.replace. Schedule scaling is
    applied by the caller AFTER this."""
    from dataclasses import replace as _replace

    pol = policy.for_layer(layer) if hasattr(policy, "for_layer") else policy
    for pat, over in getattr(pol, "hyper_overrides", ()):
        if fnmatch(key, pat):
            return _replace(base, **over)
    return base


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


def reference_field_step(dims, hyper, *, ns, layer, name, w, state,
                         step, grad_dtype, opt_dtype) -> None:
    """GOLDEN-side per-field optimizer step — mirrors AdamWStep.launch's
    policy dispatch exactly: the same ns-prefixed policy key (that
    namespacing is what routes embed/head to adamw under the muon
    recipe), the same hyper_for overrides, the same schedule scaling.

    ``state`` is the field's slot dict; slots are created lazily at the
    opt dtype per the resolved rule. adamw runs the eager reference
    (ops.adamw_step) to keep the golden's established bit behavior;
    every other rule runs the SAME step fn the engine dispatches
    (muon's matrix path is the registry aten kernel — the only impl)."""
    from dataclasses import replace as _rep

    from . import ops as _ops
    from .kernels import resolve_kernels as _rk

    key = f"{ns}.{name}" if ns else name
    pol = resolve_opt_policy(getattr(dims, "opt_policy", None))
    rule = pol.for_field(key, layer, tuple(w.shape))
    if rule == "frozen":
        return
    hp = hyper_for(pol, key, layer, hyper)
    sched = getattr(hyper, "schedule", None)
    if sched is not None:
        s = sched.scale(step)
        if s != 1.0:
            hp = _rep(hp, lr=hp.lr * s,
                      muon_lr=(hp.muon_lr * s if hp.muon_lr else hp.muon_lr))
    g = w.grad.to(grad_dtype)
    for slot in OPTIMIZERS[rule].slots:
        if slot not in state:
            state[slot] = torch.zeros_like(w, dtype=opt_dtype)
    with torch.no_grad():
        if rule == "adamw":
            _ops.adamw_step(
                w.data.view(-1), g.view(-1),
                state["m"].view(-1), state["v"].view(-1),
                lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                weight_decay=hp.weight_decay, step=step,
            )
            return
        states = {slot: state[slot].view(-1)
                  for slot in OPTIMIZERS[rule].slots}
        OPTIMIZERS[rule].step(None, _rk(), hp, step,
                              w.data.view(-1), g.view(-1), states,
                              tuple(w.shape))


def freeze(base=None, *, fields: tuple = (), layers=(), pairs: tuple = (),
           embed: bool = False, head: bool = False):
    """Compose a freeze SPEC over a base optimizer policy — the
    ergonomic front door for ``opt_policy``. Returns an OptPolicy whose
    frozen fields resolve to the "frozen" rule and whose trainable
    fields resolve through ``base`` (default adamw; "muon" for the
    recipe; any policy object works).

    Examples:
        freeze(layers=range(0, 16))                  # bottom 16 layers
        freeze(fields=("wq",))                       # one field, fleet-wide
        freeze(pairs=(("wo", 3), ("w1", 7)))         # targeted (field, layer)
        freeze(base="muon", layers=range(0, 8))      # muon on what trains
        freeze(embed=True, layers=range(0, 4))       # embedding + prefix

    ``embed``/``head`` freeze the embedding/head weights (their fields
    resolve under the "embed."/"head." key namespace, exactly as the
    optimizer executes them)."""
    base_pol = resolve_opt_policy(base)
    field_overrides = tuple((f, "frozen") for f in fields)
    if embed:
        field_overrides += (("embed.*", "frozen"),)
    if head:
        field_overrides += (("head.*", "frozen"),)
    layer_set = tuple(sorted(set(layers)))
    pair_overrides: dict[int, tuple] = {}
    for f, layer in pairs:
        pair_overrides.setdefault(layer, ())
        pair_overrides[layer] += ((f, "frozen"),)

    layer_rules = []
    if layer_set:
        layer_rules.append((layer_set, OptPolicy(default="frozen")))
    for layer, ovr in sorted(pair_overrides.items()):
        layer_rules.append(((layer,),
                            _Composed(base_pol, ovr)))
    if not layer_rules and not field_overrides:
        return base_pol
    return _Composed(base_pol, field_overrides, tuple(layer_rules))


class _Composed:
    """A freeze layer over an arbitrary base policy: frozen overrides
    are consulted first (fnmatch + per-layer), everything else defers
    to the base. Duck-types the policy protocol (for_field /
    validate)."""

    def __init__(self, base, overrides=(), layer_rules=()):
        self.base = base
        self.overrides = tuple(overrides)
        self.layer_rules = tuple(layer_rules)

    def for_field(self, key: str, layer=None, shape=None) -> str:
        # field-level freezes are layer-independent: they apply even
        # inside a layer-rule match
        for pat, name in self.overrides:
            if fnmatch(key, pat):
                return name
        if layer is not None:
            for layers_, sub in self.layer_rules:
                if layer in layers_:
                    return sub.for_field(key, None, shape) \
                        if not isinstance(sub, OptPolicy) \
                        else sub.for_field(key, layer, shape)
        return self.base.for_field(key, layer, shape)

    def validate(self):
        return self

    def __repr__(self):
        return (f"freeze(base={self.base!r}, overrides={self.overrides}, "
                f"layers={tuple(ls for ls, _ in self.layer_rules)})")
