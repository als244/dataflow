"""InitPolicy: per-(layer, field) parameter initialization as DATA.

Follows the established policy idiom (DTypePolicy / OptPolicy): a
default rule + ordered fnmatch overrides + layer routing, where each
rule is a NAME plus args resolved through a registry of init callables
— names ride configs and the wire safely; callables never do. Every
callable receives ``(n, gen)`` (flat element count, the shared CPU
torch.Generator) and returns a flat fp32 tensor; the fill casts to the
field's storage dtype. Draw ORDER stays the sequential field order —
the determinism discipline the byte-identity gates pin.

The DEFAULT policy reproduces the historical init exactly: N(0, 0.02)
draws everywhere, ``*_norm_w`` fields ones. Family ``init_specials``
(A_log schedules etc.) continue to take precedence over the policy —
they are the family's own physics, not a config choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase

import torch

INIT_RULES: dict = {}


def register_init_rule(name: str, build) -> None:
    """Register ``build(**args) -> fill(n, gen)`` under ``name``."""
    INIT_RULES[name] = build


class NormalFill:
    def __init__(self, std: float = 0.02, mean: float = 0.0):
        self.std = std
        self.mean = mean

    def __call__(self, n, gen):
        draw = torch.randn(n, generator=gen) * self.std
        if self.mean:
            draw = draw + self.mean
        return draw


class ConstantFill:
    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def __call__(self, n, gen):
        return torch.full((n,), self.value)


class ScaledNormalFill:
    """GPT-2-style residual scaling and friends: N(0, std/divide_by)."""

    def __init__(self, std: float = 0.02, divide_by: float = 1.0):
        self.std = std
        self.divide_by = divide_by

    def __call__(self, n, gen):
        return torch.randn(n, generator=gen) * (self.std / self.divide_by)


register_init_rule("normal", NormalFill)
register_init_rule("constant", ConstantFill)
register_init_rule("scaled_normal", ScaledNormalFill)


@dataclass(frozen=True)
class InitRule:
    name: str
    args: dict = field(default_factory=dict)

    def build(self):
        maker = INIT_RULES.get(self.name)
        if maker is None:
            raise KeyError(
                f"unknown init rule {self.name!r} (registered: "
                f"{sorted(INIT_RULES)})")
        return maker(**self.args)


def default_normal_rule() -> "InitRule":
    return InitRule("normal", {"std": 0.02})


@dataclass(frozen=True)
class InitPolicy:
    """default rule + FIRST-match fnmatch overrides + layer routing
    (a layer listed in layer_overrides is answered ENTIRELY by its
    sub-policy — same ownership semantics as DTypePolicy)."""

    default: InitRule = field(default_factory=default_normal_rule)
    overrides: tuple = (("*_norm_w", InitRule("constant", {"value": 1.0})),)
    layer_overrides: tuple = ()

    def rule(self, name: str, layer: int | None = None) -> InitRule:
        if layer is not None:
            for layers, sub in self.layer_overrides:
                if layer in layers:
                    return sub.rule(name, None)
        for pattern, r in self.overrides:
            if fnmatchcase(name, pattern):
                return r
        return self.default


DEFAULT_INIT_POLICY = InitPolicy()


def parse_init_rule(d: dict) -> "InitRule":
    return InitRule(d["rule"], dict(d.get("args") or {}))


def build_init_policy(spec: dict | None) -> InitPolicy:
    """Config/wire form -> InitPolicy. ``None`` -> the default policy
    (byte-identical to the historical init). Shape:

        {"default": {"rule": "normal", "args": {"std": 0.02}},
         "overrides": [["*_norm_w", {"rule": "constant",
                                     "args": {"value": 1.0}}], ...],
         "layer_overrides": [[[0, 1], {<sub policy>}], ...]}
    """
    if spec is None:
        return DEFAULT_INIT_POLICY
    default = (parse_init_rule(spec["default"]) if spec.get("default")
               else DEFAULT_INIT_POLICY.default)
    overrides = tuple(
        (pattern, parse_init_rule(rd))
        for pattern, rd in spec.get(
            "overrides",
            [["*_norm_w", {"rule": "constant", "args": {"value": 1.0}}]]))
    layer_overrides = tuple(
        (tuple(int(i) for i in layers), build_init_policy(sub))
        for layers, sub in spec.get("layer_overrides", ()))
    return InitPolicy(default=default, overrides=overrides,
                      layer_overrides=layer_overrides)
