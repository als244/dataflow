"""Program-resolver registry: THE workload<->engine seam.

The engine executes programs; a program's tasks resolve through a
REGISTERED KIND. Registration carries an opaque ``resolver_spec`` dict
of which the engine reads exactly one key — ``"kind"`` — and hands the
whole spec to the registered build:

    build(resolver_spec) -> Resolver
    Resolver(task: TaskSpec) -> Executable        # every task resolves

Workloads register at import time (the daemon default-loads
``dataflow_training.register.register_all``; ``--plugin`` modules may
register more). Unknown kinds fail loudly, naming what IS registered.
The engine never learns family/model vocabulary — that all lives
behind the build callables.
"""
from __future__ import annotations

import json

from .wire import ServiceError

_BUILDS: dict = {}
_CACHE: dict = {}


def register_program_resolver(kind: str, build) -> None:
    """Register ``build`` for ``resolver_spec["kind"] == kind``.
    Re-registering the same kind replaces the build (plugin reload);
    the spec cache for that kind is dropped with it."""
    _BUILDS[kind] = build
    for key in [k for k in _CACHE if k[0] == kind]:
        del _CACHE[key]


def registered_kinds() -> list[str]:
    return sorted(_BUILDS)


class JitterExecutable:
    """Fire a random spin-kernel delay before the wrapped task launches, so
    tasks finish in scrambled relative orders."""

    def __init__(self, inner, kernel, low_us, high_us, rng):
        self.inner = inner
        self.kernel = kernel
        self.low_us = low_us
        self.high_us = high_us
        self.rng = rng

    def launch(self, ctx):
        self.kernel.launch_us(ctx.stream, self.rng.uniform(self.low_us,
                                                           self.high_us))
        self.inner.launch(ctx)


class JitterResolver:
    """Wrap ANY resolver so every task's launch is preceded by a random spin
    delay — a general resolver DEBUG option.

    The point is to make ACTUAL task runtimes diverge wildly from the plan's
    cost ESTIMATES (which drive the engine's memory scheduling — offload /
    prefetch / evict). The engine sequences real execution on completion
    EVENTS, so its results must be invariant to that divergence: correctness
    never depends on the estimates being accurate (they only affect
    performance). A test turns this on and asserts identical output — proving
    the run is estimate-independent, not that any particular ordering holds.

    Kind-agnostic: enabled by a top-level ``debug_jitter`` key on the resolver
    spec ({"min_us", "max_us", "seed"}), so any workload or test turns it on
    with no engine or program change."""

    def __init__(self, inner, min_us, max_us, seed):
        import random

        from dataflow.runtime.device.cuda_spin import SpinKernel

        self.inner = inner
        self.kernel = SpinKernel()
        self.low_us = float(min_us)
        self.high_us = float(max_us)
        self.rng = random.Random(seed)

    def __call__(self, task):
        return JitterExecutable(self.inner(task), self.kernel, self.low_us,
                                self.high_us, self.rng)


def lookup_resolver(spec: dict):
    """Resolve a registered kind's build over ``spec`` (cached by the
    spec's canonical JSON — builds are pure functions of their spec). A
    top-level ``debug_jitter`` key wraps the result in a JitterResolver,
    kind-agnostically (the build never sees it)."""
    kind = spec.get("kind")
    if kind is None:
        raise ServiceError(
            "BAD_REQUEST",
            f"resolver_spec has no 'kind' (registered kinds: "
            f"{registered_kinds() or 'NONE'})")
    build = _BUILDS.get(kind)
    if build is None:
        raise ServiceError(
            "BAD_REQUEST",
            f"unknown resolver kind {kind!r} (registered kinds: "
            f"{registered_kinds() or 'NONE'} — did the daemon load its "
            f"workload registrations?)")
    key = (kind, json.dumps(spec, sort_keys=True))
    hit = _CACHE.get(key)
    if hit is None:
        hit = build(spec)
        jitter = spec.get("debug_jitter")
        if jitter:
            hit = JitterResolver(hit, jitter.get("min_us", 20),
                                 jitter.get("max_us", 400),
                                 jitter.get("seed", 0))
        _CACHE[key] = hit
    return hit
