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


def lookup_resolver(spec: dict):
    """Resolve a registered kind's build over ``spec`` (cached by the
    spec's canonical JSON — builds are pure functions of their spec)."""
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
        _CACHE[key] = hit
    return hit
