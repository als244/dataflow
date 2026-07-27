"""Shared, process-lifetime CUDA stream trios.

Creating a stream is a MEMORY-LIFECYCLE decision, not just a scheduling
one: torch's caching allocator tags every cached block with the stream
it ran on, and a freed block is reusable ONLY on that stream. Work that
mints a fresh stream per unit (per program, per cost table) strands
each unit's cached kernel scratch on a dead stream — reserved memory
then grows monotonically with the number of units, invisibly on a
large card and fatally on a small one, and raw cudaMalloc callers
(which cannot see torch's cached-idle blocks) starve first.

The rule: long-lived work SHARES a stream trio per scope, created once
per process through this module. Scopes isolate workloads that must
not interleave on one compute stream (each in-process daemon keys by
its store; the profiler keys by backend) — they share the discipline,
not the streams. Create a private stream only when you own its whole
lifecycle and either free its cache or accept the retention.
"""
from __future__ import annotations

SHARED_STREAMS: dict = {}


def shared_streams(backend, scope) -> tuple:
    """The (compute, h2d, d2h) trio for ``scope``, created on first use
    and reused for the life of the process. ``scope`` is any hashable
    identity the caller isolates by (a daemon's store id, a profiler's
    backend id)."""
    key = (scope, id(backend))
    if key not in SHARED_STREAMS:
        SHARED_STREAMS[key] = (backend.create_stream("compute"),
                               backend.create_stream("h2d"),
                               backend.create_stream("d2h"))
    return SHARED_STREAMS[key]
