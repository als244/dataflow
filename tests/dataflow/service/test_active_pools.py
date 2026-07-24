"""The active-pools registry: which program pools are still resident.

``active_pools()`` is the leak table behind ``engine_status["program_pools"]`` —
one row per registered program whose engine pool is still alive (get_session
makes it on first run; close_session / unregister_program frees it). It exists so
a pool leak (programs that ran but were never unregistered, each pinning a
placement base) is INVESTIGABLE instead of silent. These are unit tests over the
accessor: they stub the session registry, so no server or device is needed.

Tests:
- test_active_pools_reports_live_pools_scoped_to_a_daemon: one row per resident program pool, fast_extent_bytes from the placement base, scoped to a single daemon's store.
- test_active_pools_shrinks_when_a_pool_is_freed: freeing a session (close_session / unregister_program) drops its row, keeping the list bounded.
"""
import types

from dataflow.service import execution


def _sess(fast_bytes=None, unrecoverable=False):
    """A stand-in engine session: a pool carrying a placement base of
    ``fast_bytes`` (None = ran but no fast residency / not yet pooled)."""
    pool = None
    if fast_bytes is not None:
        pool = types.SimpleNamespace(
            _placement_base=types.SimpleNamespace(size_bytes=fast_bytes))
    return types.SimpleNamespace(pool=pool, unrecoverable=unrecoverable)


def test_active_pools_reports_live_pools_scoped_to_a_daemon():
    store, other = object(), object()
    saved = dict(execution._SESSIONS)
    execution._SESSIONS.clear()
    try:
        execution._SESSIONS[(id(store), "p-ran")] = _sess(fast_bytes=200 << 20)
        execution._SESSIONS[(id(store), "p-nopool")] = _sess(fast_bytes=None)
        execution._SESSIONS[(id(other), "p-otherdaemon")] = _sess(fast_bytes=64 << 20)

        rows = {r["prog_id"]: r for r in execution.active_pools(store=store)}
        assert set(rows) == {"p-ran", "p-nopool"}          # other daemon excluded
        assert rows["p-ran"]["has_pool"]
        assert rows["p-ran"]["fast_extent_bytes"] == 200 << 20
        assert not rows["p-nopool"]["has_pool"]
        assert rows["p-nopool"]["fast_extent_bytes"] == 0

        assert len(execution.active_pools()) == 3          # unscoped spans daemons
    finally:
        execution._SESSIONS.clear()
        execution._SESSIONS.update(saved)


def test_active_pools_shrinks_when_a_pool_is_freed():
    """Freeing a session (close_session / unregister_program) drops its row —
    which is what makes the per-test cleanup keep the list bounded."""
    store = object()
    saved = dict(execution._SESSIONS)
    execution._SESSIONS.clear()
    try:
        execution._SESSIONS[(id(store), "p-a")] = _sess(fast_bytes=200 << 20)
        execution._SESSIONS[(id(store), "p-b")] = _sess(fast_bytes=200 << 20)
        assert len(execution.active_pools(store=store)) == 2

        del execution._SESSIONS[(id(store), "p-a")]        # as close_session does
        remaining = execution.active_pools(store=store)
        assert [r["prog_id"] for r in remaining] == ["p-b"]
    finally:
        execution._SESSIONS.clear()
        execution._SESSIONS.update(saved)
