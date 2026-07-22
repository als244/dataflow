"""Service protocol gates: framing, handshake, FIFO vs fast-path,
tickets, events, error envelopes, shutdown. CPU-only (fake boot,
in-process server thread over a real unix socket).

Tests:
- test_handshake_and_health: handshake reports the wire schema version and health() returns ok with non-negative uptime.
- test_schema_skew_rejected: a client built against a mismatched schema version is refused with SCHEMA_VERSION_SKEW.
- test_fast_path_answers_while_dispatcher_held: fast-path engine_status answers mid-hold while a queued call sits behind the dispatcher, and the second call completes only after the hold drains (FIFO).
- test_ticket_async_and_blocking_parity: the blocking and async-ticket forms of a call return identical results.
- test_unknown_op_error_envelope: an unknown op raises ServiceError with code UNKNOWN_OP.
- test_register_program_requires_resolver_kind: a resolver spec missing "kind" or naming an unregistered kind is refused with BAD_REQUEST that names the registered kinds.
- test_subscribe_since_seq_zero_replays_all_events: subscribe_events(since_seq=0) replays the full event log including engine_started and the emitted run events.
- test_session_status_tracks_calls: session_status reports the client name, call count, and the last op and its status.
- test_engine_status_lists_all_client_sessions: engine_status lists every connected session and one client observes another's queued work.
- test_shutdown_terminates_daemon: shutdown() returns ok and serve_forever exits.
"""
from __future__ import annotations

import threading
import time

import pytest

from dataflow.service import EngineClient, EngineConfig, Server, ServiceError
from dataflow.service.wire import SCHEMA_VERSION


@pytest.fixture()
def daemon(tmp_path):
    sock = str(tmp_path / "svc.sock")
    server = Server(EngineConfig(socket_path=sock, fake=True))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(200):
        try:
            with EngineClient(sock, client_name="probe"):
                break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    else:
        raise RuntimeError("daemon did not come up")
    yield sock, server
    server.state.shutdown_requested.set()
    server.dispatcher.stop()


def test_handshake_and_health(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="t1") as c:
        assert c.engine_info["schema_version"] == SCHEMA_VERSION
        h = c.health()
        assert h["ok"] and h["uptime_s"] >= 0


def test_schema_skew_rejected(daemon, monkeypatch):
    sock, _ = daemon
    import dataflow.service.client as cl

    monkeypatch.setattr(cl, "SCHEMA_VERSION", "dataflow-service/s0-bogus")
    with pytest.raises(ServiceError) as ei:
        EngineClient(sock, client_name="skew")
    assert ei.value.code == "SCHEMA_VERSION_SKEW"


def test_fast_path_answers_while_dispatcher_held(daemon):
    """The core timing-model gate: a queued call occupies the
    dispatcher; fast-path status stays responsive; a second queued
    call completes strictly after the first (FIFO)."""
    sock, _ = daemon
    hold_s = 0.6
    with EngineClient(sock, client_name="t2") as c:
        t0 = time.perf_counter()
        hold = c._debug_hold(hold_s, wait=False)        # occupies dispatcher
        second = c._debug_hold(0.0, wait=False)        # queued behind it

        st = c.engine_status()                          # fast path, mid-hold
        fast_latency = time.perf_counter() - t0
        # queue_depth >= 1 is the ORDERING proof: the fast-path reply
        # came back while the hold still occupied the dispatcher and the
        # second call sat queued behind it (a FIFO status call could only
        # answer after the hold drained, leaving depth 0). The latency
        # bound is relative to that same hold, not an absolute wall.
        assert st["queue_depth"] >= 1
        assert fast_latency < hold_s / 2, \
            f"fast path stalled {fast_latency:.2f}s under a {hold_s}s hold"

        c.wait(second)                                  # FIFO: after hold
        total = time.perf_counter() - t0
        assert total >= hold_s, f"second call jumped the queue ({total:.2f}s)"
        c.wait(hold)


def test_ticket_async_and_blocking_parity(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="t3") as c:
        r1 = c._debug_hold(0.01)                        # blocking form
        tk = c._debug_hold(0.01, wait=False)            # async form
        r2 = c.wait(tk)
        assert r1 == r2 == {"held": 0.01}


def test_unknown_op_error_envelope(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="t4") as c:
        with pytest.raises(ServiceError) as ei:
            c._call("no_such_op", {})
        assert ei.value.code == "UNKNOWN_OP"


def test_register_program_requires_resolver_kind(daemon):
    """The resolver registry dispatches on resolver_spec["kind"]: a
    spec without one — and an unregistered kind — are refused loudly,
    naming what IS registered (register_all provides model_family)."""
    from dataflow.core.jsonio import program_to_dict
    from dataflow.core.program import Program
    from dataflow_training.register import register_all

    register_all()
    prog = program_to_dict(Program(name="empty"))
    sock, _ = daemon
    with EngineClient(sock, client_name="t4b") as c:
        with pytest.raises(ServiceError) as ei:
            c.register_program(prog, resolver={"cfg": {}})
        assert ei.value.code == "BAD_REQUEST"
        assert "kind" in ei.value.message
        assert "model_family" in ei.value.message
        with pytest.raises(ServiceError) as ei:
            c.register_program(prog, resolver={"kind": "not_registered"})
        assert ei.value.code == "BAD_REQUEST"
        assert "model_family" in ei.value.message


def test_subscribe_since_seq_zero_replays_all_events(daemon):
    sock, server = daemon
    with EngineClient(sock, client_name="t5") as c:
        server.state.emit("run_started", run_id="r-000001")
        server.state.emit("run_done", run_id="r-000001")
        it = c.subscribe_events(since_seq=0)            # replay everything
        seen = []
        deadline = time.time() + 2
        while time.time() < deadline and len(seen) < 3:
            ev = next(it)
            seen.append(ev["event"])
        assert "engine_started" in seen[0:1] or "engine_started" in seen
        assert "run_started" in seen and "run_done" in seen


def test_session_status_tracks_calls(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="tracked") as c:
        c._debug_hold(0.0)
        s = c.session_status()
        assert s["client_name"] == "tracked"
        assert s["calls"] >= 1
        assert s["recent_calls"][-1]["op"] == "_debug_hold"
        assert s["recent_calls"][-1]["status"] == "ok"


def test_engine_status_lists_all_client_sessions(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="a") as a, \
            EngineClient(sock, client_name="b") as b:
        st = a.engine_status()
        names = {s["client_name"] for s in st["sessions"]}
        assert {"a", "b"} <= names
        # b sees a's work in the shared queue: a holds the dispatcher and
        # queues a second call behind it, so b observes a real depth >= 1.
        hold = a._debug_hold(0.4, wait=False)   # occupies the dispatcher
        tk = a._debug_hold(0.0, wait=False)      # queued behind the hold
        assert b.engine_status()["queue_depth"] >= 1
        a.wait(tk)
        a.wait(hold)


def test_shutdown_terminates_daemon(tmp_path):
    sock = str(tmp_path / "svc2.sock")
    server = Server(EngineConfig(socket_path=sock, fake=True))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(200):
        try:
            c = EngineClient(sock, client_name="killer")
            break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    assert c.shutdown()["ok"] is True
    t.join(timeout=5)
    assert not t.is_alive(), "serve_forever did not exit after shutdown"
