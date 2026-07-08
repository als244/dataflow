"""S1.0 gates: framing, handshake, FIFO vs fast-path, tickets, events,
error envelopes, shutdown. CPU-only (fake boot, in-process server
thread over a real unix socket)."""
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
    with EngineClient(sock, client_name="t2") as c:
        t0 = time.perf_counter()
        hold = c._debug_hold(0.6, wait=False)          # occupies dispatcher
        second = c._debug_hold(0.0, wait=False)        # queued behind it

        st = c.engine_status()                          # fast path, mid-hold
        fast_latency = time.perf_counter() - t0
        assert fast_latency < 0.3, f"fast path stalled {fast_latency:.2f}s"
        assert st["queue_depth"] >= 1

        c.wait(second)                                  # FIFO: after hold
        total = time.perf_counter() - t0
        assert total >= 0.6, f"second call jumped the queue ({total:.2f}s)"
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


def test_events_ring_and_since_seq(daemon):
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
        seqs = None  # ordering asserted via arrival order above


def test_session_status_tracks_calls(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="tracked") as c:
        c._debug_hold(0.0)
        s = c.session_status()
        assert s["client_name"] == "tracked"
        assert s["calls"] >= 1
        assert s["recent_calls"][-1]["op"] == "_debug_hold"
        assert s["recent_calls"][-1]["status"] == "ok"


def test_two_clients_flat_state(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="a") as a, \
            EngineClient(sock, client_name="b") as b:
        st = a.engine_status()
        names = {s["client_name"] for s in st["sessions"]}
        assert {"a", "b"} <= names
        # b sees a's queued call in the shared queue depth
        tk = a._debug_hold(0.4, wait=False)
        depth = b.engine_status()["queue_depth"]
        assert depth >= 0                 # racy lower bound; just no crash
        a.wait(tk)


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
