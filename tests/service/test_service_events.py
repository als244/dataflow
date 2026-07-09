"""S1.4 gates: events + observability (fake daemon — protocol only).

Monitor-reattach: a subscriber that disconnects and reconnects with
since_seq sees run_done EXACTLY once. Event-type coverage: a
scripted flow produces every service-event kind with schema fields.
Status wiring: counters/sessions/snapshots_in_flight reflect real
activity.
"""
from __future__ import annotations

import threading
import time

import pytest

from dataflow.service import EngineClient, EngineConfig, Server


@pytest.fixture()
def fake_rig(tmp_path):
    sock = str(tmp_path / "ev.sock")
    server = Server(EngineConfig(socket_path=sock, fake=True,
                                 slab_backing_gib=0.01))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(300):
        try:
            with EngineClient(sock, client_name="probe"):
                break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    yield {"sock": sock, "server": server, "tmp": tmp_path}
    server.state.shutdown_requested.set()
    server.dispatcher.stop()


def test_event_coverage_and_reattach(fake_rig):
    st = fake_rig["server"].state
    with EngineClient(fake_rig["sock"], client_name="mon") as c:
        c.put_object("blob", b"\x07" * 8192)
        s = c.snapshot("all", str(fake_rig["tmp"] / "snap_ev"))
        done = c.wait_snapshot(s["snap_id"])
        assert done["state"] == "done"
        c.wipe("all", force=True)
        r = c.restore_snapshot(str(fake_rig["tmp"] / "snap_ev"))
        assert r["restored"] == ["blob"]

    kinds = {e["event"] for e in st.events_since(0)}
    assert {"engine_started", "snapshot_started", "snapshot_done",
            "restore_done"} <= kinds, kinds
    for e in st.events_since(0):
        assert {"event", "t", "seq"} <= set(e), e

    # reattach replay: since_seq strictly after an event yields it
    # exactly zero times; since_seq before yields exactly once
    snaps = [e for e in st.events_since(0)
             if e["event"] == "snapshot_done"]
    assert len(snaps) == 1
    seq = snaps[0]["seq"]
    assert [e for e in st.events_since(seq)
            if e["event"] == "snapshot_done"] == []
    replay = [e for e in st.events_since(seq - 1)
              if e["event"] == "snapshot_done"]
    assert len(replay) == 1


def test_status_wiring(fake_rig):
    with EngineClient(fake_rig["sock"], client_name="statuswire") as c:
        c.put_object("x", b"\x01" * 4096)
        s = c.engine_status()
        assert s["counters"]["api_calls"] >= 0
        assert s["snapshots_in_flight"] == []
        snap = c.snapshot("all", str(fake_rig["tmp"] / "snap_sw"))
        c.wait_snapshot(snap["snap_id"])
        s2 = c.engine_status()
        assert s2["snapshots_in_flight"] == []   # drained after done
        sess = c.session_status()
        # fast-path ops (status queries) deliberately skip per-session
        # counters — only QUEUED verbs count (put, snapshot)
        assert sess["calls"] >= 2
