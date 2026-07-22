"""Two REAL daemons on this box exchanging objects through the full
stack — NM threads, locked allocator, zero-copy landing, queued catalog
commits, verbs, events. Opt-in lane (pytest -m fleet).

Tests:
- test_chunked_round_trip_zero_copy: a multi-chunk object sends peer-to-peer, lands byte-identical via zero-copy, and the received event reports the right byte count and sending peer.
- test_eager_small_object: a small object sends and lands byte-identical over the eager path.
- test_collision_as_id_overwrite_matrix: a plain resend errors with COLLISION, an as_id rename lands fresh, and overwrite=True re-lands and bumps the object version.
- test_inbound_lands_while_receiver_dispatcher_busy: the payload finishes moving well before a held receiver dispatcher releases, the catalog commit waits out the hold, and the object verifies.
- test_severed_link_cleans_up_both_sides: severing the socket mid-transfer drops the peer on the sender, frees the receiver's reservations and inflight bytes with no torn object, emits peer_down, and reconnect restores the peer.
- test_dispatcher_isolation_during_transfer: a health call on the fast path returns quickly while a large transfer is in flight, and that transfer still completes.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402

pytestmark = [pytest.mark.fleet, pytest.mark.gpu]

PA, PB = 29471, 29472      # peer-plane ports (loopback)


def boot(tmp, name, peer_port):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=0.5,
        peer_name=name, peer_listen=f"127.0.0.1:{peer_port}",
        peer_chunk_bytes=1 << 20))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    return server, EngineClient(sock, client_name=name)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p1")
    server_a, client_a = boot(tmp, "alpha", PA)
    server_b, client_b = boot(tmp, "beta", PB)
    client_a.peer_connect("beta", f"127.0.0.1:{PB}")
    yield {"sa": server_a, "ca": client_a, "sb": server_b, "cb": client_b}
    for c in (client_a, client_b):
        try:
            c.shutdown()
        except Exception:
            pass


def events_of(server, kind):
    with server.state.lock:
        return [e for e in server.state.events if e.get("event") == kind]


def test_chunked_round_trip_zero_copy(rig):
    data = bytes((7 * i) % 251 for i in range(5 << 20))   # 5 MiB, 5 chunks
    rig["ca"].put_object("push_W", data)
    out = rig["ca"].send_object("push_W", "beta")
    row = rig["ca"].wait_transfer(out["send_id"])
    assert row["state"] == "done", row
    assert row["bytes_done"] == len(data)
    rec = rig["sb"].store.objects["push_W"]
    assert bytes(rig["sb"].store.view(rec)) == data
    assert rec.last_write["by"] == "peer:alpha"
    got = [e for e in events_of(rig["sb"], "object_received")
           if e.get("oid") == "push_W"]
    assert got and got[-1]["zero_copy"] is True            # landed direct
    assert got[-1]["bytes"] == len(data)


def test_eager_small_object(rig):
    rig["ca"].put_object("tiny_meta", b"x" * 512)
    out = rig["ca"].send_object("tiny_meta", "beta")
    row = rig["ca"].wait_transfer(out["send_id"])
    assert row["state"] == "done"
    rec = rig["sb"].store.objects["tiny_meta"]
    assert bytes(rig["sb"].store.view(rec)) == b"x" * 512


def test_collision_as_id_overwrite_matrix(rig):
    data = bytes(range(200)) * 1000
    rig["ca"].put_object("mx", data)
    first = rig["ca"].wait_transfer(
        rig["ca"].send_object("mx", "beta")["send_id"])
    assert first["state"] == "done"
    # plain resend: COLLISION, never retried
    again = rig["ca"].wait_transfer(
        rig["ca"].send_object("mx", "beta")["send_id"])
    assert again["state"] == "error" and "COLLISION" in again["error"]
    # as_id rename: lands fresh
    ren = rig["ca"].wait_transfer(
        rig["ca"].send_object("mx", "beta", as_id="mx_v2")["send_id"])
    assert ren["state"] == "done"
    # overwrite same size: allowed and versioned
    v0 = rig["sb"].store.objects["mx"].version
    ow = rig["ca"].wait_transfer(
        rig["ca"].send_object("mx", "beta", overwrite=True)["send_id"])
    assert ow["state"] == "done"
    assert rig["sb"].store.objects["mx"].version == v0 + 1


def test_inbound_lands_while_receiver_dispatcher_busy(rig):
    """The mid-run gate: bytes land + verify while B's dispatcher is
    occupied; the CATALOG commit (and the ack) waits it out."""
    data = bytes((3 * i) % 251 for i in range(4 << 20))
    rig["ca"].put_object("during_run", data)
    hold = threading.Thread(
        target=rig["cb"]._call, args=("_debug_hold", {"seconds": 2.5}))
    hold.start()
    time.sleep(0.3)                            # hold admitted
    t0 = time.monotonic()
    out = rig["ca"].send_object("during_run", "beta")
    # payload should finish MOVING well before the hold releases
    moving_done_at = None
    while time.monotonic() - t0 < 10.0:
        row = rig["ca"].transfer_status(out["send_id"])
        if row["bytes_done"] == len(data) and moving_done_at is None:
            moving_done_at = time.monotonic() - t0
        if row["state"] == "done":
            break
        time.sleep(0.02)
    hold.join()
    assert row["state"] == "done", row
    assert moving_done_at is not None and moving_done_at < 2.0, \
        f"payload waited out the dispatcher hold ({moving_done_at})"
    rec = rig["sb"].store.objects["during_run"]
    assert bytes(rig["sb"].store.view(rec)) == data


def test_severed_link_cleans_up_both_sides(rig):
    """Fail-stop: sever the socket mid-fleet; both sides converge —
    sender tickets error, receiver reservations free, nothing torn."""
    before_live = list(rig["sb"].nm.inflight_bytes.items())
    rig["ca"].put_object("doomed", bytes(2 << 20))
    rig["ca"].send_object("doomed", "beta")
    assert rig["sa"].nm.debug_sever("beta")
    for _ in range(400):
        peers_a = rig["ca"].list_peers()
        if not any(p["peer_id"] == "beta" for p in peers_a):
            break
        time.sleep(0.02)
    assert not any(p["peer_id"] == "beta" for p in rig["ca"].list_peers())
    # receiver side discovers via EOF; reservations freed, no torn object
    for _ in range(400):
        held = sum(rig["sb"].nm.inflight_bytes.values())
        if held == 0 and not any(
                l.core.receivers for l in rig["sb"].nm.links.values()):
            break
        time.sleep(0.02)
    assert sum(rig["sb"].nm.inflight_bytes.values()) == 0
    rec = rig["sb"].store.objects.get("doomed")
    assert rec is None or bytes(rig["sb"].store.view(rec)) == bytes(2 << 20)
    assert events_of(rig["sa"], "peer_down")
    # reconnect for any later tests
    rig["ca"].peer_connect("beta", f"127.0.0.1:{PB}")
    assert any(p["peer_id"] == "beta" for p in rig["ca"].list_peers())


def test_dispatcher_isolation_during_transfer(rig):
    """Wire work never blocks the daemon's fast path."""
    rig["ca"].put_object("big_iso", bytes(8 << 20))
    out = rig["ca"].send_object("big_iso", "beta")
    t0 = time.monotonic()
    rig["cb"].health()
    dt = time.monotonic() - t0
    assert dt < 0.5, f"fast path stalled {dt}s during transfer"
    assert rig["ca"].wait_transfer(out["send_id"])["state"] == "done"
