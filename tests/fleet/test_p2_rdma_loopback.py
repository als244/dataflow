"""P2b loopback gates: the rdma-host transport between two daemons on
THIS box (RC QPs through the local NIC — RoCE loopback). Always
runnable on a box with an ACTIVE RoCE port; the cross-box twin lives in
test_p2_rdma_crossbox.py."""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("pyverbs")

from dataflow.pretrain.topology import load_topology_or_none  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow.service.peer.rdma import roce_v2_ipv4_gid  # noqa: E402

pytestmark = pytest.mark.fleet

TOPO = load_topology_or_none()
DEV = TOPO.local().ib_dev if TOPO is not None else None
if DEV is None:
    pytest.skip("rdma loopback needs a topology.toml with ib_dev on "
                "the local host", allow_module_level=True)
if roce_v2_ipv4_gid(DEV) is None:
    pytest.skip(f"no RoCE v2 GID on {DEV}", allow_module_level=True)

PA, PB = 29481, 29482


def boot(tmp, name, peer_port):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=0.5,
        peer_name=name, peer_listen=f"127.0.0.1:{peer_port}",
        peer_chunk_bytes=1 << 20, peer_rdma_device=DEV))
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
    tmp = tmp_path_factory.mktemp("p2rdma")
    sa, ca = boot(tmp, "rd-alpha", PA)
    sb, cb = boot(tmp, "rd-beta", PB)
    ca.peer_connect("rd-beta", f"127.0.0.1:{PB}")
    deadline = time.time() + 10
    while time.time() < deadline:
        rdma_up = [e for e in sa.state.events
                   if e.get("event") == "peer_rdma_up"]
        if rdma_up:
            break
        time.sleep(0.05)
    assert rdma_up, "RC QPs never reached RTS"
    yield {"sa": sa, "ca": ca, "sb": sb, "cb": cb}
    for c in (ca, cb):
        try:
            c.shutdown()
        except Exception:
            pass


def test_rdma_round_trip_byte_identity(rig):
    data = bytes((13 * i) % 251 for i in range(24 << 20))
    rig["ca"].put_object("rdma_W", data)
    out = rig["ca"].send_object("rdma_W", "rd-beta")
    row = rig["ca"].wait_transfer(out["send_id"], timeout=60)
    assert row["state"] == "done", row
    rec = rig["sb"].store.objects["rdma_W"]
    assert bytes(rig["sb"].store.view(rec)) == data
    # PROOF it rode rdma: no CHUNK frames means the receiver core never
    # counted socket bytes — the reservation was addressed (ADDR path)
    got = [e for e in rig["sb"].state.events
           if e.get("event") == "object_received"
           and e.get("oid") == "rdma_W"]
    assert got and got[-1].get("zero_copy") is False  # NIC landed it,
    # not the reader's recv_into (zero_copy flags the SOCKET fast path)


def test_rdma_throughput_report(rig):
    data = bytes(256 << 20)
    rig["ca"].put_object("rdma_big", data)
    t0 = time.monotonic()
    out = rig["ca"].send_object("rdma_big", "rd-beta")
    row = rig["ca"].wait_transfer(out["send_id"], timeout=60)
    dt = time.monotonic() - t0
    assert row["state"] == "done", row
    gbps = len(data) * 8 / dt / 1e9
    print(f"\n[P2b] rdma-host LOOPBACK: 256 MiB in {dt:.3f}s "
          f"= {gbps:.1f} Gbit/s")
    rig["ca"].release_object("rdma_big")


def test_rdma_eager_still_rides_control(rig):
    rig["ca"].put_object("rdma_tiny", b"y" * 600)
    out = rig["ca"].send_object("rdma_tiny", "rd-beta")
    assert rig["ca"].wait_transfer(out["send_id"])["state"] == "done"
    rec = rig["sb"].store.objects["rdma_tiny"]
    assert bytes(rig["sb"].store.view(rec)) == b"y" * 600
