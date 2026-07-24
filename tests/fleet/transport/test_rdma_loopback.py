"""Loopback gates: the rdma-host transport between two daemons on
THIS box (RC QPs through the local NIC — RoCE loopback). Always
runnable on a box with an ACTIVE RoCE port; the cross-box twin lives in
test_rdma_crossbox.py.

Tests:
- test_rdma_round_trip_byte_identity: a round-trip object lands byte-identical and the received event flags zero_copy False, proving the NIC addressed the landing region rather than the socket reader.
- test_rdma_throughput_matches_probed_bw: a 256 MiB send completes within a small factor of size / probed-rdma-bandwidth, catching any hidden staging copy.
- test_rdma_small_object_round_trip: a tiny object round-trips and lands byte-identical.
- test_rdma_allreduce_zero_copy_from_registered_slab: the rdma-lane collective scratch lives inside the registered slab MR, the rdma QP is up, and the native-dtype allreduce matches the fp32 reference bitwise with identical replicas at world 2.
"""
import socket
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")
pytest.importorskip("pyverbs")

from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow.service.peer.rdma import roce_v2_ipv4_gid  # noqa: E402

pytestmark = [pytest.mark.fleet, pytest.mark.gpu]

TOPO = load_topology_or_none()
DEV = TOPO.local().ib_dev if TOPO is not None else None
if DEV is None:
    pytest.skip("rdma loopback needs a topology.toml with ib_dev on "
                "the local host", allow_module_level=True)
if roce_v2_ipv4_gid(DEV) is None:
    pytest.skip(f"no RoCE v2 GID on {DEV}", allow_module_level=True)

def free_port():
    """An unused loopback port, claimed at fixture time.

    Fixed ports made this rig hostage to anything still holding them — a
    daemon leaked by an earlier run, or a second copy of the suite. The
    symptom was the peer never reaching RTS ten seconds later, reported as
    an empty-list assertion with nothing to act on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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
    pa, pb = free_port(), free_port()
    sa, ca = boot(tmp, "rd-alpha", pa)
    sb, cb = boot(tmp, "rd-beta", pb)
    ca.peer_connect("rd-beta", f"127.0.0.1:{pb}")
    deadline = time.time() + 10
    rdma_up = []
    while time.time() < deadline:
        rdma_up = [e for e in sa.state.events
                   if e.get("event") == "peer_rdma_up"]
        if rdma_up:
            break
        time.sleep(0.05)
    assert rdma_up, (
        f"RC QPs never reached RTS within 10s on {DEV} "
        f"(127.0.0.1:{pa} <-> :{pb}). Peer events seen: "
        f"{sorted({e.get('event') for e in sa.state.events})}. "
        f"A DOWN port or a daemon left over from an earlier run holding the "
        f"device are the usual causes; `ibv_devinfo` shows port state.")
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


def probed_gbps(server, peer: str, plane: str, timeout=15.0):
    """The connect-time bandwidth probe's measurement for a plane —
    the reference every transfer-time gate derives its bound from
    (no hardcoded link speeds)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        link = server.nm.links.get(peer)
        if link is not None and plane in link.peak_gbps:
            return link.peak_gbps[plane]
        time.sleep(0.1)
    raise AssertionError(f"bw probe never measured {plane} to {peer}")


def test_rdma_throughput_matches_probed_bw(rig):
    """ZERO-COPY time gate: a large send must complete in about
    size/peak_bw — any hidden staging copy shows up as a miss."""
    peak = probed_gbps(rig["sa"], "rd-beta", "rdma")
    data = bytes(256 << 20)
    rig["ca"].put_object("rdma_big", data)
    t0 = time.monotonic()
    out = rig["ca"].send_object("rdma_big", "rd-beta")
    row = rig["ca"].wait_transfer(out["send_id"], timeout=60)
    dt = time.monotonic() - t0
    assert row["state"] == "done", row
    gbps = len(data) * 8 / dt / 1e9
    expected = len(data) * 8 / (peak * 1e9)
    assert dt <= expected * 1.35 + 0.30, (
        f"256 MiB took {dt:.3f}s vs {expected:.3f}s at the probed "
        f"{peak} Gbit/s — a copy crept into the path")
    print(f"\nrdma-host LOOPBACK: 256 MiB in {dt:.3f}s "
          f"= {gbps:.1f} Gbit/s")
    rig["ca"].release_object("rdma_big")


def test_rdma_small_object_round_trip(rig):
    rig["ca"].put_object("rdma_tiny", b"y" * 600)
    out = rig["ca"].send_object("rdma_tiny", "rd-beta")
    assert rig["ca"].wait_transfer(out["send_id"])["state"] == "done"
    rec = rig["sb"].store.objects["rdma_tiny"]
    assert bytes(rig["sb"].store.view(rec)) == b"y" * 600


def test_rdma_allreduce_zero_copy_from_registered_slab(rig):
    """Collectives over the rdma lane: staging regions are SLAB extents
    (inside the NIC's MR — nothing is copied to become NIC-reachable),
    the exchange is a one-sided RDMA_WRITE into the peer's landing
    region, and the native-dtype reduce matches the fp32 reference
    bitwise at world 2."""
    sa, sb, ca = rig["sa"], rig["sb"], rig["ca"]
    ca._call("create_peer_group",
             {"name": "rdp", "members": ["rd-alpha", "rd-beta"],
              "backend": "hostmem"})
    ha = sa.nm.group_handles()["rdp"]
    hb = sb.nm.group_handles()["rdp"]
    for server, h in ((sa, ha), (sb, hb)):
        comm = h.comm
        assert comm is not None
        assert comm.rdma_qp() is not None, "rdma lane not up for COLL"
        slab = server.store.slab
        cap = server.store.allocator.capacity
        for region in (comm.out, comm.land):
            assert slab.ptr <= region.ptr < slab.ptr + cap, \
                "scratch must live inside the registered slab MR"

    g = torch.Generator(device="cuda").manual_seed(7)
    a = torch.randn(1 << 20, device="cuda", generator=g,
                    dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(1 << 20, device="cuda", generator=g,
                    dtype=torch.float32).to(torch.bfloat16)
    want = (a.float() + b.float()).to(torch.bfloat16)
    ta, tb = a.clone(), b.clone()
    err = []

    def post(h, t):
        try:
            h.allreduce(t)
        except Exception as e:
            err.append(e)

    torch.cuda.default_stream().synchronize()   # producer contract
    ja = threading.Thread(target=post, args=(ha, ta))
    jb = threading.Thread(target=post, args=(hb, tb))
    ja.start(); jb.start(); ja.join(30); jb.join(30)
    assert not err, err
    ha.stream.synchronize(); hb.stream.synchronize()
    assert torch.equal(ta, tb), "replicas diverged"
    assert torch.equal(ta, want), "sum wrong vs fp32 reference"
