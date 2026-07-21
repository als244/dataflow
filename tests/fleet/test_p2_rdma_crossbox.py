"""Cross-box rdma-host gates over the real direct link: RDMA_WRITE
into the remote pinned slab. Hosts + HCA names come from topology.toml
(skipped when absent, when it has no remote host, or when either side
lacks an ib_dev). Remote daemon owned via the portable daemonizer."""
import hashlib
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("pyverbs")

from dataflow_training.distributed.hosts import run_py
from dataflow_training.distributed import daemons
from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("cross-box gates need a topology.toml with a remote "
                "host", allow_module_level=True)
if TOPO.local().ib_dev is None or TOPO.remotes()[0].ib_dev is None:
    pytest.skip("rdma cross-box gates need ib_dev on both hosts",
                allow_module_level=True)

pytestmark = pytest.mark.fleet

LOCAL = TOPO.local()
REMOTE = TOPO.remotes()[0]
LANE = "p2rdma"
PORT = 29610
REMOTE_SOCK = daemons.paths(REMOTE, LANE)["sock"]

REMOTE_PRELUDE = (
    "import sys; sys.path.insert(0, 'src'); "
    "from dataflow.service import EngineClient; "
    f"c = EngineClient('{REMOTE_SOCK}', client_name='p2rdma-verify'); "
)


def remote_py(code: str, *, timeout: float = 120.0) -> str:
    return run_py(REMOTE, code, timeout=timeout)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    daemons.kill(REMOTE, lane=LANE)
    daemons.launch(REMOTE, lane=LANE, backing_gib=0.5, peer_port=PORT,
                  extra_flags=f"--peer-rdma-device {REMOTE.ib_dev}")
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            remote_py(REMOTE_PRELUDE + "print(c.health()['ok']); "
                                       "c.close()", timeout=20)
            break
        except Exception:
            time.sleep(1.0)
    else:
        raise RuntimeError(
            f"{REMOTE.name} daemon did not come up; see "
            f"{daemons.paths(REMOTE, LANE)['log']} on that host")

    tmp = tmp_path_factory.mktemp(LANE)
    sock = str(tmp / "local.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=0.5,
        peer_name=LOCAL.name, peer_listen=LOCAL.peer_addr(PORT),
        peer_rdma_device=LOCAL.ib_dev))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    client = EngineClient(sock, client_name=LOCAL.name)
    client.peer_connect(REMOTE.name, REMOTE.peer_addr(PORT))
    deadline = time.time() + 15
    while time.time() < deadline:
        if any(e.get("event") == "peer_rdma_up" for e in server.state.events):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("cross-box RC QPs never reached RTS")
    yield {"server": server, "client": client}
    try:
        client.shutdown()
    except Exception:
        pass
    daemons.kill(REMOTE, lane=LANE)


def test_rdma_crossbox_byte_identity(rig):
    data = bytes((17 * i) % 251 for i in range(32 << 20))
    rig["client"].put_object("xr_W", data)
    out = rig["client"].send_object("xr_W", REMOTE.name)
    row = rig["client"].wait_transfer(out["send_id"], timeout=60)
    assert row["state"] == "done", row
    remote = remote_py(REMOTE_PRELUDE
                       + "import hashlib; b = c.get_object('xr_W'); "
                         "print(len(b), "
                         "hashlib.sha256(bytes(b)).hexdigest()); "
                         "c.close()")
    nbytes, sha = remote.split()
    assert int(nbytes) == len(data)
    assert sha == hashlib.sha256(data).hexdigest()


def test_rdma_crossbox_reverse(rig):
    remote_py(REMOTE_PRELUDE
              + "c.put_object('xr_back', bytes(range(251)) * 65536); "
                f"r = c.send_object('xr_back', '{LOCAL.name}'); "
                "print(c.wait_transfer(r['send_id'], timeout=60)['state']); "
                "c.close()")
    rec = rig["server"].store.objects.get("xr_back")
    assert rec is not None
    assert bytes(rig["server"].store.view(rec)) == bytes(range(251)) * 65536


def test_rdma_crossbox_throughput_matches_probed_bw(rig):
    """ZERO-COPY time gate against the probe's rdma-plane measurement
    (sender slab MR -> receiver slab extent; no staging copies)."""
    deadline = time.time() + 15
    peak = None
    while time.time() < deadline:
        link = rig["server"].nm.links.get(REMOTE.name)
        if link is not None and "rdma" in link.peak_gbps:
            peak = link.peak_gbps["rdma"]
            break
        time.sleep(0.1)
    assert peak is not None, "rdma bw probe never completed"
    data = bytes(256 << 20)
    rig["client"].put_object("xr_big", data)
    t0 = time.monotonic()
    out = rig["client"].send_object("xr_big", REMOTE.name)
    row = rig["client"].wait_transfer(out["send_id"], timeout=120)
    dt = time.monotonic() - t0
    assert row["state"] == "done", row
    gbps = len(data) * 8 / dt / 1e9
    expected = len(data) * 8 / (peak * 1e9)
    print(f"\n[P2b] rdma cross-box: 256 MiB in {dt:.3f}s "
          f"= {gbps:.1f} Gbit/s (probe {peak})")
    assert dt <= expected * 1.35 + 0.30, (
        f"256 MiB took {dt:.3f}s vs {expected:.3f}s at the probed "
        f"{peak} Gbit/s")
