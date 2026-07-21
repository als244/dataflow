"""Cross-box gates: the real two-box pair over the direct link,
socket transport. Hosts come from topology.toml (skipped when absent
or when it has no remote host — see topology.example.toml).

The test OWNS the remote daemon lifecycle (portable daemonizer +
pidfile group kill over ssh): boots dataflowd on the remote host with
the NM listening on its topology data-plane address, runs the gates,
kills it on teardown. Remote-side verification runs small
EngineClient snippets on the remote host's python.
"""
import hashlib
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.distributed.hosts import run_py
from dataflow_training.distributed import daemons
from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("cross-box gates need a topology.toml with a remote "
                "host", allow_module_level=True)

pytestmark = pytest.mark.fleet

LOCAL = TOPO.local()
REMOTE = TOPO.remotes()[0]
LANE = "p2"
PORT = 29600
REMOTE_SOCK = daemons.paths(REMOTE, LANE)["sock"]

REMOTE_PRELUDE = (
    "import sys; sys.path.insert(0, 'src'); "
    "from dataflow.service import EngineClient; "
    f"c = EngineClient('{REMOTE_SOCK}', client_name='p2-verify'); "
)


def remote_py(code: str, *, timeout: float = 120.0) -> str:
    return run_py(REMOTE, code, timeout=timeout)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    daemons.kill(REMOTE, lane=LANE)
    daemons.launch(REMOTE, lane=LANE, slab_gib=0.5, peer_port=PORT)
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
        peer_chunk_bytes=8 << 20))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    client = EngineClient(sock, client_name=LOCAL.name)
    client.peer_connect(REMOTE.name, REMOTE.peer_addr(PORT))
    yield {"server": server, "client": client, "sock": sock}
    try:
        client.shutdown()
    except Exception:
        pass
    daemons.kill(REMOTE, lane=LANE)


def test_crossbox_byte_identity_and_zero_copy(rig):
    data = bytes((11 * i) % 251 for i in range(32 << 20))    # 32 MiB
    rig["client"].put_object("xbox_W", data)
    out = rig["client"].send_object("xbox_W", REMOTE.name)
    row = rig["client"].wait_transfer(out["send_id"], timeout=60)
    assert row["state"] == "done", row
    remote = remote_py(
        REMOTE_PRELUDE
        + "import hashlib; "
          "b = c.get_object('xbox_W'); "
          "print(len(b), hashlib.sha256(bytes(b)).hexdigest()); c.close()")
    nbytes, sha = remote.split()
    assert int(nbytes) == len(data)
    assert sha == hashlib.sha256(data).hexdigest()


def test_crossbox_reverse_direction(rig):
    """remote -> local: the remote daemon initiates the send over the
    SAME link (peer links are bidirectional)."""
    remote_py(REMOTE_PRELUDE
              + "c.put_object('xbox_back', bytes(range(256)) * 8192); "
                f"r = c.send_object('xbox_back', '{LOCAL.name}'); "
                "print(c.wait_transfer(r['send_id'], timeout=60)['state']); "
                "c.close()")
    rec = rig["server"].store.objects.get("xbox_back")
    assert rec is not None
    assert bytes(rig["server"].store.view(rec)) == bytes(range(256)) * 8192
    assert rec.last_write["by"] == f"peer:{REMOTE.name}"


def test_crossbox_throughput_matches_probed_bw(rig):
    """ZERO-COPY time gate: 256 MiB must move in about size/peak_bw,
    where peak_bw was measured by the connect-time probe on THIS
    link — no hardcoded speeds, and hidden copies show up as a miss."""
    link = rig["server"].nm.links.get(REMOTE.name)
    assert link is not None and "socket" in link.peak_gbps, \
        "connect-time bw probe missing"
    peak = link.peak_gbps["socket"]
    data = bytes(256 << 20)
    rig["client"].put_object("xbox_big", data)
    t0 = time.monotonic()
    out = rig["client"].send_object("xbox_big", REMOTE.name)
    row = rig["client"].wait_transfer(out["send_id"], timeout=120)
    dt = time.monotonic() - t0
    assert row["state"] == "done", row
    gbps = len(data) * 8 / dt / 1e9
    expected = len(data) * 8 / (peak * 1e9)
    print(f"\n[P2a] socket cross-box: 256 MiB in {dt:.2f}s "
          f"= {gbps:.1f} Gbit/s (probe {peak})")
    assert dt <= expected * 1.35 + 0.30, (
        f"256 MiB took {dt:.2f}s vs {expected:.2f}s at the probed "
        f"{peak} Gbit/s")


def test_crossbox_capacity_backpressure(rig):
    """An object bigger than the remote's free slab NACKs CAPACITY,
    retries with backoff, then fails LOUD — nothing torn remotely."""
    rig["client"].release_object("xbox_big")  # room LOCALLY for the src
    data = bytes(300 << 20)   # fits local free; exceeds remote free
    rig["client"].put_object("xbox_toobig", data)
    out = rig["client"].send_object("xbox_toobig", REMOTE.name)
    row = rig["client"].wait_transfer(out["send_id"], timeout=60)
    assert row["state"] == "error"
    assert "CAPACITY" in row["error"]
    remote = remote_py(REMOTE_PRELUDE
                       + "print('xbox_toobig' in [o['id'] for o in "
                         "c.list_objects()]); c.close()")
    assert remote.strip() == "False"
    rig["client"].release_object("xbox_toobig")
