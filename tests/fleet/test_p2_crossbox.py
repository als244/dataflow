"""P2a gates: the REAL two-box pair over the 25 GbE direct link —
chicago (this box, 5090) <-> tubingen (3090, `ssh tubingen`), socket
transport. Requires the tubingen environment; enable with
DATAFLOW_TUBINGEN=1 (skipped otherwise even in the fleet lane).

The test OWNS the remote daemon lifecycle over ssh: boots dataflowd on
tubingen with the NM listening on the direct-link address
(192.168.50.32 — host routes pin this traffic to the 25G ports), runs
the gates, kills it on teardown. Remote-side verification runs small
EngineClient snippets over ssh against tubingen's UDS.
"""
import hashlib
import os
import subprocess
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
if not os.environ.get("DATAFLOW_TUBINGEN"):
    pytest.skip("cross-box gates need DATAFLOW_TUBINGEN=1",
                allow_module_level=True)

from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402

pytestmark = pytest.mark.fleet

TUB = "tubingen"
TUB_PY = "~/miniconda3/envs/dataflow/bin/python"
TUB_REPO = "~/Documents/dataflow"
TUB_SOCK = "/tmp/dataflowd-p2.sock"
TUB_PEER = "192.168.50.32:29600"
CHI_PEER = "192.168.50.23:29600"


def ssh(cmd: str, *, timeout: float = 60.0) -> str:
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", TUB, cmd],
                         capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"ssh rc={out.returncode}: {out.stderr[-500:]}")
    return out.stdout


def ssh_fire_and_forget(cmd: str) -> None:
    """Launch-and-detach: pipes to DEVNULL so ssh never waits on the
    remote session's fds (a captured-pipe ssh can hang on daemon
    launches — findings ledger, P2)."""
    subprocess.Popen(["ssh", "-o", "BatchMode=yes", TUB, cmd],
                     stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def kill_remote_daemon() -> None:
    port = TUB_PEER.rsplit(":", 1)[1]
    ssh(f"pkill -f '[d]ataflowd.py start --socket {TUB_SOCK}' || true; "
        f"fuser -k {port}/tcp 2>/dev/null || true")


def tub_py(code: str, *, timeout: float = 120.0) -> str:
    quoted = code.replace("'", "'\"'\"'")
    return ssh(f"cd {TUB_REPO} && {TUB_PY} -c '{quoted}'",
               timeout=timeout)


REMOTE_PRELUDE = (
    "import sys; sys.path.insert(0, 'src'); "
    "from dataflow.service import EngineClient; "
    f"c = EngineClient('{TUB_SOCK}', client_name='p2-verify'); "
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    kill_remote_daemon()
    time.sleep(1.0)
    ssh_fire_and_forget(
        f"cd {TUB_REPO} && setsid nohup {TUB_PY} tools/dataflowd.py start "
        f"--socket {TUB_SOCK} --slab-gib 0.5 --peer-name tubingen "
        f"--peer-listen {TUB_PEER} > /tmp/dataflowd-p2.log 2>&1 "
        f"< /dev/null & exit")
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            tub_py(REMOTE_PRELUDE + "print(c.health()['ok']); c.close()",
                   timeout=20)
            break
        except (RuntimeError, subprocess.TimeoutExpired):
            time.sleep(1.0)
    else:
        raise RuntimeError("tubingen daemon did not come up; see "
                           "tubingen:/tmp/dataflowd-p2.log")

    tmp = tmp_path_factory.mktemp("p2")
    sock = str(tmp / "chicago.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=0.5,
        peer_name="chicago", peer_listen=CHI_PEER,
        peer_chunk_bytes=8 << 20))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    client = EngineClient(sock, client_name="chicago")
    client.peer_connect("tubingen", TUB_PEER)
    yield {"server": server, "client": client, "sock": sock}
    try:
        client.shutdown()
    except Exception:
        pass
    kill_remote_daemon()


def test_crossbox_byte_identity_and_zero_copy(rig):
    data = bytes((11 * i) % 251 for i in range(32 << 20))    # 32 MiB
    rig["client"].put_object("xbox_W", data)
    out = rig["client"].send_object("xbox_W", "tubingen")
    row = rig["client"].wait_transfer(out["send_id"], timeout=60)
    assert row["state"] == "done", row
    remote = tub_py(
        REMOTE_PRELUDE
        + "import hashlib; "
          "b = c.get_object('xbox_W'); "
          "print(len(b), hashlib.sha256(bytes(b)).hexdigest()); c.close()")
    nbytes, sha = remote.split()
    assert int(nbytes) == len(data)
    assert sha == hashlib.sha256(data).hexdigest()


def test_crossbox_reverse_direction(rig):
    """tubingen -> chicago: the remote daemon initiates the send over
    the SAME link (peer links are bidirectional)."""
    tub_py(REMOTE_PRELUDE
           + "c.put_object('xbox_back', bytes(range(256)) * 8192); "
             "r = c.send_object('xbox_back', 'chicago'); "
             "print(c.wait_transfer(r['send_id'], timeout=60)['state']); "
             "c.close()")
    rec = rig["server"].store.objects.get("xbox_back")
    assert rec is not None
    assert bytes(rig["server"].store.view(rec)) == bytes(range(256)) * 8192
    assert rec.last_write["by"] == "peer:tubingen"


def test_crossbox_throughput_report(rig):
    """MEASURE + REPORT (no hard gate): socket-transport goodput over
    the 25 GbE link, 256 MiB payload."""
    data = bytes(256 << 20)
    rig["client"].put_object("xbox_big", data)
    t0 = time.monotonic()
    out = rig["client"].send_object("xbox_big", "tubingen")
    row = rig["client"].wait_transfer(out["send_id"], timeout=120)
    dt = time.monotonic() - t0
    assert row["state"] == "done", row
    gbps = len(data) * 8 / dt / 1e9
    print(f"\n[P2a] socket transport cross-box: 256 MiB in {dt:.2f}s "
          f"= {gbps:.1f} Gbit/s (wire ceiling 19.9 TCP / 23.1 RDMA)")
    assert gbps > 2.0, f"implausibly slow for the 25G link: {gbps}"


def test_crossbox_capacity_backpressure(rig):
    """An object bigger than tubingen's free slab NACKs CAPACITY,
    retries with backoff, then fails LOUD — nothing torn remotely."""
    rig["client"].release_object("xbox_big")  # room LOCALLY for the src
    data = bytes(300 << 20)   # fits local free; exceeds remote free
    rig["client"].put_object("xbox_toobig", data)
    out = rig["client"].send_object("xbox_toobig", "tubingen")
    row = rig["client"].wait_transfer(out["send_id"], timeout=60)
    assert row["state"] == "error"
    assert "CAPACITY" in row["error"]
    remote = tub_py(REMOTE_PRELUDE
                    + "print('xbox_toobig' in [o['id'] for o in "
                      "c.list_objects()]); c.close()")
    assert remote.strip() == "False"
    rig["client"].release_object("xbox_toobig")
