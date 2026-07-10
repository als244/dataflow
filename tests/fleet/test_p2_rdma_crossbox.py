"""P2b cross-box gates: rdma-host over the REAL 25 GbE link —
chicago mlx5_1 <-> tubingen mlx5_0, RDMA_WRITE into the remote pinned
slab. DATAFLOW_TUBINGEN=1 gated; remote daemon owned over ssh."""
import hashlib
import os
import subprocess
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("pyverbs")
if not os.environ.get("DATAFLOW_TUBINGEN"):
    pytest.skip("cross-box gates need DATAFLOW_TUBINGEN=1",
                allow_module_level=True)

from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402

pytestmark = pytest.mark.fleet

TUB = "tubingen"
TUB_PY = "~/miniconda3/envs/dataflow/bin/python"
TUB_REPO = "~/Documents/dataflow"
TUB_SOCK = "/tmp/dataflowd-p2rdma.sock"
TUB_PEER = "192.168.50.32:29610"
CHI_PEER = "192.168.50.23:29610"


def ssh(cmd: str, *, timeout: float = 60.0) -> str:
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", TUB, cmd],
                         capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"ssh rc={out.returncode}: {out.stderr[-500:]}")
    return out.stdout


def ssh_fire_and_forget(cmd: str) -> None:
    subprocess.Popen(["ssh", "-o", "BatchMode=yes", TUB, cmd],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def kill_remote_daemon() -> None:
    port = TUB_PEER.rsplit(":", 1)[1]
    ssh(f"pkill -f '[d]ataflowd.py start --socket {TUB_SOCK}' || true; "
        f"fuser -k {port}/tcp 2>/dev/null || true")


def tub_py(code: str, *, timeout: float = 120.0) -> str:
    quoted = code.replace("'", "'\"'\"'")
    return ssh(f"cd {TUB_REPO} && {TUB_PY} -c '{quoted}'", timeout=timeout)


REMOTE_PRELUDE = (
    "import sys; sys.path.insert(0, 'src'); "
    "from dataflow.service import EngineClient; "
    f"c = EngineClient('{TUB_SOCK}', client_name='p2rdma-verify'); "
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    kill_remote_daemon()
    time.sleep(1.0)
    ssh_fire_and_forget(
        f"cd {TUB_REPO} && setsid nohup {TUB_PY} tools/dataflowd.py start "
        f"--socket {TUB_SOCK} --slab-gib 0.5 --peer-name tubingen "
        f"--peer-listen {TUB_PEER} --peer-rdma-device mlx5_0 "
        f"> /tmp/dataflowd-p2rdma.log 2>&1 < /dev/null & exit")
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
                           "tubingen:/tmp/dataflowd-p2rdma.log")

    tmp = tmp_path_factory.mktemp("p2rdma")
    sock = str(tmp / "chicago.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=0.5,
        peer_name="chicago", peer_listen=CHI_PEER,
        peer_rdma_device="mlx5_1"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    client = EngineClient(sock, client_name="chicago")
    client.peer_connect("tubingen", TUB_PEER)
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
    kill_remote_daemon()


def test_rdma_crossbox_byte_identity(rig):
    data = bytes((17 * i) % 251 for i in range(32 << 20))
    rig["client"].put_object("xr_W", data)
    out = rig["client"].send_object("xr_W", "tubingen")
    row = rig["client"].wait_transfer(out["send_id"], timeout=60)
    assert row["state"] == "done", row
    remote = tub_py(REMOTE_PRELUDE
                    + "import hashlib; b = c.get_object('xr_W'); "
                      "print(len(b), hashlib.sha256(bytes(b)).hexdigest()); "
                      "c.close()")
    nbytes, sha = remote.split()
    assert int(nbytes) == len(data)
    assert sha == hashlib.sha256(data).hexdigest()


def test_rdma_crossbox_reverse(rig):
    tub_py(REMOTE_PRELUDE
           + "c.put_object('xr_back', bytes(range(251)) * 65536); "
             "r = c.send_object('xr_back', 'chicago'); "
             "print(c.wait_transfer(r['send_id'], timeout=60)['state']); "
             "c.close()")
    rec = rig["server"].store.objects.get("xr_back")
    assert rec is not None
    assert bytes(rig["server"].store.view(rec)) == bytes(range(251)) * 65536


def test_rdma_crossbox_throughput_report(rig):
    data = bytes(256 << 20)
    rig["client"].put_object("xr_big", data)
    t0 = time.monotonic()
    out = rig["client"].send_object("xr_big", "tubingen")
    row = rig["client"].wait_transfer(out["send_id"], timeout=120)
    dt = time.monotonic() - t0
    assert row["state"] == "done", row
    gbps = len(data) * 8 / dt / 1e9
    print(f"\n[P2b] rdma-host CROSS-BOX: 256 MiB in {dt:.3f}s "
          f"= {gbps:.1f} Gbit/s (ib_write_bw ceiling 23.1)")
    assert gbps > 10.0, f"rdma path implausibly slow: {gbps}"
