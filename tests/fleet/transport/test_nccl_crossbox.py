"""The NCCL group backend across a real host pair — bootstrap
(uniqueId over GROUP_JOIN, blocking collective init + warm-up proof
at creation), verified allreduce, the rs+ag==allreduce identity at
the GroupHandle surface, and a fused-layer timing report. Hosts from
topology.toml; both sides launched daemons (NCCL env rides
daemons.launch from the topology's iface/ib_dev).

Tests:
- test_nccl_bootstrap_and_verified_collectives: creating an nccl peer group bootstraps the comm, and on both ranks allreduce verifies and the rs+ag==allreduce identity holds.
- test_nccl_fused_layer_allreduce_within_plausible_wall: a fused single-layer bf16 allreduce verifies on both ranks and its steady-state wall stays under a liveness ceiling (self-creating the group if the bootstrap test did not run first).
"""
import json
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataflow_training.distributed.hosts import run_py, uds_forward
from dataflow_training.distributed import daemons
from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402
from dataflow.service import EngineClient  # noqa: E402
from dataflow.service.wire import ServiceError  # noqa: E402

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("nccl cross-box gates need a topology.toml with a "
                "remote host", allow_module_level=True)

pytestmark = [pytest.mark.fleet, pytest.mark.gpu, pytest.mark.ncclbind]

LOCAL = TOPO.local()
REMOTE = TOPO.remotes()[0]
LANE = "ncclx"
PORT = 29640
GROUP = "ndp"
LAYER_FUSED = 121634816          # one l3_1b layer's grads, bf16


class LocalBench:
    def __init__(self, client, args):
        self.client = client
        self.args = args
        self.out = None
        self.err = None

    def __call__(self):
        try:
            self.out = self.client._call("coll_bench", self.args,
                                         timeout=300)
        except Exception as e:
            self.err = e


class RemoteBench:
    def __init__(self, sock, args):
        self.sock = sock
        self.args = args
        self.out = None
        self.err = None

    def __call__(self):
        try:
            code = (
                "import sys, json; sys.path.insert(0, 'src'); "
                "from dataflow.service import EngineClient; "
                f"c = EngineClient('{self.sock}', client_name='nb'); "
                f"r = c._call('coll_bench', {self.args!r}, timeout=300); "
                "print(json.dumps(r)); c.close()")
            self.out = json.loads(run_py(REMOTE, code, timeout=360))
        except Exception as e:
            self.err = e


def run_both(client, remote_sock, args) -> tuple:
    a = LocalBench(client, args)
    b = RemoteBench(remote_sock, args)
    ta = threading.Thread(target=a)
    tb = threading.Thread(target=b)
    ta.start(); tb.start(); ta.join(400); tb.join(400)
    assert a.err is None, a.err
    assert b.err is None, b.err
    return a.out, b.out


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    # the nccl lane needs libnccl on the REMOTE member too (the local
    # side is gated by the ncclbind marker); an unreachable or nccl-less
    # peer degrades to a clean skip
    probe = ("import sys; sys.path.insert(0, 'src'); "
             "from dataflow.service.peer import nccl; "
             "print(nccl.available())")
    try:
        remote_ok = run_py(REMOTE, probe, timeout=30).strip() == "True"
    except Exception as exc:
        pytest.skip(f"remote nccl probe failed: {exc}")
    if not remote_ok:
        pytest.skip("libnccl unavailable on the remote host")
    for host in (LOCAL, REMOTE):
        daemons.kill(host, lane=LANE)
        daemons.launch(host, lane=LANE, backing_gib=4.0, peer_port=PORT)
    remote_sock = daemons.paths(REMOTE, LANE)["sock"]
    local_sock = daemons.paths(LOCAL, LANE)["sock"]
    tmp = tmp_path_factory.mktemp(LANE)
    fwd_sock = str(tmp / "remote.sock")
    fwd = uds_forward(REMOTE, remote_sock, fwd_sock)
    deadline = time.time() + 120
    client = None
    while time.time() < deadline:
        try:
            probe = EngineClient(local_sock, client_name="probe")
            probe.health()
            probe.close()
            probe = EngineClient(fwd_sock, client_name="probe")
            probe.health()
            probe.close()
            client = EngineClient(local_sock, client_name="nccl-gate")
            break
        except Exception:
            time.sleep(1.0)
    assert client is not None, "daemons unreachable"
    client.peer_connect(REMOTE.name, REMOTE.peer_addr(PORT))
    time.sleep(1.0)
    yield {"client": client, "remote_sock": remote_sock,
           "fwd_sock": fwd_sock}
    try:
        client.shutdown()
    except Exception:
        pass
    if fwd is not None:
        fwd.terminate()
    for host in (LOCAL, REMOTE):
        daemons.kill(host, lane=LANE)


def test_nccl_bootstrap_and_verified_collectives(rig):
    client = rig["client"]
    out = client._call(
        "create_peer_group",
        {"name": GROUP, "members": [LOCAL.name, REMOTE.name],
         "backend": "nccl"}, timeout=120)
    assert out["backend"] == "nccl", out

    args = {"group": GROUP, "sizes": [1 << 20], "dtype": "bf16",
            "reps": 2, "verify": True,
            "rs_ag_identity": 1 << 20}
    la, ra = run_both(client, rig["remote_sock"], args)
    for side, r in (("local", la), ("remote", ra)):
        assert r["lane"] == "nccl", (side, r["lane"])
        assert r["verified"] is True
        assert r["rs_ag_ok"] is True, side
    print(f"\n[nccl] bootstrap + verified allreduce + rs/ag identity "
          f"OK on both ranks")


def test_nccl_fused_layer_allreduce_within_plausible_wall(rig):
    client = rig["client"]
    # self-sufficient under -k: create the nccl group if the bootstrap
    # test did not run first (create_peer_group is not idempotent)
    try:
        client._call(
            "create_peer_group",
            {"name": GROUP, "members": [LOCAL.name, REMOTE.name],
             "backend": "nccl"}, timeout=120)
    except ServiceError as exc:
        if exc.code != "GROUP_EXISTS":
            raise
    args = {"group": GROUP, "sizes": [LAYER_FUSED], "dtype": "bf16",
            "reps": 4, "verify": True}
    la, ra = run_both(client, rig["remote_sock"], args)
    for side, r in (("local", la), ("remote", ra)):
        assert r["verified"] is True, (side, r)
    steady = min(la["walls_s"][1:])
    gbps = LAYER_FUSED * 8 / steady / 1e9
    print(f"\n[nccl] fused layer {LAYER_FUSED/1e6:.0f} MB: "
          f"steady {steady*1e3:.0f} ms = {gbps:.1f} Gbit/s "
          f"(walls {la['walls_s']})")
    # liveness guard, not a perf claim: a healthy fabric completes far
    # sooner; this trips only on a hang or wedge
    assert steady < 30.0
