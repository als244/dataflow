"""Cross-box collective DTYPE gate: verified allreduce for every
dtype the training paths ship (bf16 grads/params, fp32 tensor-parallel
partials) on the topology-default lane of each backend.

Born from the fp32-partials incident: the tp fp32 activation
allreduce produced NaN training on the crossbox hostmem lane (and a
wedge on nccl) while the same-code loopback gate — socket lane,
in-process — was green and bitwise. Collective correctness must be
gated per (backend, dtype) on the REAL wire, not inferred from one
dtype's success.

Tests:
- test_verified_allreduce_dtype: for each backend (hostmem, nccl) and dtype (bf16, fp32), a self-verifying allreduce over the real cross-box wire reports verified on both the local and remote ranks.
"""
import json
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataflow_training.distributed.hosts import run_on, run_py, uds_forward
from dataflow_training.distributed import daemons
from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402
from dataflow.service import EngineClient  # noqa: E402

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("needs a topology.toml with a remote host",
                allow_module_level=True)

pytestmark = [pytest.mark.fleet, pytest.mark.gpu]

LOCAL = TOPO.local()
REMOTE = TOPO.remotes()[0]
LANE = "dtx"
PORT = 29665


class LocalBench:
    def __init__(self, client, args):
        self.client = client
        self.args = args
        self.out = None
        self.err = None

    def __call__(self):
        try:
            self.out = self.client._call("coll_bench", self.args,
                                         timeout=180)
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
                f"c = EngineClient('{self.sock}', client_name='db'); "
                f"r = c._call('coll_bench', {self.args!r}, timeout=180); "
                "print(json.dumps(r)); c.close()")
            self.out = json.loads(run_py(REMOTE, code, timeout=240))
        except Exception as e:
            self.err = e


def run_both(client, remote_sock, args) -> tuple:
    a = LocalBench(client, args)
    b = RemoteBench(remote_sock, args)
    ta = threading.Thread(target=a)
    tb = threading.Thread(target=b)
    ta.start(); tb.start(); ta.join(300); tb.join(300)
    assert a.err is None, a.err
    assert b.err is None, b.err
    return a.out, b.out


@pytest.fixture(scope="module",
                params=["hostmem",
                        pytest.param("nccl", marks=pytest.mark.ncclbind)])
def rig(request, tmp_path_factory):
    backend = request.param
    # a configured-but-unreachable peer degrades to a clean skip
    try:
        run_on(REMOTE, "true", timeout=15)
    except Exception as exc:
        pytest.skip(f"remote host unreachable: {exc}")
    # the nccl lane also needs libnccl on the REMOTE member (the local
    # side is gated by the ncclbind marker on the param)
    if backend == "nccl":
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
    tmp = tmp_path_factory.mktemp(LANE)
    fwd_sock = str(tmp / "r.sock")
    fwd = uds_forward(REMOTE, remote_sock, fwd_sock)
    deadline = time.time() + 120
    client = None
    while time.time() < deadline:
        try:
            for sock in (daemons.paths(LOCAL, LANE)["sock"], fwd_sock):
                probe = EngineClient(sock, client_name="probe")
                probe.health()
                probe.close()
            client = EngineClient(daemons.paths(LOCAL, LANE)["sock"],
                                  client_name="dtx")
            break
        except Exception:
            time.sleep(1.0)
    assert client is not None, "daemons unreachable"
    client.peer_connect(REMOTE.name, REMOTE.peer_addr(PORT))
    time.sleep(1.0)
    out = client._call("create_peer_group",
                       {"name": "dt", "members": [LOCAL.name, REMOTE.name],
                        "backend": backend}, timeout=120)
    yield {"client": client, "remote_sock": remote_sock,
           "backend": out["backend"]}
    try:
        client.shutdown()
    except Exception:
        pass
    if fwd is not None:
        fwd.terminate()
    for host in (LOCAL, REMOTE):
        daemons.kill(host, lane=LANE)


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_verified_allreduce_dtype(rig, dtype):
    args = {"group": "dt", "sizes": [1 << 20], "dtype": dtype,
            "reps": 2, "verify": True}
    la, ra = run_both(rig["client"], rig["remote_sock"], args)
    for side, r in (("local", la), ("remote", ra)):
        assert r["verified"] is True, (rig["backend"], dtype, side, r)
    print(f"\n[coll-dtypes] {rig['backend']}/{dtype}: verified on "
          f"both ranks (lane {la['lane']})")
