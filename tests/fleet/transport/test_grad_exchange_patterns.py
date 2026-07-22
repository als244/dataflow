"""Grad-exchange pattern benches: replay the ACTUAL collective
patterns a data-parallel run drives — one layer's gradient fields, and
the embed+head tail — through real cross-box daemons, on both data
planes.

Purpose (in order): (1) confirm the bench reproduces the per-field
socket-lane idle-gap shape a past profile attributed to the per-layer
and head/embed exchanges, (2) break the time down via the comm's phase
stats, (3) gate the fixed path (rdma lane + fused per-layer exchange)
against the PROBED wire bandwidth.

Sizes are the real grad-field bytes (bf16): 2 norms + wq/wk/wv/wo +
w1/w3/w2 = 121.6 MB/layer; embed = head = 206 MB (untied).

Tests:
- test_per_field_socket_exchange_within_wall_bounds: per-field grad exchange on the socket lane (one layer, then head+embed) rides the socket lane, not rdma, and finishes within generous liveness wall bounds.
- test_rdma_fused_layer_hits_wire_floor: one fused per-layer exchange rides the rdma lane and its steady-state wall stays near the probed wire floor.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataflow_training.distributed.hosts import run_on, run_py
from dataflow_training.distributed import daemons
from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("pattern benches need a topology.toml with a remote "
                "host", allow_module_level=True)

pytestmark = [pytest.mark.fleet, pytest.mark.gpu]

LOCAL = TOPO.local()
REMOTE = TOPO.remotes()[0]

# real l3_1b grad-field bytes (bf16), in layout order
LAYER_FIELDS = [4096,                     # attn_norm
                8388608, 2097152, 2097152, 8388608,   # wq wk wv wo
                4096,                     # ffn_norm
                33554432, 33554432, 33554432]         # w1 w3 w2
LAYER_FUSED = [sum(LAYER_FIELDS)]         # one contiguous exchange
HEAD_EMBED = [206045184, 4096, 206045184]  # head, final_norm, embed
REPS = 4


class BenchCall:
    def __init__(self, client, group, sizes, reps=REPS):
        self.client = client
        self.group = group
        self.sizes = sizes
        self.reps = reps
        self.out = None
        self.err = None

    def __call__(self):
        try:
            self.out = self.client._call(
                "coll_bench", {"group": self.group, "sizes": self.sizes,
                               "dtype": "bf16", "reps": self.reps})
        except Exception as e:
            self.err = e


class RemoteBench:
    def __init__(self, sock, group, sizes, reps=REPS):
        self.sock = sock
        self.group = group
        self.sizes = sizes
        self.reps = reps
        self.out = None
        self.err = None

    def __call__(self):
        try:
            code = (
                "import sys, json; sys.path.insert(0, 'src'); "
                "from dataflow.service import EngineClient; "
                f"c = EngineClient('{self.sock}', client_name='bench'); "
                f"r = c._call('coll_bench', {{'group': '{self.group}', "
                f"'sizes': {self.sizes}, 'dtype': 'bf16', "
                f"'reps': {self.reps}}}); "
                "print(json.dumps(r)); c.close()")
            import json

            self.out = json.loads(run_py(REMOTE, code, timeout=300))
        except Exception as e:
            self.err = e


def run_pattern(rig, sizes, reps=REPS) -> dict:
    """Both ranks post the pattern concurrently; returns both sides'
    results keyed local/remote."""
    local = BenchCall(rig["client"], rig["group"], sizes, reps)
    remote = RemoteBench(rig["remote_sock"], rig["group"], sizes, reps)
    ta = threading.Thread(target=local)
    tb = threading.Thread(target=remote)
    ta.start(); tb.start(); ta.join(400); tb.join(400)
    assert local.err is None, local.err
    assert remote.err is None, remote.err
    return {"local": local.out, "remote": remote.out}


def build_rig(tmp_path_factory, lane, port, rdma: bool):
    # a configured-but-unreachable peer degrades to a clean skip
    try:
        run_on(REMOTE, "true", timeout=15)
    except Exception as exc:
        pytest.skip(f"remote host unreachable: {exc}")
    daemons.kill(REMOTE, lane=lane)
    extra = ""
    if rdma:
        extra = f"--peer-rdma-device {REMOTE.ib_dev}"
    daemons.launch(REMOTE, lane=lane, backing_gib=4.0, peer_port=port,
                  extra_flags=extra)
    remote_sock = daemons.paths(REMOTE, lane)["sock"]
    prelude = (
        "import sys; sys.path.insert(0, 'src'); "
        "from dataflow.service import EngineClient; "
        f"c = EngineClient('{remote_sock}', client_name='probe'); ")
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            run_py(REMOTE, prelude + "print(c.health()['ok']); c.close()",
                   timeout=20)
            break
        except Exception:
            time.sleep(1.0)
    else:
        raise RuntimeError(f"{REMOTE.name} daemon did not come up "
                           f"(lane {lane})")
    tmp = tmp_path_factory.mktemp(lane)
    sock = str(tmp / "local.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=4.0,
        peer_name=LOCAL.name, peer_listen=LOCAL.peer_addr(port),
        peer_rdma_device=LOCAL.ib_dev if rdma else None))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    client = EngineClient(sock, client_name=LOCAL.name)
    client.peer_connect(REMOTE.name, REMOTE.peer_addr(port))
    if rdma:
        deadline = time.time() + 15
        while time.time() < deadline:
            if any(e.get("event") == "peer_rdma_up"
                   for e in server.state.events):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("RC QPs never reached RTS")
    group = f"gx-{lane}"
    client._call("create_peer_group",
                 {"name": group, "members": [LOCAL.name, REMOTE.name],
                  "backend": "hostmem"})
    return {"server": server, "client": client, "group": group,
            "remote_sock": remote_sock, "lane": lane}


@pytest.fixture(scope="module")
def sock_rig(tmp_path_factory):
    rig = build_rig(tmp_path_factory, "gxs", 29620, rdma=False)
    yield rig
    try:
        rig["client"].shutdown()
    except Exception:
        pass
    daemons.kill(REMOTE, lane="gxs")


@pytest.fixture(scope="module")
def rdma_rig(tmp_path_factory):
    pytest.importorskip("pyverbs")
    from dataflow.service.peer.rdma import roce_v2_ipv4_gid
    if LOCAL.ib_dev is None or REMOTE.ib_dev is None:
        pytest.skip("rdma pattern bench needs ib_dev on both hosts")
    if roce_v2_ipv4_gid(LOCAL.ib_dev) is None:
        pytest.skip(f"no ACTIVE RoCE v2 GID on {LOCAL.ib_dev}")
    remote_probe = (
        "import sys; sys.path.insert(0, 'src'); import pyverbs; "
        "from dataflow.service.peer.rdma import roce_v2_ipv4_gid; "
        f"print(roce_v2_ipv4_gid({REMOTE.ib_dev!r}) is not None)")
    try:
        remote_ok = run_py(REMOTE, remote_probe, timeout=30).strip() == "True"
    except Exception as exc:
        pytest.skip(f"remote rdma probe failed: {exc}")
    if not remote_ok:
        pytest.skip("remote host lacks pyverbs or an ACTIVE RoCE v2 GID")
    rig = build_rig(tmp_path_factory, "gxr", 29630, rdma=True)
    yield rig
    try:
        rig["client"].shutdown()
    except Exception:
        pass
    daemons.kill(REMOTE, lane="gxr")


def report(tag, res):
    for side in ("local", "remote"):
        r = res[side]
        walls = ", ".join(f"{w*1e3:.0f}" for w in r["walls_s"])
        print(f"[{tag}] {side:6s} walls ms: [{walls}]  "
              f"rdma_lane={r['rdma_lane']}  stats={r['stats']}")


def steady_ms(res) -> float:
    """Steady-state wall (max across ranks, first rep dropped —
    it absorbs pool warm-up and skew)."""
    lo = res["local"]["walls_s"][1:]
    re = res["remote"]["walls_s"][1:]
    return max(min(lo), min(re)) * 1e3


def test_per_field_socket_exchange_within_wall_bounds(sock_rig):
    """The pre-fix path (socket lane, per-field posts) walks the same
    per-field exchange a past profile on one fabric associated with the
    training idle gaps. Kept as a representativeness check on the bench;
    the wall bounds below are generous liveness guards, not machine
    claims."""
    layer = run_pattern(sock_rig, LAYER_FIELDS)
    report("socket layer 9-field", layer)
    head = run_pattern(sock_rig, HEAD_EMBED)
    report("socket head+embed", head)
    lms, hms = steady_ms(layer), steady_ms(head)
    print(f"[socket-exchange] layer {lms:.0f} ms, head {hms:.0f} ms")
    assert not layer["local"]["rdma_lane"]
    # generous liveness guards (not perf claims): a hung exchange trips
    # these, a merely slow fabric does not
    assert lms < 30000 and hms < 60000


def test_rdma_fused_layer_hits_wire_floor(rdma_rig):
    """The FIXED path: rdma lane + one fused exchange per layer must
    sit near the probed wire floor — the zero-copy gate on the actual
    training pattern."""
    link = rdma_rig["server"].nm.links[REMOTE.name]
    deadline = time.time() + 15
    while "rdma" not in link.peak_gbps and time.time() < deadline:
        time.sleep(0.1)
    peak = link.peak_gbps["rdma"]
    fused = run_pattern(rdma_rig, LAYER_FUSED)
    report("rdma layer fused", fused)
    per_field = run_pattern(rdma_rig, LAYER_FIELDS)
    report("rdma layer 9-field", per_field)
    head = run_pattern(rdma_rig, HEAD_EMBED)
    report("rdma head+embed", head)
    assert fused["local"]["rdma_lane"]
    wire_ms = LAYER_FUSED[0] * 8 / (peak * 1e9) * 1e3
    fms = steady_ms(fused)
    print(f"[rdma fused] layer {fms:.0f} ms vs wire {wire_ms:.0f} ms "
          f"at probed {peak} Gbit/s; per-field {steady_ms(per_field):.0f} "
          f"ms; head {steady_ms(head):.0f} ms")
    assert fms <= wire_ms * 1.9 + 25, (
        f"fused layer exchange {fms:.0f} ms vs wire floor "
        f"{wire_ms:.0f} ms — copies or stalls crept in")
