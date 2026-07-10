"""P3b gates: hostmem collectives between two REAL daemons sharing this
GPU — device tensors staged through pinned scratch, exchanged over the
peer link, fp32 rank-ordered reduction, cuStreamWaitValue32 stream
choreography. Correctness vs single-process fp32 references; both
sides must post (collectives are collective) so ops run from two
threads."""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402

pytestmark = pytest.mark.fleet

PA, PB = 29501, 29502


def boot(tmp, name, peer_port):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=0.25,
        peer_name=name, peer_listen=f"127.0.0.1:{peer_port}"))
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
    tmp = tmp_path_factory.mktemp("p3")
    sa, ca = boot(tmp, "hm-a", PA)
    sb, cb = boot(tmp, "hm-b", PB)
    ca.peer_connect("hm-b", f"127.0.0.1:{PB}")
    ca._call("create_peer_group",
             {"name": "dp", "members": ["hm-a", "hm-b"],
              "backend": "hostmem"})
    ha = sa.nm.group_handles()["dp"]
    hb = sb.nm.group_handles()["dp"]
    assert ha.comm is not None and hb.comm is not None
    assert ha.rank == 0 and hb.rank == 1
    yield {"ha": ha, "hb": hb, "sa": sa, "sb": sb, "ca": ca, "cb": cb}
    for c in (ca, cb):
        try:
            c.shutdown()
        except Exception:
            pass


def both(fn_a, fn_b):
    """Post both ranks' halves of a collective concurrently."""
    err = []
    def run(fn):
        try:
            fn()
        except Exception as e:
            err.append(e)
    ta = threading.Thread(target=run, args=(fn_a,))
    tb = threading.Thread(target=run, args=(fn_b,))
    ta.start(); tb.start(); ta.join(30); tb.join(30)
    assert not err, err


def synced(*handles):
    for h in handles:
        h.stream.synchronize()


def test_allreduce_matches_fp32_reference_and_is_replica_identical(rig):
    ha, hb = rig["ha"], rig["hb"]
    g = torch.Generator(device="cuda").manual_seed(5)
    a = torch.randn(1 << 20, device="cuda", generator=g,
                    dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(1 << 20, device="cuda", generator=g,
                    dtype=torch.float32).to(torch.bfloat16)
    want = (a.float() + b.float()).to(torch.bfloat16)
    ta, tb = a.clone(), b.clone()
    both(AllreduceCall(ha, ta), AllreduceCall(hb, tb))
    synced(ha, hb)
    assert torch.equal(ta, tb), "replicas diverged"
    assert torch.equal(ta, want), "sum wrong vs fp32 reference"


class AllreduceCall:
    def __init__(self, h, t):
        self.h, self.t = h, t

    def __call__(self):
        self.h.allreduce(self.t)


class BroadcastCall:
    def __init__(self, h, t, root):
        self.h, self.t, self.root = h, t, root

    def __call__(self):
        self.h.broadcast(self.t, self.root)


class RsCall:
    def __init__(self, h, full, out):
        self.h, self.full, self.out = h, full, out

    def __call__(self):
        self.h.reduce_scatter(self.full, self.out)


class AgCall:
    def __init__(self, h, mine, full):
        self.h, self.mine, self.full = h, mine, full

    def __call__(self):
        self.h.all_gather(self.mine, self.full)


def test_broadcast_root_bytes_land_everywhere(rig):
    ha, hb = rig["ha"], rig["hb"]
    src = torch.arange(4096, device="cuda", dtype=torch.int32)
    dst = torch.zeros(4096, device="cuda", dtype=torch.int32)
    both(BroadcastCall(ha, src, 0), BroadcastCall(hb, dst, 0))
    synced(ha, hb)
    assert torch.equal(dst, src)


def test_reduce_scatter_then_all_gather_roundtrip(rig):
    ha, hb = rig["ha"], rig["hb"]
    g = torch.Generator(device="cuda").manual_seed(9)
    fa = torch.randn(1 << 18, device="cuda", generator=g,
                     dtype=torch.float32).to(torch.bfloat16)
    fb = torch.randn(1 << 18, device="cuda", generator=g,
                     dtype=torch.float32).to(torch.bfloat16)
    want = (fa.float() + fb.float()).to(torch.bfloat16)
    half = fa.numel() // 2
    oa = torch.empty(half, device="cuda", dtype=torch.bfloat16)
    ob = torch.empty(half, device="cuda", dtype=torch.bfloat16)
    both(RsCall(ha, fa.clone(), oa), RsCall(hb, fb.clone(), ob))
    synced(ha, hb)
    assert torch.equal(oa, want[:half])       # rank 0 owns the low half
    assert torch.equal(ob, want[half:])
    ga = torch.empty(2 * half, device="cuda", dtype=torch.bfloat16)
    gb = torch.empty(2 * half, device="cuda", dtype=torch.bfloat16)
    both(AgCall(ha, oa, ga), AgCall(hb, ob, gb))
    synced(ha, hb)
    assert torch.equal(ga, want) and torch.equal(gb, want)
    # the ZeRO identity: rs + ag == allreduce, bit-for-bit at world 2
    ra, rb = fa.clone(), fb.clone()
    both(AllreduceCall(ha, ra), AllreduceCall(hb, rb))
    synced(ha, hb)
    assert torch.equal(ga, ra)


def test_back_to_back_ops_fifo(rig):
    ha, hb = rig["ha"], rig["hb"]
    xs_a = [torch.full((256,), float(i), device="cuda",
                       dtype=torch.bfloat16) for i in range(4)]
    xs_b = [torch.full((256,), float(10 * i), device="cuda",
                       dtype=torch.bfloat16) for i in range(4)]
    def post_a():
        for t in xs_a:
            ha.allreduce(t)
    def post_b():
        for t in xs_b:
            hb.allreduce(t)
    both(post_a, post_b)
    synced(ha, hb)
    for i in range(4):
        want = float(i) + 10.0 * i
        assert float(xs_a[i][0]) == want == float(xs_b[i][0])


def test_stream_parks_until_worker_releases(rig):
    """The async proof: enqueue-only caller; the group stream is parked
    on the flag until the exchange lands (needs wait-value support)."""
    ha, hb = rig["ha"], rig["hb"]
    if not (ha.comm.wait_value_ok and hb.comm.wait_value_ok):
        pytest.skip("wait-value unsupported; fallback path is blocking")
    t_a = torch.ones(1 << 22, device="cuda", dtype=torch.bfloat16)
    ha.allreduce(t_a)                        # only rank 0 posts...
    time.sleep(0.2)
    assert not ha.stream.query(), "stream should be parked mid-collective"
    t_b = torch.ones(1 << 22, device="cuda", dtype=torch.bfloat16)
    hb.allreduce(t_b)                        # ...until rank 1 joins
    synced(ha, hb)
    assert float(t_a[0]) == 2.0 == float(t_b[0])
