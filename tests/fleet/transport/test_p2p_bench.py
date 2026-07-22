"""p2p_bench verb gate: the in-engine transfer bench against a real
local daemon pair — verifies the verb's contract (lane report, acked
walls, pipelined sustained pass) on the socket lane, with walls that
are physically sane (positive, and no faster than the payload could
possibly move).

Tests:
- test_p2p_bench_local_pair: the p2p bench over a local daemon pair reports the socket lane, the requested iters and sizes, finite positive per-size walls, a sustained pass no faster than one iteration's share of the slowest wall, and the bench objects actually landed on the remote.
"""
import math
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataflow.service import EngineClient  # noqa: E402
from dataflow_training.distributed import daemons  # noqa: E402
from dataflow_training.distributed.topology import local_pair_topology  # noqa: E402

pytestmark = [pytest.mark.fleet, pytest.mark.gpu]

PORTS = (29571, 29572)
LANES = ("pqa", "pqb")
SIZES = (4096, 1 << 20, 8 << 20)
ITERS = 3


def test_p2p_bench_local_pair():
    topo = local_pair_topology(ports=PORTS)
    ranks = [topo.host("local0"), topo.host("local1")]
    clients = []
    try:
        for host, lane in zip(ranks, LANES):
            daemons.kill(host, lane=lane)
            daemons.launch(
                host, lane=lane, backing_gib=0.5,
                peer_port=int(host.peer_listen.rsplit(":", 1)[1]))
        deadline = time.time() + 120
        while time.time() < deadline and len(clients) < 2:
            try:
                socks = [daemons.paths(h, ln)["sock"]
                         for h, ln in zip(ranks, LANES)]
                for s in socks:
                    probe = EngineClient(s, client_name="probe")
                    probe.health()
                    probe.close()
                clients = [EngineClient(s, client_name=h.name)
                           for s, h in zip(socks, ranks)]
            except Exception:
                time.sleep(1.0)
        assert len(clients) == 2, "local pair daemons unreachable"
        clients[0].peer_connect("local1", ranks[1].peer_listen)
        time.sleep(0.5)

        out = clients[0].p2p_bench("local1", SIZES, iters=ITERS)
        assert out["lane"] == "socket"
        assert out["iters"] == ITERS
        assert [r["bytes"] for r in out["rows"]] == list(SIZES)
        for row in out["rows"]:
            assert len(row["walls_s"]) == ITERS
            for w in row["walls_s"]:
                assert math.isfinite(w) and w > 0.0, (row["bytes"], w)
            # the pipelined sustained pass runs ITERS transfers: it must
            # be positive and cannot beat one iteration's share of the
            # slowest single wall — a structural consistency check, not
            # a bandwidth claim
            assert row["sustained_s"] > 0.0
            assert row["sustained_s"] >= max(row["walls_s"]) / ITERS
        # the transfers really landed: remote holds the bench objects
        remote_ids = {o["id"] for o in clients[1].list_objects()}
        assert f"p2pbench_{SIZES[-1]}" in remote_ids
    finally:
        for c in clients:
            try:
                c.shutdown()
            except Exception:
                pass
        for host, lane in zip(ranks, LANES):
            try:
                daemons.kill(host, lane=lane)
            except Exception:
                pass
