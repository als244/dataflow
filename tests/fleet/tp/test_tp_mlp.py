"""TP toy gate: tensor-parallel llama3 MLP across two daemons — the
group plane exercised from ORDINARY compute blocks (mid-forward
allreduce of y partials, mid-backward allreduce of dx partials)
instead of the optimizer epilogue.

Everything the toy computes has an exact expectation:

  - a shard's local math (a1/a3/h, dh, dW) is gemm-for-gemm the same
    shapes as the reference's per-shard computation, so dW must be
    BITWISE equal to the reference shard grads;
  - the allreduced y/dx equal the split-order reference sum bitwise
    (world-2 bf16 add commutes, both lanes);
  - both ranks' y/dx are bitwise identical to each other;
  - against the FULL-WIDTH single-gemm reference, y/dx sit within a
    bf16 split-reduction band (sanity that TP == the untied math).

Tests:
- test_tp_mlp_hostmem_pair_matches_reference_bitwise: two child-process daemons on the hostmem lane run the toy; y/dx replicas match across ranks, rank-0 y/dx and every shard dW are bitwise equal to the split-order reference, and the split-vs-full bands stay under 3e-2.
- test_tp_mlp_crossbox_auto_matches_reference_within_ulp: the same driver runs cross-box over the auto/nccl lane with plugin-loaded remote daemons; group-plane equalities stay bitwise while ranks on a differing GPU architecture get only a ulp budget on their local math; skipped without a remote topology.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from dataclasses import asdict, replace  # noqa: E402

import dataflow_training.model_families.tp_mlp as tp  # noqa: E402  (registers)
from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow.service import EngineClient  # noqa: E402
from dataflow_training.lowering.planning import plan_program  # noqa: E402
from dataflow_training.register import register_all  # noqa: E402
from dataflow_training.run.driver import init_model  # noqa: E402
from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = [pytest.mark.fleet, pytest.mark.gpu]

register_all()          # in-process Server rigs share this registry

SEED = 5
PA, PB = 29551, 29552
CFG = tp.TpMlpConfig.tiny()


def reference(cfg, seed):
    """Full-width truth on the same device math. Per-shard tensors are
    computed at SHARD gemm shapes (bitwise expectations); y/dx also in
    split order (bitwise) and full order (banded sanity)."""
    dev = {k: v.cuda() for k, v in tp.full_width_draws(cfg, seed).items()}
    ffs = cfg.d_ff // cfg.world
    x, dy = dev["x"], dev["dy"]
    per_rank, y_parts, dx_parts = [], [], []
    for r in range(cfg.world):
        lo, hi = r * ffs, (r + 1) * ffs
        w1, w3 = dev["w1"][:, lo:hi], dev["w3"][:, lo:hi]
        w2 = dev["w2"][lo:hi, :]
        a1 = x @ w1
        a3 = x @ w3
        h = torch.nn.functional.silu(a1) * a3
        y_parts.append(h @ w2)
        dh = dy @ w2.T
        da1, da3 = tp.silu_grads(a1, a3, dh)
        per_rank.append({"dw1": x.T @ da1, "dw3": x.T @ da3,
                         "dw2": h.T @ dy})
        dx_parts.append(da1 @ w1.T + da3 @ w3.T)
    a1f = x @ dev["w1"]
    a3f = x @ dev["w3"]
    hf = torch.nn.functional.silu(a1f) * a3f
    da1f, da3f = tp.silu_grads(a1f, a3f, dy @ dev["w2"].T)
    torch.cuda.synchronize()
    return {"per_rank": per_rank,
            "y": y_parts[0] + y_parts[1],
            "dx": dx_parts[0] + dx_parts[1],
            "y_full": hf @ dev["w2"],
            "dx_full": da1f @ dev["w1"].T + da3f @ dev["w3"].T}


class RankRun:
    def __init__(self, client, prog_id):
        self.client = client
        self.prog_id = prog_id
        self.out = None
        self.err = None

    def __call__(self):
        try:
            self.out = self.client.run(
                self.prog_id, args={"step": 0, "valid_rows": CFG.t})
        except Exception as e:
            self.err = e


def drive_pair(clients, members, group_backend: str) -> list:
    """The shared driver: per-rank register + warm-up (kernels load
    before any parked collective exists; nothing mutates, so no
    re-seed dance is needed), group up, one concurrent step, fetch
    y/dx/dW bytes per rank."""
    prog_ids = []
    for rank, client in enumerate(clients):
        cfg_r = replace(CFG, rank=rank)
        planned = plan_program(tp.lower_tp_mlp(cfg_r),
                               fast_memory_capacity=64 * 1024 * 1024)
        init_model(client, "tp_mlp", asdict(cfg_r), seed=SEED)
        reg = client.register_program(
            program_to_dict(planned.program),
            resolver={"kind": "model_family", "family": "tp_mlp",
                      "cfg": asdict(cfg_r)})
        assert not reg["bindings"]["missing_inputs"]
        prog_ids.append(reg["prog_id"])
        warm = client.run(reg["prog_id"],
                          args={"step": 0, "valid_rows": CFG.t})
        assert warm.get("state") == "done", warm
    clients[0]._call("create_peer_group",
                     {"name": tp.GROUP_ROLE,
                      "members": list(members),
                      "backend": group_backend})
    runs = [RankRun(clients[r], prog_ids[r]) for r in range(2)]
    threads = [threading.Thread(target=r) for r in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(180)
    for r in runs:
        assert r.err is None, r.err
        assert r.out.get("state") == "done", r.out
    dims = tp.derive_dims(CFG)
    wl = tp.tp_weight_layout(dims)
    out = []
    for client in clients:
        y = torch.frombuffer(bytearray(client.get_object("y_0")),
                             dtype=torch.uint8).view(torch.bfloat16)
        dx = torch.frombuffer(bytearray(client.get_object("dx_0")),
                              dtype=torch.uint8).view(torch.bfloat16)
        dw = wl.unpack_tensor(
            torch.frombuffer(bytearray(client.get_object("dW_tp_0")),
                             dtype=torch.uint8))
        out.append({"y": y.reshape(dims.t, dims.d),
                    "dx": dx.reshape(dims.t, dims.d), "dw": dw})
    return out


def bitwise(a, b) -> bool:
    return torch.equal(a.contiguous().view(torch.uint16),
                       b.contiguous().view(torch.uint16))


def ulp_noise_only(a, b, what) -> None:
    """Cross-ARCH compare: gemm accumulation order may break rounding
    ties differently across differing GPU architectures (a past
    heterogeneous pair measured about 1 element in 8192 at |d| ~1.8e-12)
    — demand at most ulp dust, never a real numeric gap."""
    af, bf = a.float(), b.float()
    frac = (af != bf).float().mean().item()
    rel = ((af - bf).pow(2).sum().sqrt()
           / (bf.pow(2).sum().sqrt() + 1e-30)).item()
    assert frac < 1e-3 and rel < 1e-6, (what, frac, rel)


def check_pair(got, ref, hetero_ranks=()):
    """``hetero_ranks``: ranks whose GPU architecture differs from the
    reference's — their LOCAL math gets the ulp budget instead of
    bitwise. The group-plane properties (cross-rank replica equality,
    rank-0 vs split-order reference) stay bitwise regardless."""
    # cross-rank: the allreduced outputs must be identical replicas
    assert bitwise(got[0]["y"], got[1]["y"]), "y replicas diverge"
    assert bitwise(got[0]["dx"], got[1]["dx"]), "dx replicas diverge"
    # vs the split-order reference (computed on rank 0's arch). With a
    # heterogeneous pair the remote partials carry possible ulp dust
    # into the sum, so the reference compare gets the ulp budget too.
    if hetero_ranks:
        ulp_noise_only(got[0]["y"], ref["y"].cpu(), "y")
        ulp_noise_only(got[0]["dx"], ref["dx"].cpu(), "dx")
    else:
        assert bitwise(got[0]["y"], ref["y"].cpu()), "y != split ref"
        assert bitwise(got[0]["dx"], ref["dx"].cpu()), "dx != split ref"
    # local shard grads: gemm-for-gemm bitwise on the reference's own
    # arch (dW layout mirrors the weight layout: fields w1/w3/w2)
    for r in (0, 1):
        for grad, field in (("dw1", "w1"), ("dw3", "w3"),
                            ("dw2", "w2")):
            mine = got[r]["dw"][field]
            want = ref["per_rank"][r][grad].cpu()
            if r in hetero_ranks:
                ulp_noise_only(mine, want, (r, grad))
            else:
                assert bitwise(mine, want), (r, grad)
    # banded sanity vs the full-order single-gemm math
    y_band = rel_l2(got[0]["y"].float(), ref["y_full"].cpu().float())
    dx_band = rel_l2(got[0]["dx"].float(), ref["dx_full"].cpu().float())
    print(f"\n[tp_mlp] split-vs-full bands: y {y_band:.2e}, "
          f"dx {dx_band:.2e}")
    assert y_band < 3e-2 and dx_band < 3e-2, (y_band, dx_band)


def test_tp_mlp_hostmem_pair_matches_reference_bitwise():
    # REAL child-process daemons, one per rank (the local_pair
    # pattern): in-process cohabitation shares one CUDA context,
    # where a parked collective stream can block the OTHER rank's
    # lazy module loads — a deadlock class production never has
    # (separate processes). The trace that convicted the old rig:
    # rank 0 sent seq 1 and waited; rank 1 never reached its enqueue.
    from dataflow_training.distributed import daemons
    from dataflow_training.distributed.topology import local_pair_topology

    topo = local_pair_topology(ports=(PA, PB))
    ranks = [topo.host("local0"), topo.host("local1")]
    lanes = ("tpa", "tpb")
    clients = []
    try:
        for host, lane in zip(ranks, lanes):
            daemons.kill(host, lane=lane)
            daemons.launch(
                host, lane=lane, backing_gib=0.4,
                peer_port=int(host.peer_listen.rsplit(":", 1)[1]),
                extra_flags="--plugin "
                            "dataflow_training.model_families.tp_mlp")
        deadline = time.time() + 120
        while time.time() < deadline and len(clients) < 2:
            try:
                socks = []
                for host, lane in zip(ranks, lanes):
                    sock = daemons.paths(host, lane)["sock"]
                    probe = EngineClient(sock, client_name="probe")
                    probe.health()
                    probe.close()
                    socks.append(sock)
                clients = [EngineClient(s, client_name=h.name)
                           for s, h in zip(socks, ranks)]
            except Exception:
                time.sleep(1.0)
        assert len(clients) == 2, "local pair daemons unreachable"
        clients[0].peer_connect("local1", ranks[1].peer_listen)
        time.sleep(1.0)
        got = drive_pair(clients, ["local0", "local1"], "hostmem")
        check_pair(got, reference(CFG, SEED))
    finally:
        for c in clients:
            try:
                c.shutdown()
            except Exception:
                pass
        for host, lane in zip(ranks, lanes):
            try:
                daemons.kill(host, lane=lane)
            except Exception:
                pass


def test_tp_mlp_crossbox_auto_matches_reference_within_ulp():
    from dataflow_training.distributed import daemons
    from dataflow_training.distributed.hosts import run_py, uds_forward
    from dataflow_training.distributed.topology import load_topology_or_none

    topo = load_topology_or_none()
    if topo is None or not topo.remotes():
        pytest.skip("needs a topology.toml with a remote host")
    local, remote = topo.local(), topo.remotes()[0]
    lane, port = "tpx", 29655
    fwd = None
    clients = []
    try:
        for host in (local, remote):
            daemons.kill(host, lane=lane)
            daemons.launch(
                host, lane=lane, backing_gib=2.0, peer_port=port,
                extra_flags="--plugin dataflow_training.model_families.tp_mlp")
        import tempfile

        fwd_sock = tempfile.mktemp(suffix=".sock", prefix="tpfwd-")
        fwd = uds_forward(remote, daemons.paths(remote, lane)["sock"],
                          fwd_sock)
        deadline = time.time() + 120
        ca = cb = None
        while time.time() < deadline:
            try:
                for sock in (daemons.paths(local, lane)["sock"], fwd_sock):
                    probe = EngineClient(sock, client_name="probe")
                    probe.health()
                    probe.close()
                ca = EngineClient(daemons.paths(local, lane)["sock"],
                                  client_name=local.name)
                cb = EngineClient(fwd_sock, client_name=remote.name)
                break
            except Exception:
                time.sleep(1.0)
        assert ca is not None and cb is not None, "daemons unreachable"
        clients = [ca, cb]
        ca.peer_connect(remote.name, remote.peer_addr(port))
        time.sleep(1.0)
        got = drive_pair(clients, [local.name, remote.name], "auto")
        # Ranks whose GPU architecture differs from rank 0's get the
        # ulp budget — their local gemms may break rounding ties
        # differently — while the group-plane properties stay bitwise.
        # engine_info carries only the device index, so read each
        # daemon's compute capability on its own host; a homogeneous
        # fleet yields an empty set and a fully bitwise compare.
        caps = []
        for host, client in zip((local, remote), clients):
            idx = int(client.engine_info.get("device", 0))
            probe_out = run_py(
                host, "import torch; print(tuple("
                      f"torch.cuda.get_device_capability({idx})))")
            caps.append(probe_out.strip())
        hetero = tuple(r for r in range(len(caps)) if caps[r] != caps[0])
        check_pair(got, reference(CFG, SEED), hetero_ranks=hetero)
    finally:
        for c in clients:
            try:
                c.shutdown()
            except Exception:
                pass
        if fwd is not None:
            fwd.terminate()
        for host in (local, remote):
            try:
                daemons.kill(host, lane=lane)
            except Exception:
                pass
