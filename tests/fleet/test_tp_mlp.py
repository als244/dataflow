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

Test 1 runs the hostmem lane on two in-process daemons (the family
registers on import in this interpreter). Test 2 runs the identical
driver cross-box through the auto->nccl lane, with remote daemons
loading the family via `dataflowd --plugin`.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataclasses import asdict, replace  # noqa: E402

import dataflow.training.models.tp_mlp as tp  # noqa: E402  (registers)
from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow.training.planning import plan_program  # noqa: E402
from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.fleet

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


def boot(tmp, name, peer_port):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=0.4,
        peer_name=name, peer_listen=f"127.0.0.1:{peer_port}"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    return EngineClient(sock, client_name=name)


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
        client.materialize_group({"kind": "family_init_all",
                                  "family": "tp_mlp",
                                  "cfg": asdict(cfg_r), "seed": SEED})
        reg = client.register_program(
            program_to_dict(planned.program),
            resolver={"family": "tp_mlp", "cfg": asdict(cfg_r)})
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
    dims = tp.dims_of_tp_mlp(CFG)
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


def check_pair(got, ref):
    # cross-rank: the allreduced outputs must be identical replicas
    assert bitwise(got[0]["y"], got[1]["y"]), "y replicas diverge"
    assert bitwise(got[0]["dx"], got[1]["dx"]), "dx replicas diverge"
    # bitwise vs the split-order reference
    assert bitwise(got[0]["y"], ref["y"].cpu()), "y != split reference"
    assert bitwise(got[0]["dx"], ref["dx"].cpu()), "dx != split reference"
    # local shard grads: gemm-for-gemm bitwise (dW layout mirrors the
    # weight layout, so its fields are named w1/w3/w2)
    for r in (0, 1):
        for grad, field in (("dw1", "w1"), ("dw3", "w3"),
                            ("dw2", "w2")):
            assert bitwise(got[r]["dw"][field],
                           ref["per_rank"][r][grad].cpu()), (r, grad)
    # banded sanity vs the full-order single-gemm math
    y_band = rel_l2(got[0]["y"].float(), ref["y_full"].cpu().float())
    dx_band = rel_l2(got[0]["dx"].float(), ref["dx_full"].cpu().float())
    print(f"\n[tp_mlp] split-vs-full bands: y {y_band:.2e}, "
          f"dx {dx_band:.2e}")
    assert y_band < 3e-2 and dx_band < 3e-2, (y_band, dx_band)


def test_tp_mlp_loopback_hostmem(tmp_path):
    ca = boot(tmp_path, "tp-a", PA)
    cb = boot(tmp_path, "tp-b", PB)
    try:
        ca.peer_connect("tp-b", f"127.0.0.1:{PB}")
        got = drive_pair([ca, cb], ["tp-a", "tp-b"], "hostmem")
        check_pair(got, reference(CFG, SEED))
    finally:
        for c in (ca, cb):
            try:
                c.shutdown()
            except Exception:
                pass


def test_tp_mlp_crossbox_auto():
    from dataflow.pretrain.hostops import (
        daemon_paths,
        kill_daemon,
        launch_daemon,
        uds_forward,
    )
    from dataflow.pretrain.topology import load_topology_or_none

    topo = load_topology_or_none()
    if topo is None or not topo.remotes():
        pytest.skip("needs a topology.toml with a remote host")
    local, remote = topo.local(), topo.remotes()[0]
    lane, port = "tpx", 29655
    fwd = None
    clients = []
    try:
        for host in (local, remote):
            kill_daemon(host, lane=lane)
            launch_daemon(
                host, lane=lane, slab_gib=2.0, peer_port=port,
                extra_flags="--plugin dataflow.training.models.tp_mlp")
        import tempfile

        fwd_sock = tempfile.mktemp(suffix=".sock", prefix="tpfwd-")
        fwd = uds_forward(remote, daemon_paths(remote, lane)["sock"],
                          fwd_sock)
        deadline = time.time() + 120
        ca = cb = None
        while time.time() < deadline:
            try:
                for sock in (daemon_paths(local, lane)["sock"], fwd_sock):
                    probe = EngineClient(sock, client_name="probe")
                    probe.health()
                    probe.close()
                ca = EngineClient(daemon_paths(local, lane)["sock"],
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
        check_pair(got, reference(CFG, SEED))
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
                kill_daemon(host, lane=lane)
            except Exception:
                pass
