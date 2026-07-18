"""Z1 gate: sharded optimizer == plain data parallelism, BITWISE.

Two real daemons on this GPU run the same two training steps twice:
once with the plain replicated-optimizer DP lowering, once with the
zero1 sharded lowering (region reduce -> owned-only update -> W
broadcast; O objects shrink to owned slots). Elementwise math over
identically-reduced gradients is the same in both shapes, so the gate
demands exact bytes:

  1. sharded replicas end bitwise identical to each other;
  2. sharded weights end bitwise identical to the plain-DP run's;
  3. per-step losses match exactly across the two configs;
  4. each rank's O store objects actually SHRANK (the memory win).

The embed root is a single-matrix field, so this also exercises the
row-split path (rows-ranged reduce/update/broadcast + row-sliced O
slots) on real machinery.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataclasses import replace  # noqa: E402

from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow_training.distributed.fleet import lower_with_group  # noqa: E402
from dataflow_training.distributed.sharding import (  # noqa: E402
    ParallelConfig,
    layer_fields_by_root,
    zero1_halves,
)
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow_training.model_families.llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow_training.lowering.planning import plan_program  # noqa: E402
from dataflow_training.register import register_all  # noqa: E402
from dataflow_training.run.driver import init_model  # noqa: E402

pytestmark = pytest.mark.fleet

register_all()          # in-process Server rigs share this registry

T_STEP = 128
SEQ = 32
SEED = 11
STEPS = 2
PORTS = {"plain": (29531, 29532), "zero1": (29533, 29534)}


def make_cfg():
    return replace(ShapedLlamaConfig.tiny(), seq_len=SEQ,
                   batch=(T_STEP // 2) // SEQ, grad_accum_rounds=1)


def cfg_dict(cfg):
    return {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len, "batch": cfg.batch,
            "grad_accum_rounds": cfg.grad_accum_rounds}


def step_tokens(cfg, step: int):
    g = torch.Generator().manual_seed(1000 + step)
    tok = torch.randint(0, cfg.vocab_size, (T_STEP,), generator=g,
                        dtype=torch.int32)
    tgt = torch.randint(0, cfg.vocab_size, (T_STEP,), generator=g,
                        dtype=torch.int32)
    return tok, tgt


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


def put_step(client, cfg, rank: int, step: int) -> None:
    tok, tgt = step_tokens(cfg, step)
    per = T_STEP // 2
    lo = rank * per
    client.put_object("tokens_0_0", tok[lo:lo + per].numpy().tobytes())
    client.put_object("targets_0_0", tgt[lo:lo + per].numpy().tobytes())


class RankRun:
    def __init__(self, client, prog_id, step: int):
        self.client = client
        self.prog_id = prog_id
        self.step = step
        self.out = None
        self.err = None

    def __call__(self):
        try:
            self.out = self.client.run(
                self.prog_id,
                args={"step": self.step, "valid_rows": T_STEP},
                fetch=["loss_0_0"])
        except Exception as e:
            self.err = e


def run_config(tmp_path, label: str, parallels) -> dict:
    """Boot a two-daemon pair, train STEPS steps, return final W bytes
    per rank, per-step per-rank losses, and O object sizes."""
    cfg = make_cfg()
    pa, pb = PORTS[label]
    ca = boot(tmp_path, f"{label}-a", pa)
    cb = boot(tmp_path, f"{label}-b", pb)
    clients = (ca, cb)
    out = {"w": [], "losses": [], "o_bytes": []}
    try:
        ca.peer_connect(f"{label}-b", f"127.0.0.1:{pb}")
        progs = []
        for rank, client in enumerate(clients):
            planned = plan_program(
                lower_with_group(cfg, "dp", parallel=parallels[rank]),
                fast_memory_capacity=96 * 1024 * 1024)
            progs.append(planned.program)
            init_kwargs = {"seed": SEED}
            if parallels[rank] is not None:
                init_kwargs["object_sizes"] = {
                    s.id: s.size_bytes
                    for s in planned.program.initial_objects
                    if s.id.startswith("O_")}
            init_model(client, "llama3", cfg_dict(cfg), **init_kwargs)
            put_step(client, cfg, rank, 0)
            reg = client.register_program(program_to_dict(planned.program),
                                          resolver={"kind": "model_family",
                                                    "family": "llama3",
                                                    "cfg": cfg_dict(cfg)})
            assert not reg["bindings"]["missing_inputs"]
            prog_id = reg["prog_id"]
            # warm-up (no group yet: comm skips), then re-seed + re-put
            warm = client.run(prog_id,
                              args={"step": 0, "valid_rows": T_STEP})
            assert warm.get("state") == "done", warm
            init_model(client, "llama3", cfg_dict(cfg), **init_kwargs)
            put_step(client, cfg, rank, 0)
            out.setdefault("prog_ids", []).append(prog_id)
        ca._call("create_peer_group",
                 {"name": "dp", "members": [f"{label}-a", f"{label}-b"],
                  "backend": "hostmem"})

        for step in range(STEPS):
            if step > 0:
                for rank, client in enumerate(clients):
                    put_step(client, cfg, rank, step)
            runs = [RankRun(clients[r], out["prog_ids"][r], step)
                    for r in range(2)]
            threads = [threading.Thread(target=r) for r in runs]
            for t in threads:
                t.start()
            for t in threads:
                t.join(120)
            for r in runs:
                assert r.err is None, (label, step, r.err)
                assert r.out.get("state") == "done", (label, step, r.out)
            out["losses"].append(
                [r.out["fetched"]["loss_0_0"] for r in runs])

        w_ids = sorted(s.id for s in progs[0].initial_objects
                       if s.id.startswith("W_"))
        o_ids = sorted(s.id for s in progs[0].initial_objects
                       if s.id.startswith("O_"))
        for client in clients:
            out["w"].append({oid: bytes(client.get_object(oid))
                             for oid in w_ids})
            out["o_bytes"].append({oid: len(bytes(client.get_object(oid)))
                                   for oid in o_ids})
        return out
    finally:
        for c in clients:
            try:
                c.shutdown()
            except Exception:
                pass


def test_zero1_bitwise_equals_plain_dp(tmp_path):
    cfg = make_cfg()
    plan = zero1_halves(layer_fields_by_root(cfg), "dp", 2,
                        replicate_below_bytes=256)
    plan.validate()
    plan.v1_consumable()
    assert any(a.owner != "all" for a in plan.assignments), \
        "plan degenerated to fully replicated — gate is vacuous"

    plain = run_config(tmp_path, "plain", [None, None])
    zero1 = run_config(tmp_path, "zero1",
                       [ParallelConfig("dp", r, 2, plan) for r in (0, 1)])

    # per-rank losses identical across configs, step by step
    assert zero1["losses"] == plain["losses"], (
        plain["losses"], zero1["losses"])

    # sharded replicas bitwise identical to each other...
    for oid, blob in zero1["w"][0].items():
        assert blob == zero1["w"][1][oid], f"replica divergence at {oid}"
    # ...and bitwise identical to the plain-DP weights
    for oid, blob in zero1["w"][0].items():
        assert blob == plain["w"][0][oid], f"zero1 != plain at {oid}"

    # the memory win: every sharded O object shrank on both ranks
    shrank = 0
    for rank in (0, 1):
        for oid, n in zero1["o_bytes"][rank].items():
            full = plain["o_bytes"][rank][oid]
            assert n <= full, (oid, n, full)
            if n < full:
                shrank += 1
    assert shrank >= 2, zero1["o_bytes"]
    tot_z = sum(sum(d.values()) for d in zero1["o_bytes"])
    tot_p = sum(sum(d.values()) for d in plain["o_bytes"])
    print(f"\n[zero1] O bytes {tot_z} vs plain {tot_p} "
          f"({tot_z / tot_p:.2f}x); losses {zero1['losses']}")
    assert tot_z < 0.75 * tot_p
