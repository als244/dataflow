"""T2 gate: llama3-TP (MLP tensor parallelism through the sharding
API) on two real daemons sharing this GPU.

TP splits compute, not data: both ranks run the SAME full batch, the
forward allreduces each layer's MLP partial product, the backward
allreduces dx, and the optimizer runs in replica-grads mode (local
updates, owner broadcasts for replicated fields, fully-local shard
fields). Asserts:

  1. per-step losses are BITWISE equal across ranks (the allreduced
     activations make every replicated computation identical);
  2. after 2 steps every replicated W field is bitwise identical
     across ranks (the owner-broadcast re-pin), while the MLP shard
     fields hold DIFFERENT halves;
  3. the whole run sits in a tight band of a plain single-daemon run
     over the same tokens (split-sum rounding is the only
     difference);
  4. per-rank W/O objects are genuinely smaller than plain.
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
from dataflow_training.lowering.emit import narrow_layouts
from dataflow_training.distributed.sharding import (  # noqa: E402
    ParallelConfig,
    layer_fields_by_root,
    tp_mlp_shards,
    tp_view,
)
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow_training.model_families.llama3 import (  # noqa: E402
    ShapedLlamaConfig,
    family_layouts,
)
from dataflow_training.lowering.planning import plan_program  # noqa: E402
from dataflow_training.register import register_all  # noqa: E402
from dataflow_training.run.driver import init_model  # noqa: E402

pytestmark = pytest.mark.fleet

register_all()          # in-process Server rigs share this registry

T_STEP = 128
SEQ = 32
SEED = 11
STEPS = 2
PORTS = (29561, 29562, 29563)


def make_cfg():
    return replace(ShapedLlamaConfig.tiny(), seq_len=SEQ,
                   batch=T_STEP // SEQ, grad_accum_rounds=1)


def cfg_dict(cfg):
    return {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len, "batch": cfg.batch,
            "grad_accum_rounds": cfg.grad_accum_rounds}


def step_tokens(cfg, step: int):
    g = torch.Generator().manual_seed(2000 + step)
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


def put_full_batch(client, cfg, step: int) -> None:
    tok, tgt = step_tokens(cfg, step)
    client.put_object("tokens_0_0", tok.numpy().tobytes())
    client.put_object("targets_0_0", tgt.numpy().tobytes())


def setup_rank(client, cfg, planned, parallel) -> str:
    init_kwargs = {"seed": SEED}
    if parallel is not None:
        view = tp_view(parallel.plan, parallel.rank)
        init_kwargs["tp_view"] = {
            root: {f: list(sl) for f, sl in per.items()}
            for root, per in view.items()}
        init_kwargs["object_sizes"] = {
            s.id: s.size_bytes for s in planned.program.initial_objects
            if s.id.startswith(("O_", "W_"))}
    init_model(client, "llama3", cfg_dict(cfg), **init_kwargs)
    put_full_batch(client, cfg, 0)
    reg = client.register_program(
        program_to_dict(planned.program),
        resolver={"kind": "model_family", "family": "llama3",
                  "cfg": cfg_dict(cfg)})
    assert not reg["bindings"]["missing_inputs"]
    prog_id = reg["prog_id"]
    warm = client.run(prog_id, args={"step": 0, "valid_rows": T_STEP})
    assert warm.get("state") == "done", warm
    # warm-up mutated W/O: re-seed
    init_model(client, "llama3", cfg_dict(cfg), **init_kwargs)
    put_full_batch(client, cfg, 0)
    return prog_id


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


def run_steps(clients, prog_ids, cfg) -> list:
    losses = []
    for step in range(STEPS):
        if step > 0:
            for client in clients:
                put_full_batch(client, cfg, step)
        runs = [RankRun(clients[i], prog_ids[i], step)
                for i in range(len(clients))]
        threads = [threading.Thread(target=r) for r in runs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(180)
        for r in runs:
            assert r.err is None, (step, r.err)
            assert r.out.get("state") == "done", (step, r.out)
        losses.append([r.out["fetched"]["loss_0_0"] for r in runs])
    return losses


def test_tp_llama3_two_daemons_vs_plain(tmp_path):
    cfg = make_cfg()
    plan = tp_mlp_shards(layer_fields_by_root(cfg), "dp", 2)
    plan.validate()
    plan.consumable("tp")

    # ---- plain single-daemon reference over the SAME tokens --------
    ref = boot(tmp_path, "tpref", PORTS[2])
    try:
        planned_ref = plan_program(lower_with_group(cfg, "dp"),
                                   fast_memory_capacity=96 * 1024 * 1024)
        ref_prog = setup_rank(ref, cfg, planned_ref, None)
        ref_losses = run_steps([ref], [ref_prog], cfg)
    finally:
        try:
            ref.shutdown()
        except Exception:
            pass

    # ---- the TP pair ------------------------------------------------
    ca = boot(tmp_path, "tp-a", PORTS[0])
    cb = boot(tmp_path, "tp-b", PORTS[1])
    try:
        ca.peer_connect("tp-b", f"127.0.0.1:{PORTS[1]}")
        parallels = [ParallelConfig("dp", r, 2, plan) for r in (0, 1)]
        planned = [plan_program(
            lower_with_group(cfg, "dp", parallel=parallels[r]),
            fast_memory_capacity=96 * 1024 * 1024) for r in (0, 1)]
        prog_ids = [setup_rank((ca, cb)[r], cfg, planned[r],
                               parallels[r]) for r in (0, 1)]
        ca._call("create_peer_group",
                 {"name": "dp", "members": ["tp-a", "tp-b"],
                  "backend": "hostmem"})
        losses = run_steps([ca, cb], prog_ids, cfg)

        # 1) replicas: bitwise-equal losses every step
        for step, (la, lb) in enumerate(losses):
            assert la == lb, (step, la, lb)

        # 2) replicated W fields bitwise across ranks; shards differ
        views = [tp_view(plan, r) for r in (0, 1)]
        fls = [narrow_layouts(family_layouts(cfg)[1], views[r]) for r in (0, 1)]
        wa = bytearray(ca.get_object("W_0"))
        wb = bytearray(cb.get_object("W_0"))
        la_ = fls[0].layers[0].weights
        lb_ = fls[1].layers[0].weights
        ta = torch.frombuffer(wa, dtype=torch.uint8)
        tb = torch.frombuffer(wb, dtype=torch.uint8)
        replicated = ("attn_norm_w", "wq", "wk", "wv", "wo",
                      "ffn_norm_w")
        for name in replicated:
            fa, fb = la_.field(name), lb_.field(name)
            assert torch.equal(
                ta[fa.offset_bytes:fa.offset_bytes + fa.nbytes],
                tb[fb.offset_bytes:fb.offset_bytes + fb.nbytes]), name
        for name in ("w1", "w3", "w2"):
            fa, fb = la_.field(name), lb_.field(name)
            assert not torch.equal(
                ta[fa.offset_bytes:fa.offset_bytes + fa.nbytes],
                tb[fb.offset_bytes:fb.offset_bytes + fb.nbytes]), name

        # 3) tight band vs the plain run (split-sum rounding only)
        worst = max(abs(losses[s][0] - ref_losses[s][0])
                    for s in range(STEPS))
        print(f"\n[tp-llama3] losses tp {losses} vs plain "
              f"{ref_losses}; worst |d| {worst:.2e}")
        assert worst < 2e-2, (losses, ref_losses)

        # 4) the memory shrink is real
        plain_w0 = next(s.size_bytes
                        for s in planned_ref.program.initial_objects
                        if s.id == "W_0")
        assert len(wa) < plain_w0
        o_plain = next(s.size_bytes
                       for s in planned_ref.program.initial_objects
                       if s.id == "O_0")
        assert len(bytes(ca.get_object("O_0"))) < o_plain
    finally:
        for c in (ca, cb):
            try:
                c.shutdown()
            except Exception:
                pass
