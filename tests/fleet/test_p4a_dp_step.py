"""P4a gate: the FIRST comm-in-task data-parallel step — two real
daemons sharing this GPU, one dp group, the SAME dp_group-lowered
program, each daemon training on ITS HALF of the master batch with the
GLOBAL valid-token denominator; optimizer tasks allreduce(dW) on the
group stream before updating.

Asserts, in order of strength:
  1. the two replicas' weights are BITWISE IDENTICAL after the step
     (rank-ordered fp32 reduction makes the summed gradient exactly
     equal on both members);
  2. both match a SINGLE-ENGINE run over the combined batch (same
     tokens, ga=2 partition, same global denominator) within bf16
     accumulation-tree noise — the P4a equivalence, at one step.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataclasses import replace  # noqa: E402

from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.fake import FakeBackend  # noqa: E402
from dataflow_training.blocks.segments import uniform_segments  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view  # noqa: E402
from dataflow_training.model_families.families import resolve_family  # noqa: E402
from dataflow_training.lowering.emit import apply_exact_sizes, size_of_factory  # noqa: E402
from dataflow_training.model_families.llama3 import (  # noqa: E402
    ShapedLlamaConfig,
    family_layouts,
)
from dataflow_training.lowering.planning import plan_program  # noqa: E402
from dataflow_training.lowering.shaped_program import (  # noqa: E402
    ShapedHardware,
    build_shaped_program,
    roofline_block_kind_spec,
)
from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.fleet

T_STEP = 128
SEQ = 32
SEED = 11
PA, PB = 29521, 29522


def dp_cfg(ga: int, world: int = 1):
    return replace(ShapedLlamaConfig.tiny(), seq_len=SEQ,
                   batch=(T_STEP // (ga * world)) // SEQ,
                   grad_accum_rounds=ga)


def lower_with_group(cfg, dp_group):
    hw = ShapedHardware()
    shaped = build_shaped_program(
        cfg, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(cfg, hw)},
        dp_group=dp_group)
    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "llama3-exact",
                             size_of=size_of_factory(dims, fl))


def master_tokens():
    g = torch.Generator().manual_seed(99)
    tok = torch.randint(0, dp_cfg(2).vocab_size, (T_STEP,), generator=g,
                        dtype=torch.int32)
    tgt = torch.randint(0, dp_cfg(2).vocab_size, (T_STEP,), generator=g,
                        dtype=torch.int32)
    return tok, tgt


def cfg_dict(cfg):
    return {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len, "batch": cfg.batch,
            "grad_accum_rounds": cfg.grad_accum_rounds}


def single_box_reference():
    """Direct-engine ga=2 over the SAME tokens (halves = the DP split),
    same global denominator. Returns {W_id: {field: cpu tensor}}."""
    cfg = dp_cfg(2)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    program = fam.lower(cfg)
    planned = plan_program(program, fast_memory_capacity=96 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=SEED)
    tok, tgt = master_tokens()
    per = T_STEP // 2
    for r in range(2):
        torch_view(values[f"tokens_0_{r}"], (per,),
                   torch.int32).copy_(tok[r * per:(r + 1) * per])
        torch_view(values[f"targets_0_{r}"], (per,),
                   torch.int32).copy_(tgt[r * per:(r + 1) * per])
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program),
                  "step": 0, "valid_rows": T_STEP})
    _, fl = family_layouts(cfg)
    out = {}
    for i, layer in enumerate(fl.layers):
        rec = result.objects.get(f"W_{i}")
        slot = rec.backing or rec.fast
        out[f"W_{i}"] = {
            f.name: torch_view(slot.buffer, f.shape,
                               TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes).clone().cpu()
            for f in layer.weights.fields}
    result.close()
    return out


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
    return server, EngineClient(sock, client_name=name)


class RankRun:
    def __init__(self, client, prog_id):
        self.client = client
        self.prog_id = prog_id
        self.out = None
        self.err = None

    def __call__(self):
        try:
            self.out = self.client.run(
                self.prog_id, args={"step": 0, "valid_rows": T_STEP},
                fetch=["loss_0_0"])
        except Exception as e:
            self.err = e


def test_p4a_two_daemon_dp_step(tmp_path):
    cfg = dp_cfg(1, world=2)            # 64 tokens per rank, ga=1
    planned = plan_program(lower_with_group(cfg, "dp"),
                           fast_memory_capacity=96 * 1024 * 1024)
    prog_dict = program_to_dict(planned.program)
    resolver = {"family": "llama3", "cfg": cfg_dict(cfg)}
    tok, tgt = master_tokens()

    sa, ca = boot(tmp_path, "dp-a", PA)
    sb, cb = boot(tmp_path, "dp-b", PB)
    try:
        ca.peer_connect("dp-b", f"127.0.0.1:{PB}")
        per = T_STEP // 2
        for rank, client in ((0, ca), (1, cb)):
            client.materialize_group({"kind": "family_init_all",
                                      "family": "llama3",
                                      "cfg": cfg_dict(cfg), "seed": SEED})
            lo = rank * per
            client.put_object("tokens_0_0",
                              tok[lo:lo + per].numpy().tobytes())
            client.put_object("targets_0_0",
                              tgt[lo:lo + per].numpy().tobytes())
            reg = client.register_program(prog_dict, resolver=resolver)
            assert not reg["bindings"]["missing_inputs"]
        prog_id = reg["prog_id"]
        # WARM-UP before the group exists: standalone runs (comm skips
        # on the rank-complete P4a artifact) compile + cuModuleLoad
        # every kernel; a FIRST launch during a parked collective
        # deadlocks the device (findings, P4a). Re-seed afterwards —
        # and RE-PUT the data: family_init_all refills EVERY initial
        # object, token buffers included (the trap that silently gave
        # both ranks identical seeded tokens on the first attempt).
        for rank, client in ((0, ca), (1, cb)):
            warm = client.run(prog_id,
                              args={"step": 0, "valid_rows": T_STEP})
            assert warm.get("state") == "done", warm
            client.materialize_group({"kind": "family_init_all",
                                      "family": "llama3",
                                      "cfg": cfg_dict(cfg), "seed": SEED})
            lo = rank * per
            client.put_object("tokens_0_0",
                              tok[lo:lo + per].numpy().tobytes())
            client.put_object("targets_0_0",
                              tgt[lo:lo + per].numpy().tobytes())
        ca._call("create_peer_group",
                 {"name": "dp", "members": ["dp-a", "dp-b"],
                  "backend": "hostmem"})

        runs = [RankRun(ca, prog_id), RankRun(cb, prog_id)]
        threads = [threading.Thread(target=r) for r in runs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        for r in runs:
            assert r.err is None, r.err
            assert r.out.get("state") == "done", r.out
        # per-rank losses are DIFFERENT partial sums whose SUM is the
        # global step mean (the identical-loss trap's tripwire)
        la = runs[0].out["fetched"]["loss_0_0"]
        lb = runs[1].out["fetched"]["loss_0_0"]
        assert abs(la - lb) > 1e-4, (la, lb)

        # 1) replicas bitwise identical
        n_layers = cfg.n_layers
        for i in range(n_layers):
            wa = ca.get_object(f"W_{i}")
            wb = cb.get_object(f"W_{i}")
            assert bytes(wa) == bytes(wb), f"replica divergence at W_{i}"
        we_a = ca.get_object("W_embed")
        we_b = cb.get_object("W_embed")
        assert bytes(we_a) == bytes(we_b)

        # 2) equivalence vs the single-engine combined-batch run
        ref = single_box_reference()
        _, fl = family_layouts(cfg)
        worst = 0.0
        for i, layer in enumerate(fl.layers):
            raw = bytes(ca.get_object(f"W_{i}"))
            buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            for f in layer.weights.fields:
                n = 1
                for s in f.shape:
                    n *= s
                dt = TORCH_DTYPE_BY_NAME[f.dtype]
                view = buf[f.offset_bytes:
                           f.offset_bytes + n * dt.itemsize].view(dt)
                d = rel_l2(view.float(),
                           ref[f"W_{i}"][f.name].reshape(-1).float())
                worst = max(worst, d)
        print(f"\n[P4a] DP(2 ranks) vs single-engine: worst field "
              f"rel_l2 = {worst:.3e}")
        assert worst < 3e-3, worst
    finally:
        for c in (ca, cb):
            try:
                c.shutdown()
            except Exception:
                pass
