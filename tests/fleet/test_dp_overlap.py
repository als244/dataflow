"""DP overlap gate: the tail-placement + grad_reduce lowering (the
exchange overlaps backward; optimizers consume PRE-REDUCED dWg
transients) must produce BITWISE-identical weights to the interleaved
lowering — the sums are the same elementwise adds, only their
placement moves. Two real daemons per variant on this GPU."""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataclasses import replace  # noqa: E402

from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow.training.models.llama3 import (  # noqa: E402
    ShapedLlamaConfig,
    family_layouts,
)
from dataflow.training.planning import plan_program  # noqa: E402
from dataflow.training.shaped_program import (  # noqa: E402
    ShapedHardware,
    build_shaped_program,
    roofline_block_kind_spec,
)

pytestmark = pytest.mark.fleet

T_STEP = 128
SEQ = 32
SEED = 11


def dp_cfg(placement: str):
    return replace(ShapedLlamaConfig.tiny(), seq_len=SEQ,
                   batch=(T_STEP // 2) // SEQ, grad_accum_rounds=1,
                   optimizer_placement=placement)


def lower(cfg, dp_overlap: bool):
    hw = ShapedHardware()
    shaped = build_shaped_program(
        cfg, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(cfg, hw)},
        dp_group="dp", dp_overlap=dp_overlap)
    from dataflow.training.lowering import apply_exact_sizes, size_of_factory

    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "llama3-exact",
                             size_of=size_of_factory(dims, fl))


def cfg_dict(cfg):
    return {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len, "batch": cfg.batch,
            "grad_accum_rounds": cfg.grad_accum_rounds,
            "optimizer_placement": cfg.optimizer_placement}


def master_tokens(vocab):
    g = torch.Generator().manual_seed(99)
    tok = torch.randint(0, vocab, (T_STEP,), generator=g, dtype=torch.int32)
    tgt = torch.randint(0, vocab, (T_STEP,), generator=g, dtype=torch.int32)
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


def run_dp_step(tmp, cfg, dp_overlap: bool, names, ports) -> dict:
    """One DP step across two fresh daemons; returns weight bytes."""
    planned = plan_program(lower(cfg, dp_overlap),
                           fast_memory_capacity=96 * 1024 * 1024)
    prog_dict = program_to_dict(planned.program)
    resolver = {"family": "llama3", "cfg": cfg_dict(cfg)}
    tok, tgt = master_tokens(cfg.vocab_size)
    per = T_STEP // 2

    sa, ca = boot(tmp, names[0], ports[0])
    sb, cb = boot(tmp, names[1], ports[1])
    try:
        ca.peer_connect(names[1], f"127.0.0.1:{ports[1]}")
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
        # warm-up (kernel loads precede any parked collective), then
        # re-seed and RE-PUT (init refills token buffers — findings)
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
                 {"name": "dp", "members": list(names),
                  "backend": "hostmem"})
        runs = [RankRun(ca, prog_id), RankRun(cb, prog_id)]
        threads = [threading.Thread(target=r) for r in runs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(180)
        for r in runs:
            assert r.err is None, r.err
            assert r.out.get("state") == "done", r.out
        weights = {}
        for i in range(cfg.n_layers):
            wa = bytes(ca.get_object(f"W_{i}"))
            wb = bytes(cb.get_object(f"W_{i}"))
            assert wa == wb, f"replica divergence at W_{i}"
            weights[f"W_{i}"] = wa
        weights["W_embed"] = bytes(ca.get_object("W_embed"))
        return weights
    finally:
        for c in (ca, cb):
            try:
                c.shutdown()
            except Exception:
                pass


def test_overlap_lowering_matches_interleaved_bitwise(tmp_path):
    base = run_dp_step(tmp_path, dp_cfg("interleaved"), False,
                       ("ovl-a", "ovl-b"), (29561, 29562))
    ovl = run_dp_step(tmp_path, dp_cfg("tail"), True,
                      ("ovl-c", "ovl-d"), (29563, 29564))
    assert set(base) == set(ovl)
    for wid, blob in base.items():
        assert ovl[wid] == blob, f"{wid}: overlap != interleaved"
    print(f"\n[overlap] {len(base)} weight objects bitwise-identical "
          f"across lowerings")
