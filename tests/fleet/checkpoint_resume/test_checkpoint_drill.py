"""The checkpoint/resume drill — the snapshot/restore API exercised
end-to-end.

Truth run: one daemon trains steps 0..2K-1 straight through, with a
snapshot taken at the step-K boundary (client_meta carrying the
driver's resume state: step, seed, cfg). Drill: a FRESH daemon
(fresh store — the in-process equivalent of process death) restores
the snapshot, re-registers the same program (content-derived id),
and resumes K..2K-1.

The daemon is stateless per step by design — every piece of
trajectory state lives in store objects (W/O) plus the driver's
``step`` argument — so the resumed tail must equal the truth tail
BITWISE. Any miss means the resume manifest is incomplete, and the
drill names the missing field.

Tests:
- test_checkpoint_resume_bitwise: a fresh daemon that restores the K-boundary snapshot, re-registers the same content-derived program, and resumes reproduces the uninterrupted tail bitwise, and restoring into a non-empty store is refused.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("dataflow_sim")
pytest.importorskip("cuda.bindings")

from dataclasses import replace  # noqa: E402

from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow_training.distributed.fleet import lower_with_group  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow.service.wire import ServiceError  # noqa: E402
from dataflow_training.model_families.llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow_training.lowering.planning import plan_program  # noqa: E402
from dataflow_training.register import register_all  # noqa: E402
from dataflow_training.run.driver import init_model  # noqa: E402

pytestmark = [pytest.mark.fleet, pytest.mark.gpu, pytest.mark.sim]

register_all()          # in-process Server rigs share this registry

T_STEP = 128
SEQ = 32
SEED = 11
K = 6                       # snapshot boundary; run to 2K


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
    g = torch.Generator().manual_seed(4000 + step)
    tok = torch.randint(0, cfg.vocab_size, (T_STEP,), generator=g,
                        dtype=torch.int32)
    tgt = torch.randint(0, cfg.vocab_size, (T_STEP,), generator=g,
                        dtype=torch.int32)
    return tok, tgt


def boot(tmp, name):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(socket_path=sock, fake=False,
                                 slab_backing_gib=0.4))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    return EngineClient(sock, client_name=name)


def put_step(client, cfg, step: int) -> None:
    tok, tgt = step_tokens(cfg, step)
    client.put_object("tokens_0_0", tok.numpy().tobytes())
    client.put_object("targets_0_0", tgt.numpy().tobytes())


def train_steps(client, prog_id, cfg, lo: int, hi: int) -> list:
    losses = []
    for step in range(lo, hi):
        put_step(client, cfg, step)
        out = client.run(prog_id,
                         args={"step": step, "valid_rows": T_STEP},
                         fetch=["loss_0_0"])
        assert out.get("state") == "done", (step, out)
        losses.append(out["fetched"]["loss_0_0"])
    return losses


def register(client, cfg, prog_dict) -> str:
    reg = client.register_program(
        prog_dict, resolver={"kind": "model_family", "family": "llama3",
                             "cfg": cfg_dict(cfg)})
    assert not reg["bindings"]["missing_inputs"]
    return reg["prog_id"]


def test_checkpoint_resume_bitwise(tmp_path):
    cfg = make_cfg()
    planned = plan_program(lower_with_group(cfg, "dp"),
                           fast_memory_capacity=96 * 1024 * 1024)
    prog_dict = program_to_dict(planned.program)
    snap_dir = tmp_path / "ck"

    # ---- truth daemon: train through, snapshot at the K boundary ----
    ca = boot(tmp_path, "ck-a")
    try:
        init_model(ca, "llama3", cfg_dict(cfg), seed=SEED)
        put_step(ca, cfg, 0)
        prog_a = register(ca, cfg, prog_dict)
        head = train_steps(ca, prog_a, cfg, 0, K)
        out = ca.snapshot("all", snap_dir,
                          client_meta={"step": K, "seed": SEED,
                                       "cfg": cfg_dict(cfg),
                                       "prog_id": prog_a})
        snap = ca.wait_snapshot(out["snap_id"])
        assert snap["state"] == "done", snap
        # restoring into a non-empty store must refuse loudly
        with pytest.raises(ServiceError):
            ca.restore_snapshot(snap_dir)
        truth_tail = train_steps(ca, prog_a, cfg, K, 2 * K)
    finally:
        try:
            ca.shutdown()
        except Exception:
            pass

    # ---- fresh daemon: restore, re-register, resume ----
    cb = boot(tmp_path, "ck-b")
    try:
        res = cb.restore_snapshot(snap_dir)
        meta = res["client_meta"]
        assert meta["step"] == K and meta["seed"] == SEED
        assert "W_0" in res["restored"] and "O_0" in res["restored"]
        prog_b = register(cb, cfg, prog_dict)
        assert prog_b == meta["prog_id"], "content-derived id changed"
        resumed_tail = train_steps(cb, prog_b, cfg, meta["step"], 2 * K)
    finally:
        try:
            cb.shutdown()
        except Exception:
            pass

    assert resumed_tail == truth_tail, (
        "resume diverged from the uninterrupted twin:\n"
        f"  truth   {truth_tail}\n  resumed {resumed_tail}")
    print(f"\n[ck0] bitwise resume: head {['%.4f' % x for x in head]} "
          f"tail {['%.4f' % x for x in truth_tail]}")
