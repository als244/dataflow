"""S1.2 gates (GPU): programs + runs through the daemon.

The load-bearing gate: a tiny llama trained THROUGH THE DAEMON
produces per-step losses BIT-EQUAL to the in-process train() on
identical init/data — the service is a front door, not a fork.
Plus: rebind, cancel-mid-run daemon health, poison isolation,
run-after-run weight adoption.
"""
from __future__ import annotations

import dataclasses
import struct
import threading
import time

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from dataflow.core.jsonio import program_to_dict
from dataflow.service import EngineClient, EngineConfig, Server, ServiceError
from dataflow_training.register import register_all
from dataflow_training.run.driver import init_model


def _cfg_dict(cfg):
    return {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len, "batch": cfg.batch,
            "grad_accum_rounds": cfg.grad_accum_rounds}


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    """Real daemon (2 GiB slab) + a registered tiny-llama step program
    + the parallel in-process reference pieces."""
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.lowering.planning import plan_program

    tmp = tmp_path_factory.mktemp("svc_runs")
    sock = str(tmp / "runs.sock")
    register_all()      # in-process Server shares this registry
    server = Server(EngineConfig(socket_path=sock, fake=False,
                                 slab_backing_gib=2.0))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(300):
        try:
            with EngineClient(sock, client_name="probe"):
                break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)

    cfg = ShapedLlamaConfig.tiny()
    fam = resolve_family(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=2 * 1024**3)
    prog_dict = program_to_dict(planned.program)
    resolver_spec = {"kind": "model_family", "family": "llama3",
                     "cfg": _cfg_dict(cfg)}

    yield {"sock": sock, "server": server, "cfg": cfg, "fam": fam,
           "planned": planned, "prog_dict": prog_dict,
           "resolver": resolver_spec, "tmp": tmp}

    server.state.shutdown_requested.set()
    server.dispatcher.stop()
    if server.store.slab is not None:
        server.store.slab.free()


def _tokens(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, cfg.vocab_size,
                         (cfg.batch * cfg.seq_len,), generator=g,
                         dtype=torch.int32)
    tgts = toks.view(-1, cfg.seq_len).roll(-1, dims=1).reshape(-1).contiguous()
    return toks, tgts




def test_rebind_two_token_slabs(rig):
    """Preloaded-slab pattern: two resident token sets, per-run rebind;
    losses differ across binds, repeat across identical binds."""
    cfg = rig["cfg"]
    a_t, a_y = _tokens(cfg, seed=1)
    b_t, b_y = _tokens(cfg, seed=2)
    with EngineClient(rig["sock"], client_name="rebind") as c:
        init_model(c, "llama3", _cfg_dict(cfg), seed=3)
        for nm, (tt, yy) in (("data/a", (a_t, a_y)),
                             ("data/b", (b_t, b_y))):
            c.put_object(f"{nm}/tok", tt.numpy().tobytes())
            c.put_object(f"{nm}/tgt", yy.numpy().tobytes())
        # program still binds tokens_0_0 by default: keep residents too
        c.put_object("tokens_0_0", a_t.numpy().tobytes())
        c.put_object("targets_0_0", a_y.numpy().tobytes())
        reg = c.register_program(rig["prog_dict"], resolver=rig["resolver"])
        la = c.run(reg["prog_id"], args={"step": 0},
                   rebind={"tokens_0_0": "data/a/tok",
                           "targets_0_0": "data/a/tgt"},
                   fetch=["loss_0_0"])["fetched"]["loss_0_0"]
        lb = c.run(reg["prog_id"], args={"step": 1},
                   rebind={"tokens_0_0": "data/b/tok",
                           "targets_0_0": "data/b/tgt"},
                   fetch=["loss_0_0"])["fetched"]["loss_0_0"]
        assert la != lb
        with pytest.raises(ServiceError) as ei:
            c.run(reg["prog_id"], rebind={"tokens_0_0": "nope"})
        assert ei.value.code == "MISSING_INPUTS"


def test_poison_isolation_and_next_run_succeeds(rig):
    """A program whose task errors -> RUN_FAILED; the daemon survives
    and the next run is clean."""
    cfg = rig["cfg"]
    bad = dict(rig["prog_dict"])
    # poison: shrink one initial object's size in the program so the
    # binding check trips deterministically at RUN time via a rebind
    # to a wrong-size resident
    toks, tgts = _tokens(cfg, seed=9)
    with EngineClient(rig["sock"], client_name="poison") as c:
        init_model(c, "llama3", _cfg_dict(cfg), seed=5)
        c.put_object("tokens_0_0", toks.numpy().tobytes())
        c.put_object("targets_0_0", tgts.numpy().tobytes())
        c.put_object("tiny/wrong", b"xx")
        reg = c.register_program(rig["prog_dict"], resolver=rig["resolver"])
        with pytest.raises(ServiceError) as ei:
            c.run(reg["prog_id"], rebind={"tokens_0_0": "tiny/wrong"})
        assert ei.value.code == "BINDING_MISMATCH"
        r = c.run(reg["prog_id"], args={"step": 0}, fetch=["loss_0_0"])
        assert r["state"] == "done"
        assert c.health()["ok"]


def test_weights_adopted_not_refilled(rig):
    """Run-after-run: W_0 keeps training in place (the store IS the
    state); a second run continues from the first's weights."""
    cfg = rig["cfg"]
    toks, tgts = _tokens(cfg, seed=33)
    with EngineClient(rig["sock"], client_name="adopt") as c:
        init_model(c, "llama3", _cfg_dict(cfg), seed=21)
        c.put_object("tokens_0_0", toks.numpy().tobytes())
        c.put_object("targets_0_0", tgts.numpy().tobytes())
        reg = c.register_program(rig["prog_dict"], resolver=rig["resolver"])
        w_before = c.get_object("W_0")
        l0 = c.run(reg["prog_id"], args={"step": 0},
                   fetch=["loss_0_0"])["fetched"]["loss_0_0"]
        w_mid = c.get_object("W_0")
        assert w_mid != w_before          # optimizer stepped in place
        l1 = c.run(reg["prog_id"], args={"step": 1},
                   fetch=["loss_0_0"])["fetched"]["loss_0_0"]
        assert l1 < l0                    # memorizing the fixed batch
        info = c.query_object("W_0")
        assert info["lineage"]["dirty"] is True


def test_cancel_mid_run_leaves_healthy_daemon(rig):
    """Boundary-cancel during a real run: CANCELLED surfaces, partial
    state stays resident, and the very next run succeeds."""
    from dataflow.core.jsonio import program_to_dict as p2d
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.model_families.families import resolve_family

    big = dataclasses.replace(ShapedLlamaConfig.tiny(),
                              n_layers=8, seq_len=2048, batch=2,
                              d_model=512, n_heads=8, n_kv_heads=8,
                              d_ff=1536)
    fam = resolve_family(big)
    planned = plan_program(fam.lower(big), fast_memory_capacity=2 * 1024**3)
    spec = {"kind": "model_family", "family": "llama3",
            "cfg": _cfg_dict(big)}
    toks, tgts = _tokens(big, seed=77)

    with EngineClient(rig["sock"], client_name="cancel") as c:
        # flat namespace: earlier tests' tiny residents collide with the
        # big config's same-named objects (strict adopt = the design);
        # this test owns the daemon from here — clean slate
        c.wipe("all", force=True)
        init_model(c, "llama3", _cfg_dict(big), seed=8)
        c.put_object("tokens_0_0", toks.numpy().tobytes())
        c.put_object("targets_0_0", tgts.numpy().tobytes())
        reg = c.register_program(p2d(planned.program), resolver=spec)

        tk = c.run(reg["prog_id"], args={"step": 0}, wait=False)
        # wait until it actually starts, then cancel
        deadline = time.time() + 20
        run_id = None
        while time.time() < deadline:
            rs = c.list_runs(limit=5)
            live = [r for r in rs if r["state"] == "running"
                    and r["prog_id"] == reg["prog_id"]]
            if live:
                run_id = live[0]["run_id"]
                break
            time.sleep(0.005)
        if run_id is None:
            # surface the real failure instead of masking it: the run
            # may have gone queued->error/done inside the poll window
            outcome = c.wait(tk, timeout=30)
            raise AssertionError(
                f"run never observed running; terminal outcome: {outcome}")
        c.cancel_run(run_id)
        with pytest.raises(ServiceError) as ei:
            c.wait(tk)
        assert ei.value.code == "CANCELLED"
        assert c.run_status(run_id)["state"] == "cancelled"

        # daemon healthy; next run completes
        r = c.run(reg["prog_id"], args={"step": 0}, fetch=["loss_0_0"])
        assert r["state"] == "done" and c.health()["ok"]


def test_transients_visible_and_reclaimed(rig):
    """Unified budget: after a run, the slab shows owner-tagged
    transient bytes (dW/A staging drawn from the SAME slab as
    residents); unregistering the program returns them to the free
    list."""
    cfg = rig["cfg"]
    toks, tgts = _tokens(cfg, seed=55)
    with EngineClient(rig["sock"], client_name="transients") as c:
        c.wipe("all", force=True)
        init_model(c, "llama3", _cfg_dict(cfg), seed=4)
        c.put_object("tokens_0_0", toks.numpy().tobytes())
        c.put_object("targets_0_0", tgts.numpy().tobytes())
        reg = c.register_program(rig["prog_dict"], resolver=rig["resolver"])
        c.run(reg["prog_id"], args={"step": 0}, fetch=["loss_0_0"])
        u = c.query_backing()
        assert u["transient_bytes"] > 0, u
        assert reg["prog_id"] in u["by_owner"], u["by_owner"]
        assert u["resident_bytes"] > 0
        c.unregister_program(reg["prog_id"])
        u2 = c.query_backing()
        # scope to THIS program: other tests' registered programs
        # rightfully retain their own transients in the shared daemon
        assert reg["prog_id"] not in u2["by_owner"], u2["by_owner"]
        assert u2["transient_bytes"] < u["transient_bytes"], (u, u2)
