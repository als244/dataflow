"""Snapshot / restore / read-leases (GPU).

Load-bearing gate: checkpoint round-trip — train 2 steps, snapshot,
wipe everything, restore, train 2 more ⇒ losses continue EXACTLY as
an uninterrupted 4-step run (same process ⇒ bit-equal). Plus: the
dedup soundness pair (clean dup dedups; parent-mutated dup does NOT
— the version-counter rule), lease-parked writers waking on release,
and client_meta round-trip.

Tests:
- test_checkpoint_roundtrip_bit_continuity: a snapshot/wipe/restore split at step 2 reproduces the uninterrupted 4-step loss sequence and round-trips client_meta.
- test_dedup_clean_vs_mutated_parent: a clean duplicate dedups to one payload while mutating the parent via a run drops the dedup count to zero.
- test_leased_writer_parks_until_release: a release on a leased object parks until the lease is dropped, while an unrelated writer proceeds, then completes.
- test_snapshot_status_unknown_id_rejected: snapshot_status on an unknown id raises ServiceError with code UNKNOWN_SNAPSHOT.
"""
from __future__ import annotations

import threading
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cuda.bindings.runtime")  # real (fake=False) boot + CudaBackend
pytest.importorskip("dataflow_sim")           # rig plans the program (plan_program)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.sim,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

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
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.lowering.planning import plan_program

    tmp = tmp_path_factory.mktemp("svc_snap")
    sock = str(tmp / "snap.sock")
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
    yield {"sock": sock, "server": server, "cfg": cfg,
           "prog_dict": program_to_dict(planned.program),
           "resolver": {"kind": "model_family", "family": "llama3",
                        "cfg": _cfg_dict(cfg)},
           "tmp": tmp}

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


def _fresh_state(c, rig, seed=7):
    c.wipe("all", force=True)
    init_model(c, "llama3", rig["resolver"]["cfg"], seed=seed)
    toks, tgts = _tokens(rig["cfg"], seed=31)
    c.put_object("tokens_0_0", toks.numpy().tobytes())
    c.put_object("targets_0_0", tgts.numpy().tobytes())
    return c.register_program(rig["prog_dict"], resolver=rig["resolver"])


def _steps(c, prog_id, ks):
    out = []
    for k in ks:
        r = c.run(prog_id, args={"step": k}, fetch=["loss_0_0"])
        out.append(r["fetched"]["loss_0_0"])
    return out


def test_checkpoint_roundtrip_bit_continuity(rig):
    with EngineClient(rig["sock"], client_name="roundtrip") as c:
        # uninterrupted reference: 4 steps
        reg = _fresh_state(c, rig)
        ref = _steps(c, reg["prog_id"], [0, 1, 2, 3])
        c.unregister_program(reg["prog_id"])

        # interrupted twin: 2 steps -> snapshot -> wipe -> restore -> 2
        reg = _fresh_state(c, rig)
        first = _steps(c, reg["prog_id"], [0, 1])
        dest = str(rig["tmp"] / "ck2")
        s = c.snapshot("all", dest,
                       client_meta={"step": 2, "cursor": [3, 128]},
                       label="step2")
        done = c.wait_snapshot(s["snap_id"])
        assert done["state"] == "done", done
        c.unregister_program(reg["prog_id"])
        c.wipe("all", force=True)

        r = c.restore_snapshot(dest)
        assert r["client_meta"] == {"step": 2, "cursor": [3, 128]}
        reg2 = c.register_program(rig["prog_dict"],
                                  resolver=rig["resolver"])
        assert not reg2["bindings"]["missing_inputs"]
        second = _steps(c, reg2["prog_id"], [2, 3])
        c.unregister_program(reg2["prog_id"])

        got = first + second
        assert [round(x, 10) for x in got] == \
            [round(x, 10) for x in ref], (got, ref)


def test_dedup_clean_vs_mutated_parent(rig):
    with EngineClient(rig["sock"], client_name="dedup") as c:
        reg = _fresh_state(c, rig)
        # find one weight object to duplicate
        w = next(o["id"] for o in c.list_objects("W_*"))
        c.duplicate_object(w, f"{w}@ck")

        # clean dup + untouched parent => ONE payload (dedup)
        s1 = c.snapshot("all", str(rig["tmp"] / "dd1"))
        assert s1["n_deduped"] == 1, s1
        c.wait_snapshot(s1["snap_id"])

        # a run MUTATES the parent (bound resident) => dedup must die
        _steps(c, reg["prog_id"], [0])
        s2 = c.snapshot("all", str(rig["tmp"] / "dd2"))
        assert s2["n_deduped"] == 0, s2
        c.wait_snapshot(s2["snap_id"])
        c.unregister_program(reg["prog_id"])


def test_leased_writer_parks_until_release(rig):
    store = rig["server"].store
    with EngineClient(rig["sock"], client_name="leases") as c:
        c.wipe("all", force=True)
        c.put_object("lease_probe", b"\x01" * 4096)
        store.acquire_leases(["lease_probe"])
        try:
            tk = c.release_object("lease_probe", wait=False)
            time.sleep(0.3)
            assert not tk.done.is_set(), \
                "leased release must PARK, not complete"
            # unrelated writer proceeds while the other is parked
            c.put_object("bystander", b"\x02" * 4096)
        finally:
            store.release_leases(["lease_probe"])
        c.wait(tk, timeout=10)
        assert c.query_object("lease_probe") is None
        c.release_object("bystander")


def test_snapshot_status_unknown_id_rejected(rig):
    with EngineClient(rig["sock"], client_name="status") as c:
        with pytest.raises(ServiceError) as ei:
            c.snapshot_status("snap-999999")
        assert ei.value.code == "UNKNOWN_SNAPSHOT"
