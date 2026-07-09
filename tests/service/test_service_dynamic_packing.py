"""C4 daemon gate (GPU): dynamic-bounds programs through the daemon —
DIFFERENT packing every step via rebind, losses BIT-EQUAL to the
in-process engine on identical bytes.

Step 0: full round (three docs, no padding).
Step 1: DIFFERENT lens WITH a padded tail (ignore-targets) +
        valid_rows via run_args — the ignore-index CE path end to
        end through the daemon.
"""
from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from dataflow.core.jsonio import program_to_dict
from dataflow.service import EngineClient, EngineConfig, Server

T, S_MAX, VOCAB = 128, 16, 512
STEP_PACKS = [((73, 38, 17), T), ((50, 41), 91)]   # (lens, valid)


def _cfg():
    from dataflow.training.models.llama3 import ShapedLlamaConfig

    return ShapedLlamaConfig(
        n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
        vocab_size=VOCAB, seq_len=T, batch=1, s_max=S_MAX)


def _pack(lens, valid, seed):
    rng = np.random.default_rng(seed)
    toks = np.zeros(T, dtype=np.int32)
    tgts = np.full(T, -1, dtype=np.int32)
    toks[:valid] = rng.integers(0, VOCAB, valid)
    tgts[:valid] = rng.integers(0, VOCAB, valid)
    cu = np.full(S_MAX + 1, T, dtype=np.int32)
    cu[0] = 0
    acc = 0
    for j, L in enumerate(lens):
        acc += L
        cu[j + 1] = acc
    pos = np.concatenate(
        [np.arange(L, dtype=np.int32) for L in lens]
        + [np.zeros(T - valid, dtype=np.int32)])
    return toks, tgts, cu, pos


def _cfg_dict(cfg):
    return {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len, "batch": cfg.batch,
            "s_max": cfg.s_max}


def test_daemon_dynamic_packing_bit_equal(tmp_path):
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _cfg()
    fam = resolve_family(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=64 * 1024 * 1024)
    prog = planned.program
    dims = fam.dims_of(cfg)

    # ---------- in-process twin ----------
    backend = CudaBackend()
    values = fam.initial_values(prog, cfg, backend, seed=11)
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    from dataflow.runtime.engine import Session

    session = Session(backend=backend)
    inproc_losses = []
    for k, (lens, valid) in enumerate(STEP_PACKS):
        toks, tgts, cu, pos = _pack(lens, valid, seed=100 + k)
        for oid, arr in (("tokens_0_0", toks), ("targets_0_0", tgts),
                         ("bounds_0_0", cu), ("positions_0_0", pos)):
            torch_view(values[oid], arr.shape, torch.int32).copy_(
                torch.from_numpy(arr))
        result = Engine(backend, session=session).execute(
            prog, resolver=fam.build_resolver(dims),
            initial_buffers=values, pool_prewarm=dry.pool_demand,
            run_args={"step": k,
                      "valid_rows": {"loss_0_0": valid}})
        loss_buf = result.objects.get("loss_0_0").backing.buffer
        inproc_losses.append(
            float(torch_view(loss_buf, (1,), torch.float32)[0]))
        result.close()
    session.close()

    # ---------- daemon twin (same process => bit-equal contract) ----
    sock = str(tmp_path / "dyn.sock")
    server = Server(EngineConfig(socket_path=sock, fake=False,
                                 slab_backing_gib=1.0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(300):
        try:
            with EngineClient(sock, client_name="probe"):
                break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    try:
        with EngineClient(sock, client_name="dynpack") as c:
            c.materialize_group({"kind": "family_init_all",
                                 "family": "llama3",
                                 "cfg": _cfg_dict(cfg), "seed": 11})
            reg = c.register_program(program_to_dict(prog),
                                     resolver={"family": "llama3",
                                               "cfg": _cfg_dict(cfg)})
            assert not reg["bindings"]["missing_inputs"]
            daemon_losses = []
            for k, (lens, valid) in enumerate(STEP_PACKS):
                toks, tgts, cu, pos = _pack(lens, valid, seed=100 + k)
                rebind = {}
                for base, arr in (("tokens_0_0", toks),
                                  ("targets_0_0", tgts),
                                  ("bounds_0_0", cu),
                                  ("positions_0_0", pos)):
                    oid = f"{base}@s{k}"
                    c.put_object(oid, arr.tobytes())
                    rebind[base] = oid
                r = c.run(reg["prog_id"],
                          args={"step": k,
                                "valid_rows": {"loss_0_0": valid}},
                          rebind=rebind, fetch=["loss_0_0"])
                daemon_losses.append(r["fetched"]["loss_0_0"])
            c.unregister_program(reg["prog_id"])
    finally:
        server.state.shutdown_requested.set()
        server.dispatcher.stop()
        if server.store.slab is not None:
            server.store.slab.free()

    assert [round(x, 10) for x in daemon_losses] == \
        [round(x, 10) for x in inproc_losses], \
        (daemon_losses, inproc_losses)
    # sanity: both steps produced finite, distinct losses
    assert all(np.isfinite(daemon_losses))
    assert daemon_losses[0] != daemon_losses[1]
