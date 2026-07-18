"""C4 daemon gate (redesigned): per-step DIFFERENT packing via
run_args seq_lens (+ valid_rows for a padded round) — losses
BIT-EQUAL to the in-process engine on identical bytes. No extra
objects; args carry the metadata.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from dataflow.core.jsonio import program_to_dict
from dataflow.service import EngineClient, EngineConfig, Server

T, VOCAB = 128, 512
STEPS = [((73, 38, 17), T), ((50, 41, 37), T), ((60, 31), 91)]


def _cfg():
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    return ShapedLlamaConfig(
        n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
        vocab_size=VOCAB, seq_len=T, batch=1)


def _cfg_dict(cfg):
    return {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len, "batch": cfg.batch}


def _data(lens, valid, seed):
    rng = np.random.default_rng(seed)
    toks = np.zeros(T, dtype=np.int32)
    tgts = np.full(T, -1, dtype=np.int32)
    toks[:valid] = rng.integers(0, VOCAB, valid)
    tgts[:valid] = rng.integers(0, VOCAB, valid)
    return toks, tgts


def _args(k, lens, valid):
    # BOUNDARY notation [0, ..., T]; padded tail (valid < T) is its
    # own segment so attention stays defined there; CE ignores it via
    # targets == -1 + valid_rows normalization
    seg = list(lens) if valid == T else list(lens) + [T - valid]
    b = [0]
    for L in seg:
        b.append(b[-1] + L)
    return {"step": k, "seq_lens": {"0": b},
            "valid_rows": {"0": valid}}


def test_daemon_packed_args_bit_equal(tmp_path):
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.engine import Session
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    cfg = _cfg()
    fam = resolve_family(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=64 * 1024 * 1024)
    prog = planned.program
    dims = fam.dims_of(cfg)

    backend = CudaBackend()
    values = fam.initial_values(prog, cfg, backend, seed=11)
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    session = Session(backend=backend)
    inproc = []
    for k, (lens, valid) in enumerate(STEPS):
        toks, tgts = _data(lens, valid, seed=200 + k)
        torch_view(values["tokens_0_0"], (T,), torch.int32).copy_(
            torch.from_numpy(toks))
        torch_view(values["targets_0_0"], (T,), torch.int32).copy_(
            torch.from_numpy(tgts))
        res = Engine(backend, session=session).execute(
            prog, resolver=fam.build_resolver(dims),
            initial_buffers=values, pool_prewarm=dry.pool_demand,
            run_args=_args(k, lens, valid))
        buf = res.objects.get("loss_0_0").backing.buffer
        inproc.append(float(torch_view(buf, (1,), torch.float32)[0]))
        res.close()
    session.close()

    from dataflow_training.register import register_all
    from dataflow_training.run.driver import init_model

    register_all()      # in-process Server shares this registry
    sock = str(tmp_path / "pa.sock")
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
        with EngineClient(sock, client_name="packedargs") as c:
            init_model(c, "llama3", _cfg_dict(cfg), seed=11)
            reg = c.register_program(program_to_dict(prog),
                                     resolver={"kind": "model_family",
                                               "family": "llama3",
                                               "cfg": _cfg_dict(cfg)})
            daemon = []
            for k, (lens, valid) in enumerate(STEPS):
                toks, tgts = _data(lens, valid, seed=200 + k)
                c.put_object(f"tok@{k}", toks.tobytes())
                c.put_object(f"tgt@{k}", tgts.tobytes())
                r = c.run(reg["prog_id"], args=_args(k, lens, valid),
                          rebind={"tokens_0_0": f"tok@{k}",
                                  "targets_0_0": f"tgt@{k}"},
                          fetch=["loss_0_0"])
                daemon.append(r["fetched"]["loss_0_0"])
            c.unregister_program(reg["prog_id"])
    finally:
        server.state.shutdown_requested.set()
        server.dispatcher.stop()
        if server.store.slab is not None:
            server.store.slab.free()

    assert [round(x, 10) for x in daemon] == \
        [round(x, 10) for x in inproc], (daemon, inproc)
    assert len(set(daemon)) == len(daemon)
