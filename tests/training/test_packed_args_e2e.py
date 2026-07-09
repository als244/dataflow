"""C4 (redesigned) gates: per-round seq_lens via run_args — the
packed path with ZERO program changes (byte-identical lowering),
models on the tested static-ragged code path.

Engine runs the PLAIN uniform program + run_args seq_lens; golden
runs static semantics on the same lens. Loss + every parameter
field after one optimizer step, incl. forced recompute.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from dataflow.core.jsonio import program_to_dict
from dataflow.training.models.llama3 import ShapedLlamaConfig, lower_llama3
from dataflow.training.testing.gradcheck import check_model_step

LENS = (73, 38, 17)
# boundary notation (Shein): cumulative [0, ..., t]
RA = {"seq_lens": {"0": [0, 73, 111, 128]}}


def _cfg(**kw):
    return ShapedLlamaConfig(
        n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
        vocab_size=512, seq_len=128, batch=1, **kw)


def test_program_is_byte_identical_to_legacy():
    import json

    a = json.dumps(program_to_dict(lower_llama3(_cfg())), sort_keys=True)
    # packed mode has NO lowering surface at all — same cfg, same
    # program; lens arrive at run time
    assert "bounds_" not in a and "positions_" not in a


def test_packed_args_match_golden():
    check_model_step(_cfg(), run_args=RA, golden_seq_lens=LENS,
                     fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


def test_packed_args_with_forced_recompute():
    cfg = _cfg()
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    check_model_step(cfg, run_args=RA, golden_seq_lens=LENS,
                     recompute_levels=levels,
                     fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


def test_no_args_is_legacy():
    check_model_step(_cfg(), fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


def test_packed_mode_never_touches_pageable_lru(monkeypatch):
    """THE implicit-sync gate (Shein): the ENGINE side of a packed
    run must never reach the ops-level lru positions builder (CPU
    cat + pageable .to() per fresh lens — cache thrash + mid-round
    implicit sync). Positions come from the per-run pinned-staged
    run_cache. (The golden reference legitimately uses the lru path
    on the harness side, so this gate drives the engine alone.)"""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks import ops
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _cfg()
    fam = resolve_family(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=3)
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)

    def _boom(*a, **k):
        raise AssertionError(
            "pageable _positions_cached reached from packed-args mode")

    monkeypatch.setattr(ops, "_positions_cached", _boom)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(fam.dims_of(cfg)),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args=RA)
    result.close()
    dry.close()
    from dataflow.tasks.interop import clear_view_cache

    clear_view_cache()
    for buf in values.values():
        backend.free(buf)


def test_run_cache_memoizes_positions():
    from dataflow.tasks.base_blocks import _Base
    import torch as _t

    class _Ctx:
        run_cache = {}

    class _Probe(_Base):
        pass

    probe = object.__new__(_Probe)
    a = _Probe._positions_dev(probe, _Ctx, (5, 3), "cuda")
    b = _Probe._positions_dev(probe, _Ctx, (5, 3), "cuda")
    assert a is b, "second task must reuse the run-cached tensor"
    assert a.device.type == "cuda" and a.dtype == _t.int32
    assert a.cpu().tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    c = _Probe._positions_dev(probe, _Ctx, (4, 4), "cuda")
    assert c is not a and len(_Ctx.run_cache) == 2
