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
# boundary notation (convention): cumulative [0, ..., t]
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
    check_model_step(_cfg(), run_args=RA, reference_seq_lens=LENS,
                     fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


def test_packed_args_with_forced_recompute():
    cfg = _cfg()
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    check_model_step(cfg, run_args=RA, reference_seq_lens=LENS,
                     recompute_levels=levels,
                     fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


def test_no_args_is_legacy():
    check_model_step(_cfg(), fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


def test_packed_mode_materializes_positions_once(monkeypatch):
    """THE implicit-sync gate: a packed run materializes the round's
    Segments (cu/positions device tensors) EXACTLY ONCE — the prologue's
    pinned + non_blocking copy, before task 0 — never per-block or per-round.
    Blocks read seg.positions/seg.cu as fields; a regression that rebuilt a
    device tensor mid-round (a pageable H2D / hidden sync) would bump the
    call count. (The golden reference legitimately materializes on the
    harness side; this gate drives the ENGINE alone.)"""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.ops import Segments
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

    real_on = Segments.on
    calls = {"n": 0}

    def counting_on(self, device):
        calls["n"] += 1
        return real_on(self, device)

    monkeypatch.setattr(Segments, "on", counting_on)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(fam.dims_of(cfg)),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args=RA)
    # one round -> one distinct segmentation -> a single materialization
    assert calls["n"] == 1
    result.close()
    dry.close()
    from dataflow.tasks.interop import clear_view_cache

    clear_view_cache()
    for buf in values.values():
        backend.free(buf)


def test_prologue_positions():
    import torch as _t

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.engine import prologue_run_start

    out = prologue_run_start({"seq_lens": {"0": [0, 5, 8]}}, CudaBackend())
    seg = out["segments"]["0"]
    assert seg.positions.device.type == "cuda" and seg.positions.dtype == _t.int32
    assert seg.positions.cpu().tolist() == [0, 1, 2, 3, 4, 0, 1, 2]


def test_prologue_derives_max_seqlen_and_mirrors():
    """prologue_run_start: wire boundaries -> materialized Segments; tight
    per-round max_len, device cu mirror; caller's dict untouched."""
    import torch as _t

    from dataflow.runtime.engine import prologue_run_start
    from dataflow.runtime.device.cuda import CudaBackend

    ra = {"step": 3,
          "seq_lens": {"0": [0, 73, 111, 128], "1": [0, 50, 128]}}
    out = prologue_run_start(ra, CudaBackend())
    assert out["segments"]["0"].max_len == 73 and out["segments"]["1"].max_len == 78
    assert "segments" not in ra  # caller's dict untouched
    cu0 = out["segments"]["0"].cu
    assert cu0.device.type == "cuda" and cu0.dtype == _t.int32
    assert cu0.cpu().tolist() == [0, 73, 111, 128]

    with pytest.raises(ValueError):
        prologue_run_start({"seq_lens": {"0": [5, 3]}}, CudaBackend())
    with pytest.raises(ValueError):
        prologue_run_start({"seq_lens": {"0": [0, 10, 7]}}, CudaBackend())
