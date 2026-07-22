"""Packed-args gates: per-round seq_lens via run_args — the
packed path with ZERO program changes (byte-identical lowering),
models on the tested static-ragged code path.

Engine runs the PLAIN uniform program + run_args seq_lens; golden
runs static semantics on the same lens. Loss + every parameter
field after one optimizer step, incl. forced recompute.

Tests:
- test_packed_mode_has_no_lowering_surface: the packed lowering emits no bounds_/positions_ objects, so the program matches the plain uniform one.
- test_packed_args_match_golden: an engine run with run_args seq_lens matches the golden static-ragged loss and every parameter field.
- test_packed_args_with_forced_recompute: the packed path still matches the golden with activations forced to recompute.
- test_no_args_is_legacy: a run with no run_args (uniform sequences) matches the golden.
- test_packed_mode_materializes_positions_once: a one-round packed run builds the round's Segments device tensors exactly once, in the prologue.
- test_resolve_segments_materializes_cuda_positions_and_caches: resolve_segments builds cuda int32 positions that reset per sequence and returns the same cached object on re-call.
- test_workload_segments_derives_max_seqlen_and_mirrors: resolve_segments derives each round's max_len and device cu mirror, leaves run_args untouched, and rejects non-monotonic bounds.
"""
from __future__ import annotations

import pytest
import torch

from dataflow.core.jsonio import program_to_dict
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, lower_llama3
from dataflow_training.testing.gradcheck import check_model_step

LENS = (73, 38, 17)
# boundary notation (convention): cumulative [0, ..., t]
RA = {"seq_lens": {"0": [0, 73, 111, 128]}}


def _cfg(**kw):
    return ShapedLlamaConfig(
        n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
        vocab_size=512, seq_len=128, batch=1, **kw)


def test_packed_mode_has_no_lowering_surface():
    import json

    a = json.dumps(program_to_dict(lower_llama3(_cfg())), sort_keys=True)
    # packed mode has NO lowering surface at all — same cfg, same
    # program; lens arrive at run time
    assert "bounds_" not in a and "positions_" not in a


@pytest.mark.gpu
def test_packed_args_match_golden():
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    check_model_step(_cfg(), run_args=RA, reference_seq_lens=LENS,
                     fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


@pytest.mark.gpu
def test_packed_args_with_forced_recompute():
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    cfg = _cfg()
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    check_model_step(cfg, run_args=RA, reference_seq_lens=LENS,
                     recompute_levels=levels,
                     fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


@pytest.mark.gpu
def test_no_args_is_legacy():
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    check_model_step(_cfg(), fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


@pytest.mark.gpu
def test_packed_mode_materializes_positions_once(monkeypatch):
    """THE implicit-sync gate: a packed run materializes the round's
    Segments (cu/positions device tensors) EXACTLY ONCE — the prologue's
    pinned + non_blocking copy, before task 0 — never per-block or per-round.
    Blocks read seg.positions/seg.cu as fields; a regression that rebuilt a
    device tensor mid-round (a pageable H2D / hidden sync) would bump the
    call count. (The golden reference legitimately materializes on the
    harness side; this gate drives the ENGINE alone.)"""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow_training.data.segments import Segments
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

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
        planned.program, resolver=fam.build_resolver(fam.derive_dims(cfg)),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args=RA)
    # one round -> one distinct segmentation -> a single materialization
    assert calls["n"] == 1
    result.close()
    dry.close()
    from dataflow.runtime.interop import clear_view_cache

    clear_view_cache()
    for buf in values.values():
        backend.free(buf)


class SegmentsProbeCtx:
    """Minimal TaskContext stand-in for resolve_segments: run_args +
    run_values + backend are all it reads."""

    def __init__(self, run_args, backend):
        self.run_args = run_args
        self.run_values = {}
        self.backend = backend


@pytest.mark.gpu
def test_resolve_segments_materializes_cuda_positions_and_caches():
    import torch as _t

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.data.segments import resolve_segments

    ctx = SegmentsProbeCtx({"seq_lens": {"0": [0, 5, 8]}}, CudaBackend())
    seg = resolve_segments(ctx, None, "0")
    assert seg.positions.device.type == "cuda" and seg.positions.dtype == _t.int32
    assert seg.positions.cpu().tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    # cached: the SAME materialized object comes back for the run
    assert resolve_segments(ctx, None, "0") is seg


@pytest.mark.gpu
def test_workload_segments_derives_max_seqlen_and_mirrors():
    """resolve_segments: wire boundaries -> materialized Segments resolved
    WORKLOAD-side (run_args stay opaque to the engine); tight per-round
    max_len, device cu mirror; caller's run_args untouched."""
    import torch as _t

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.data.segments import resolve_segments

    ra = {"step": 3,
          "seq_lens": {"0": [0, 73, 111, 128], "1": [0, 50, 128]}}
    ctx = SegmentsProbeCtx(ra, CudaBackend())
    s0 = resolve_segments(ctx, None, "0")
    s1 = resolve_segments(ctx, None, "1")
    assert s0.max_len == 73 and s1.max_len == 78
    assert "segments" not in ra  # run_args untouched (opaque + immutable)
    assert s0.cu.device.type == "cuda" and s0.cu.dtype == _t.int32
    assert s0.cu.cpu().tolist() == [0, 73, 111, 128]

    with pytest.raises(ValueError):
        resolve_segments(SegmentsProbeCtx({"seq_lens": {"0": [5, 3]}},
                                      CudaBackend()), None, "0")
    with pytest.raises(ValueError):
        resolve_segments(SegmentsProbeCtx({"seq_lens": {"0": [0, 10, 7]}},
                                      CudaBackend()), None, "0")
