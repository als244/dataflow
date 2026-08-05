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
- test_packed_mode_has_one_round_prologue: a one-round program has exactly one metadata-materialization boundary before its consumers.
- test_resolve_segments_materializes_cuda_positions_and_caches: resolve_segments builds cuda int32 positions that reset per sequence and returns the same cached object on re-call.
- test_workload_segments_derives_max_seqlen_and_mirrors: resolve_segments derives each round's max_len and device cu mirror, leaves run_args untouched, and rejects non-monotonic bounds.
- test_round_segments_prepare_chunk_metadata_once: round preparation builds requested chunk metadata once and later consumers reuse the identical device tensors.
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


def test_packed_mode_has_one_round_prologue():
    """One metadata boundary precedes every consumer in a one-round chain.

    ``test_resolve_segments_materializes_cuda_positions_and_caches`` below
    proves that this boundary's resolver materializes once and returns the
    identical cached object thereafter. This structural assertion avoids
    replacing runtime methods merely to count an implementation detail.
    """
    program = lower_llama3(_cfg())
    prologues = [
        (index, task)
        for index, task in enumerate(program.tasks)
        if task.compute_block_key == "prologue_round"
    ]

    assert len(prologues) == 1
    index, prologue = prologues[0]
    assert index == 0
    assert prologue.block_params["round"] == 0
    assert all(
        task.compute_block_key != "prologue_round"
        for task in program.tasks[index + 1 :]
    )


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


@pytest.mark.gpu
def test_round_segments_prepare_chunk_metadata_once():
    """A family can request chunk metadata at round preparation time; block
    entrypoints then receive ready device tensors and never derive counts from
    CUDA data (an operation that would synchronize the device)."""
    from types import SimpleNamespace

    import torch as _t

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.data.segments import resolve_segments
    from dataflow_training.model_families.qwen35.blocks import _cu_seqlens

    dims = SimpleNamespace(segment_chunk_sizes=(64,))
    ctx = SegmentsProbeCtx(
        {"seq_lens": {"0": [0, 73, 111, 128]}}, CudaBackend())
    seg = resolve_segments(ctx, dims, "0")

    assert seg.cu_int64.dtype == _t.int64
    assert seg.cu_int64.cpu().tolist() == [0, 73, 111, 128]
    assert seg.chunk_indices[64].cpu().tolist() == [
        [0, 0], [0, 1], [1, 0], [2, 0],
    ]
    cu, chunks = _cu_seqlens(seg)
    assert cu is seg.cu_int64
    assert chunks is seg.chunk_indices[64]
    assert resolve_segments(ctx, dims, "0") is seg
