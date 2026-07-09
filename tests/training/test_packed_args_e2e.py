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
RA = {"seq_lens": {"0": list(LENS)}}


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
