"""C4 increment-2 gate (GPU): the dynamic-bounds program (bounds/
positions as DATA, single-launch varlen) through the REAL engine vs
the golden model running static semantics on the same lens — loss +
every parameter field after one optimizer step.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from dataflow.training.models.llama3 import ShapedLlamaConfig
from dataflow.training.testing.gradcheck import check_model_step

LENS = (73, 38, 17)


def _cfg(**kw):
    return ShapedLlamaConfig(
        n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
        vocab_size=512, seq_len=128, batch=1, **kw)


def test_dynamic_program_matches_golden():
    check_model_step(_cfg(s_max=16), pk_lens=LENS,
                     fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


def test_dynamic_with_forced_recompute_matches_golden():
    cfg = _cfg(s_max=16)
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    check_model_step(cfg, pk_lens=LENS,
                     recompute_levels=levels,
                     fast_memory_capacity=64 * 1024 * 1024,
                     tol=3e-2).assert_ok()


def test_pk_lens_contract():
    with pytest.raises(ValueError):
        check_model_step(_cfg(s_max=16),
                         fast_memory_capacity=64 * 1024 * 1024)
    with pytest.raises(ValueError):
        check_model_step(_cfg(), pk_lens=LENS,
                         fast_memory_capacity=64 * 1024 * 1024)
