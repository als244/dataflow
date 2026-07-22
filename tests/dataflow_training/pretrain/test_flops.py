"""Gates for FLOP accounting (lowering/flops.py): the walker reads the
SAME cost_subops the simulator prices — these gates pin the hand
formulas, the varlen quadratic scaling, the effective/hardware attention
split, the opt-policy optimizer bucket, and the completeness walk over
every registered family. CPU-only (metadata arithmetic).

Tests:
- test_gpt2_walker_matches_hand_formula: the walker's per-step effective and hardware totals for gpt2-tiny match the closed-form hand formula (adamw adds no optimizer flops).
- test_attention_split_factors: causal-dense attention bwd effective/hardware equals the fixed ratio and fwd equals the stamped 2*t*s*d.
- test_varlen_quadratic_scaling: halving every segment length halves the attention quadratic mass exactly and leaves matmul flops unchanged; seq_sq_ratio matches.
- test_optimizer_bucket_policy_aware: adamw contributes zero optimizer matmul flops while muon counts Newton-Schulz over the 2D fields into both totals; muon_ns_flops formula and orientation symmetry hold.
- test_hybrid_split_causal_vs_static: for qwen35's hybrid the softmax layers' quadratic mass lands in the varlen-scalable causal buckets while linear-attention scans land in the never-scaled static bucket.
- test_every_family_walks: every registered family's tiny program walks with positive effective flops and hardware >= effective (no unstamped tasks).
"""
from dataclasses import replace

import pytest

pytest.importorskip("torch")

from dataflow_training.lowering.flops import (
    ATTN_BWD_EFFECTIVE_OVER_HARDWARE,
    flop_report,
    muon_ns_flops,
    seq_sq_ratio,
)
from dataflow_training.model_families.families import _FAMILIES, family


def test_gpt2_walker_matches_hand_formula():
    """Walker vs closed-form for gpt2 tiny (all-causal, biased)."""
    fam = family("gpt2")
    cfg = fam.config_type.tiny()
    rep = flop_report(cfg, fam.lower(cfg))
    d, ff, L = cfg.d_model, cfg.d_ff, cfg.n_layers
    t, s, V = cfg.max_tokens, cfg.seq_len, cfg.vocab_size
    mm_layer = 2 * t * (cfg.block_params - 2 * d)   # the seed's own form
    attn_f = 2.0 * t * s * d                        # 0.5 * 4 * s^2 * H * hd
    head = 2.0 * t * d * V
    eff, hw = rep.per_step()
    hand_eff = 3 * (mm_layer * L + head) + L * (attn_f + 0.8 * 2.5 * attn_f)
    hand_hw = 3 * (mm_layer * L + head) + L * (attn_f + 2.5 * attn_f)
    assert abs(eff - hand_eff) / hand_eff < 1e-6    # adamw: opt adds 0
    assert abs(hw - hand_hw) / hand_hw < 1e-6


def test_attention_split_factors():
    """bwd effective/hardware = 8/10 on causal-dense kinds; fwd carries
    the triangular 0.5*4 count (== the stamped 2*t*s*d)."""
    fam = family("llama3")
    cfg = fam.config_type.tiny()
    rep = flop_report(cfg, fam.lower(cfg))
    assert rep.attn_bwd_hw > 0
    assert (abs(rep.attn_bwd_eff / rep.attn_bwd_hw
                - ATTN_BWD_EFFECTIVE_OVER_HARDWARE) < 1e-9)
    expect_fwd = 2.0 * cfg.max_tokens * cfg.seq_len * cfg.d_model * cfg.n_layers
    assert abs(rep.attn_fwd - expect_fwd) / expect_fwd < 1e-6


def test_varlen_quadratic_scaling():
    """Halving every segment length halves the quadratic mass exactly;
    matmul flops are unaffected."""
    fam = family("gpt2")
    cfg = fam.config_type.tiny()          # seq 64, t 64
    rep = flop_report(cfg, fam.lower(cfg))
    eff_u, hw_u = rep.per_step()
    lens = {"0": [32, 32]}
    eff_r, hw_r = rep.per_step(lens, tokens=cfg.max_tokens,
                               seq_len=cfg.seq_len)
    attn_eff = rep.attn_fwd + rep.attn_bwd_eff
    assert abs((eff_u - eff_r) - 0.5 * attn_eff) / attn_eff < 1e-9
    assert seq_sq_ratio([32, 32], 64, 64) == 0.5
    assert seq_sq_ratio([64], 64, 64) == 1.0


def test_optimizer_bucket_policy_aware():
    """adamw = 0 matmul flops; muon counts NS5 over the 2D fields."""
    fam = family("llama3")
    cfg = fam.config_type.tiny()
    rep_a = flop_report(cfg, fam.lower(cfg))
    assert rep_a.optimizer == 0.0
    cfg_m = replace(cfg, opt_policy="muon")
    rep_m = flop_report(cfg_m, fam.lower(cfg_m))
    assert rep_m.optimizer > 0.0
    # optimizer matmul work counts in BOTH quantities (sim-consistent:
    # the makespan includes optimizer-task time)
    eff_a, hw_a = rep_a.per_step()
    eff_m, hw_m = rep_m.per_step()
    assert abs((eff_m - eff_a) - rep_m.optimizer) < 1e-6 * eff_a
    assert abs((hw_m - hw_a) - rep_m.optimizer) < 1e-6 * hw_a
    # NS5 formula sanity: (m, n) with m <= n
    assert muon_ns_flops(4, 8) == 5 * (2 * 16 * 8 + 2 * 64 + 2 * 16 * 8)
    assert muon_ns_flops(8, 4) == muon_ns_flops(4, 8)   # orientation swap


def test_hybrid_split_causal_vs_static():
    """qwen35 (DeltaNet + gated-attention hybrid): the softmax layers'
    quadratic mass lands in the CAUSAL buckets (8/10 split, varlen-
    scalable) while the linear-attention scans land in the STATIC
    bucket (effective == hardware, NEVER scaled — a linear-in-t
    recurrence has no quadratic mass)."""
    fam = family("qwen35")
    cfg = fam.config_type.tiny()
    rep = flop_report(cfg, fam.lower(cfg))
    assert rep.attn_fwd > 0            # gattn layers, causal
    assert rep.attn_static > 0         # linattn scans, static
    assert (abs(rep.attn_bwd_eff / rep.attn_bwd_hw
                - ATTN_BWD_EFFECTIVE_OVER_HARDWARE) < 1e-9)
    eff_u, hw_u = rep.per_step()
    lens = {"0": [cfg.seq_len // 2] * (2 * cfg.max_tokens // cfg.seq_len)}
    eff_r, hw_r = rep.per_step(lens, tokens=cfg.max_tokens,
                               seq_len=cfg.seq_len)
    causal_eff = rep.attn_fwd + rep.attn_bwd_eff
    # only the causal share halves; the static share is untouched
    assert abs((eff_u - eff_r) - 0.5 * causal_eff) / causal_eff < 1e-9


@pytest.mark.parametrize("name", sorted(_FAMILIES))
def test_every_family_walks(name):
    """Completeness: every registered family's tiny program walks with
    positive effective flops and no unstamped tasks (the tripwire)."""
    fam = family(name)
    cfg = fam.config_type.tiny()
    rep = flop_report(cfg, fam.lower(cfg))
    eff, hw = rep.per_step()
    assert eff > 0 and hw >= eff
