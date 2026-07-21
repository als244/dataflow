"""Phase C1 property gates for the general packer (pure, no GPU)."""
from __future__ import annotations

import numpy as np
import pytest

from dataflow_training.data.lpt_packing import IGNORE_INDEX, pack_batch


def _mk(rng, n, lo=1, hi=900):
    lens = rng.integers(lo, hi, size=n)
    seqs = []
    for ln in lens:
        tok = rng.integers(0, 50304, size=ln).astype(np.int32)
        tgt = np.roll(tok, -1)
        seqs.append((tok, tgt))
    return seqs


def test_token_conservation_and_order():
    rng = np.random.default_rng(0)
    seqs = _mk(rng, 37)
    step = pack_batch(seqs, t_round=2048)
    got = np.concatenate([
        r.tokens[:r.valid_count] for r in step.rounds])
    assert got.size == step.total_tokens == sum(
        len(t) for t, _ in seqs)
    # multiset identity (order across rounds is packer's choice)
    want = np.concatenate([t for t, _ in seqs])
    assert np.array_equal(np.sort(got), np.sort(want))


def test_boundaries_and_tail_padding():
    rng = np.random.default_rng(1)
    step = pack_batch(_mk(rng, 23), t_round=2048, s_max=64)
    for r in step.rounds:
        cu = r.cu
        assert cu[0] == 0
        assert np.all(np.diff(cu) >= 0)
        assert cu[r.n_segments] == r.valid_count
        # pads strictly tail-contiguous, ignore-masked
        assert np.all(r.targets[r.valid_count:] == IGNORE_INDEX)
        assert np.all(r.targets[:r.valid_count] != IGNORE_INDEX)
        # sentinel fill
        assert np.all(cu[r.n_segments + (1 if r.valid_count <
                                         step.t_round else 0) + 1:]
                      == step.t_round)


def test_balance_objective():
    rng = np.random.default_rng(2)
    seqs = _mk(rng, 64, lo=100, hi=1500)
    step = pack_batch(seqs, t_round=4096)
    fills = [r.valid_count for r in step.rounds]
    max_len = max(len(t) for t, _ in seqs)
    # LPT guarantee: spread bounded by the largest item
    assert max(fills) - min(fills) <= max_len


def test_exact_fill_when_divisible():
    tok = np.arange(1000, dtype=np.int32)
    seqs = [(tok[:500], tok[:500]), (tok[:500], tok[:500]),
            (tok[:1000], tok[:1000])]
    step = pack_batch(seqs, t_round=1000)   # total exactly 2 rounds
    assert all(r.valid_count == 1000 for r in step.rounds)
    assert step.pad_fraction == 0.0


def test_overflow_split_preserves_content():
    rng = np.random.default_rng(3)
    big = rng.integers(0, 100, size=5000).astype(np.int32)
    step = pack_batch([(big, big)], t_round=2048)
    assert step.n_splits == 1 and len(step.rounds) == 3
    got = np.concatenate([r.tokens[:r.valid_count]
                          for r in step.rounds])
    # split chunks concatenate back to the original IN ORDER within
    # each chunk; multiset identity overall
    assert np.array_equal(np.sort(got), np.sort(big))


def test_overflow_error_policy():
    big = np.zeros(3000, dtype=np.int32)
    with pytest.raises(ValueError, match="exceeds t_round"):
        pack_batch([(big, big)], t_round=2048, on_overflow="error")
    # fragmentation (capacity fits, no single gap does): 1500+1500
    # fill two rounds to 1500 each; the 1000 fits neither whole
    a = np.zeros(1500, dtype=np.int32)
    b = np.zeros(1000, dtype=np.int32)
    with pytest.raises(ValueError, match="fits no round"):
        pack_batch([(a, a), (a, a), (b, b)], t_round=2048,
                   n_rounds=2, on_overflow="error")


def test_fixed_n_rounds_pads_whole_round():
    tok = np.ones(100, dtype=np.int32)
    step = pack_batch([(tok, tok)], t_round=256, n_rounds=3)
    assert len(step.rounds) == 3
    assert step.rounds[1].valid_count == 0
    assert np.all(step.rounds[1].targets == IGNORE_INDEX)


def test_s_max_enforced():
    seqs = [(np.ones(2, dtype=np.int32),) * 2 for _ in range(100)]
    with pytest.raises(ValueError, match="s_max"):
        pack_batch(seqs, t_round=4096, s_max=16)


def test_determinism():
    rng = np.random.default_rng(4)
    seqs = _mk(rng, 41)
    a = pack_batch(seqs, t_round=1024)
    b = pack_batch(seqs, t_round=1024)
    for ra, rb in zip(a.rounds, b.rounds):
        assert np.array_equal(ra.tokens, rb.tokens)
        assert np.array_equal(ra.targets, rb.targets)
        assert np.array_equal(ra.cu, rb.cu)


def test_sum_len_sq_statistic():
    tok = np.ones(64, dtype=np.int32)
    step = pack_batch([(tok, tok), (tok[:32], tok[:32])], t_round=128)
    (r,) = step.rounds
    assert r.sum_len_sq == 64 * 64 + 32 * 32
