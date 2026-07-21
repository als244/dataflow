"""D-phase gates: DataSource/DataFeed/Packer contracts + the legacy
byte-identity gates (the new pipeline reproduces both legacy feeds
exactly under the legacy flags)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataflow_training.data.feed import DataFeed, PrepackedFeed
from dataflow_training.data.packer import Packer
from dataflow_training.data.sequence import (
    PackedStep,
    Sequence,
    validate_sequence,
)
from dataflow_training.data.sources import parse_spec, resolve_data
from dataflow_training.data.sources.capture import CaptureSource, read_capture
from dataflow_training.data.sources.synthetic import SyntheticSource

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "datasets" / "fineweb10B"
needs_corpus = pytest.mark.skipif(not CORPUS.exists(),
                                  reason="fineweb10B corpus not present")

VOCAB = 50304


def synthetic_feed(mean_len=300, seed=7, max_seqlen=1024,
                   long_policy="exclude", **feed_kw) -> DataFeed:
    src = SyntheticSource(vocab_size=VOCAB, mean_len=mean_len, seed=seed,
                          max_seqlen=max_seqlen, long_policy=long_policy)
    return DataFeed(src, **feed_kw)


# ---------------------------- sources ---------------------------------------

def test_synthetic_determinism_and_cursor_resume():
    src = SyntheticSource(vocab_size=VOCAB, mean_len=200, seed=3,
                          max_seqlen=512)
    it = src.sequences(None)
    first = [next(it) for _ in range(20)]
    it2 = src.sequences(None)
    for seq, cur in first:
        seq2, cur2 = next(it2)
        assert np.array_equal(seq.tokens, seq2.tokens)
        assert cur == cur2
    # resume mid-iteration reproduces the suffix exactly
    seq10, cur10 = first[9]
    it3 = src.sequences(cur10)
    for want, _ in first[10:]:
        got, _ = next(it3)
        assert np.array_equal(want.tokens, got.tokens)
        assert np.array_equal(want.targets, got.targets)


def test_sequence_validation_rejects_bad_ids():
    good = Sequence(tokens=np.zeros(4, np.int32),
                    targets=np.full(4, -1, np.int32))
    validate_sequence(good, VOCAB)
    bad = Sequence(tokens=np.full(4, VOCAB, np.int32),
                   targets=np.zeros(4, np.int32))
    with pytest.raises(ValueError):
        validate_sequence(bad, VOCAB)


def test_spec_parser():
    assert parse_spec("shards:datasets/fineweb10B") == \
        ("shards", "datasets/fineweb10B", {})
    scheme, main, kv = parse_spec("shards:ROOT,window=1024,split=val")
    assert (scheme, main) == ("shards", "ROOT")
    assert kv == {"window": "1024", "split": "val"}
    scheme, main, kv = parse_spec("synthetic:vocab=50304,mean_len=800")
    assert (scheme, main) == ("synthetic", "")
    assert kv == {"vocab": "50304", "mean_len": "800"}
    with pytest.raises(ValueError):
        resolve_data("nope:x", max_seqlen=8, vocab_size=8)


@needs_corpus
def test_shard_source_doc_mode_matches_corpus():
    from dataflow_training.data.sources.shards import ShardSource

    src = ShardSource(str(CORPUS), max_seqlen=1024, long_policy="chunk",
                      vocab_size=VOCAB)
    it = src.sequences(None)
    seqs = [next(it) for _ in range(50)]
    # every emitted token id is a real (non-delimiter) corpus token,
    # targets may be the delimiter (doc-final positions)
    for seq, _ in seqs:
        assert int(seq.tokens.max()) < VOCAB
        assert src.eot_id not in seq.tokens
        assert len(seq) <= 1024
    # piece-accurate cursor: resume from the 20th yield's cursor
    it2 = src.sequences(seqs[19][1])
    for want, _ in seqs[20:]:
        got, _ = next(it2)
        assert np.array_equal(want.tokens, got.tokens)


# ----------------------------- feed -----------------------------------------

def test_feed_requeue_leads_and_cursor_carries_content():
    feed = synthetic_feed()
    a = feed.next_sequence()
    b = feed.next_sequence()
    feed.requeue([a, b])
    assert np.array_equal(feed.next_sequence().tokens, a.tokens)
    cur = feed.cursor()                     # b is still requeued: rides cursor
    assert len(cur["requeued"]) == 1
    feed2 = synthetic_feed(start_cursor=cur)
    assert np.array_equal(feed2.next_sequence().tokens, b.tokens)
    # feed 1 still holds b pending; drain it, then both continue with
    # the same next source sequence
    assert np.array_equal(feed.next_sequence().tokens, b.tokens)
    assert np.array_equal(feed2.next_sequence().tokens,
                          feed.next_sequence().tokens)


def test_capture_roundtrip(tmp_path):
    cap = tmp_path / "cap.bin"
    feed = synthetic_feed(capture=cap)
    seqs = [feed.next_sequence() for _ in range(12)]
    feed.close()
    back = read_capture(cap)
    assert len(back) == 12
    for want, got in zip(seqs, back):
        assert np.array_equal(want.tokens, got.tokens)
        assert np.array_equal(want.targets, got.targets)
    src = CaptureSource(cap, vocab_size=VOCAB)
    replay = [s for s, _ in src.sequences(None)]
    assert len(replay) == 12
    assert np.array_equal(replay[3].tokens, seqs[3].tokens)


# ---------------------------- packer ----------------------------------------

def test_ffd_invariants_and_determinism():
    T, GA = 4096, 4
    packer = Packer(synthetic_feed(), tokens_per_round=T, ga_rounds=GA,
                    max_seqlen=1024)
    packer2 = Packer(synthetic_feed(), tokens_per_round=T, ga_rounds=GA,
                     max_seqlen=1024)
    for _ in range(5):
        step = packer.next_step()
        step2 = packer2.next_step()
        assert len(step.rounds) == GA
        for r, r2 in zip(step.rounds, step2.rounds):
            assert sum(r.seq_lens) == r.content <= T
            assert np.all(r.tokens[r.content:] == 0)          # tail tokens
            assert np.all(r.targets[r.content:] == -1)        # tail masked
            assert np.array_equal(r.tokens, r2.tokens)        # deterministic
            assert r.seq_lens == r2.seq_lens
        assert 0.9 < min(r.fill_ratio for r in step.rounds) <= 1.0


def test_greedy_no_split_defers_and_underfills():
    T, GA = 2048, 2
    feed = synthetic_feed(mean_len=700, seed=11)
    packer = Packer(feed, tokens_per_round=T, ga_rounds=GA,
                    max_seqlen=1024, policy="greedy")
    step = packer.next_step()
    for r in step.rounds:
        assert r.content <= T
        for length in r.seq_lens:
            assert length <= 1024
    # the deferred sequence (if any) leads the next step's first round
    cur = step.cursor_after
    if cur["requeued"]:
        lead = np.asarray(cur["requeued"][0]["tokens"], dtype=np.int32)
        nxt = packer.next_step()
        first_len = nxt.rounds[0].seq_lens[0]
        assert np.array_equal(nxt.rounds[0].tokens[:first_len],
                              lead[:first_len])


def test_cursor_roundtrip_regenerates_next_step():
    T, GA = 4096, 4
    packer = Packer(synthetic_feed(), tokens_per_round=T, ga_rounds=GA,
                    max_seqlen=1024)
    steps = [packer.next_step() for _ in range(4)]
    resumed = Packer(synthetic_feed(start_cursor=steps[1].cursor_after),
                     tokens_per_round=T, ga_rounds=GA, max_seqlen=1024)
    for want in steps[2:]:
        got = resumed.next_step()
        for a, b in zip(want.rounds, got.rounds):
            assert np.array_equal(a.tokens, b.tokens)
            assert np.array_equal(a.targets, b.targets)
            assert a.seq_lens == b.seq_lens


def test_prepacked_feed_bypass():
    packer = Packer(synthetic_feed(), tokens_per_round=1024, ga_rounds=2,
                    max_seqlen=512)
    steps = [packer.next_step() for _ in range(3)]
    pre = PrepackedFeed(steps, start=1)
    assert isinstance(pre.next_step(), PackedStep)
    assert pre.cursor() == {"step": 2}


# ----------------- the legacy byte-identity gates ----------------------------

@needs_corpus
def test_byte_identity_legacy_doc_packing():
    """ShardSource(per-doc, whole) + greedy/allow_round_split
    reproduces make_doc_feed EXACTLY: tokens, targets, and segment
    lens, round for round."""
    from dataflow_training.data.fineweb import make_doc_feed
    from dataflow_training.data.sources.shards import ShardSource

    T, MAX, ROUNDS = 32768, 1024, 200
    legacy = make_doc_feed(T, MAX, root=CORPUS)
    src = ShardSource(str(CORPUS), max_seqlen=MAX, long_policy="whole",
                      vocab_size=VOCAB)
    packer = Packer(DataFeed(src), tokens_per_round=T, ga_rounds=1,
                    max_seqlen=MAX, allow_round_split=True, policy="greedy")
    for k in range(ROUNDS):
        want_tok, want_tgt, want_lens = legacy(k)
        got = packer.next_step().rounds[0]
        assert np.array_equal(got.tokens, want_tok.numpy()), f"round {k}"
        assert np.array_equal(got.targets, want_tgt.numpy()), f"round {k}"
        assert got.seq_lens == tuple(want_lens), f"round {k}"


@needs_corpus
def test_byte_identity_legacy_block_packing():
    """ShardSource(window=seq) + greedy reproduces make_feed exactly
    (uniform segments)."""
    from dataflow_training.data.fineweb import make_feed
    from dataflow_training.data.sources.shards import ShardSource

    SEQ, BATCH, ROUNDS = 1024, 32, 100
    T = SEQ * BATCH
    legacy = make_feed(T, root=CORPUS)
    src = ShardSource(str(CORPUS), max_seqlen=SEQ, vocab_size=VOCAB,
                      window=SEQ)
    packer = Packer(DataFeed(src), tokens_per_round=T, ga_rounds=1,
                    max_seqlen=SEQ, policy="greedy")
    for k in range(ROUNDS):
        want_tok, want_tgt = legacy(k)
        got = packer.next_step().rounds[0]
        assert np.array_equal(got.tokens, want_tok.numpy()), f"round {k}"
        assert np.array_equal(got.targets, want_tgt.numpy()), f"round {k}"
        assert got.seq_lens == tuple([SEQ] * BATCH)
        assert got.content == T
