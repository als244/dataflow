"""Gates for the fineweb token feed: header parse, deterministic +
in-range ids, contiguous next-token targets."""
import numpy as np
import pytest
import torch

from dataflow_training.data import fineweb


@pytest.fixture(scope="module")
def corpus():
    return fineweb.ShardCorpus(fineweb.DEFAULT_ROOT, "train")


def test_header_parse(corpus):
    magic, version, ntok = fineweb.read_header(corpus.paths[0])
    assert magic == fineweb.MAGIC
    assert version == 1
    # llm.c shards are 100M tokens each; a box may hold a SUBSET of
    # the corpus (test rigs ship only the feed head), so gate the
    # per-shard invariant, not the corpus total
    assert ntok == 100_000_000
    assert corpus.total_tokens == sum(corpus.shard_ntok)
    if len(corpus.paths) >= 103:
        assert corpus.total_tokens > 10_000_000_000  # full llm.c corpus
    else:
        # subset rig (e.g. the feed-head shards on a test box): the
        # per-shard invariants above are the load-bearing checks
        assert corpus.total_tokens == 100_000_000 * len(corpus.paths)


def test_feed_deterministic(corpus):
    st = fineweb.FinewebFeed(corpus, tokens_per_round=8192)
    for k in (0, 1, 7, 100):
        a = st(k)
        b = st(k)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_ids_in_range_and_dtype(corpus):
    st = fineweb.FinewebFeed(corpus, tokens_per_round=8192)
    for k in (0, 3, 50):
        tokens, targets = st(k)
        assert tokens.shape == (8192,) and targets.shape == (8192,)
        assert tokens.dtype == torch.int32 and targets.dtype == torch.int32
        # gpt2 ids <= 50256; we train at vocab 50304 -> always in range
        assert int(tokens.min()) >= 0 and int(tokens.max()) < 50304
        assert int(targets.min()) >= 0 and int(targets.max()) < 50304


def test_targets_are_next_token_shift(corpus):
    st = fineweb.FinewebFeed(corpus, tokens_per_round=4096)
    tokens, targets = st(0)
    # within a round, target[i] == token[i+1]
    assert torch.equal(targets[:-1], tokens[1:])


def test_rounds_are_contiguous(corpus):
    """Sequential k walks the corpus with stride tokens_per_round, so the
    last target of round k is the first token of round k+1 (standard llm.c
    contiguous packing)."""
    T = 2048
    st = fineweb.FinewebFeed(corpus, tokens_per_round=T)
    t0, y0 = st(0)
    t1, _ = st(1)
    assert int(y0[-1]) == int(t1[0])


def test_read_circular_wraps(corpus):
    """A start beyond the corpus wraps deterministically."""
    total = corpus.total_tokens
    a = corpus.read(0, 16)
    b = corpus.read(total, 16)  # wraps to 0
    assert np.array_equal(a, b)


def test_read_spans_shard_boundary(corpus):
    """A window straddling a shard boundary stitches correctly."""
    n0 = corpus.shard_ntok[0]
    win = corpus.read(n0 - 4, 8)  # 4 from shard0 tail, 4 from shard1 head
    assert win.shape == (8,)
    assert np.array_equal(win[:4], corpus.read(n0 - 4, 4))
    assert np.array_equal(win[4:], corpus.read(n0, 4))


# ========================= doc-aware packed feed ===========================

def doc_feed(corpus, T=8192, L=1024):
    return fineweb.DocAwareFinewebFeed(corpus, tokens_per_round=T,
                                         max_seqlen=L)


def test_doc_feed_deterministic(corpus):
    st = doc_feed(corpus)
    for k in (0, 1, 7, 100):
        a, b = st(k), st(k)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert a[2] == b[2]


def test_doc_feed_invariants(corpus):
    st = doc_feed(corpus)
    for k in (0, 3, 50, 999):
        tokens, targets, lens = st(k)
        assert tokens.shape == (8192,) and targets.shape == (8192,)
        assert sum(lens) == 8192
        assert all(1 <= n <= 1024 for n in lens)
        # EOT never an input; targets in vocab range (EOT allowed — the
        # doc-final positions PREDICT it)
        assert int((tokens == fineweb.EOT).sum()) == 0
        assert int(targets.min()) >= 0 and int(targets.max()) < 50304
        # a fineweb round of 8192 tokens spans several documents
        assert len(lens) >= 2
        assert int((targets == fineweb.EOT).sum()) >= 1


def test_doc_feed_matches_raw_reconstruction(corpus):
    """Independent numpy reconstruction of round 0: inputs are exactly
    the raw head minus EOTs; each target is its input's raw successor;
    segment starts are exactly {round edge} + {kept tokens whose raw
    predecessor is EOT} + max_seqlen chunk points."""
    T, L = 8192, 1024
    st = doc_feed(corpus, T, L)
    tokens, targets, lens = st(0)
    raw = corpus.read(0, 4 * T)          # ample raw head
    keep = np.flatnonzero(raw[:-1] != fineweb.EOT)[:T]
    assert np.array_equal(tokens.numpy(), raw[keep].astype(np.int32))
    assert np.array_equal(targets.numpy(), raw[keep + 1].astype(np.int32))
    starts = {0}
    for j in range(1, T):
        if raw[keep[j] - 1] == fineweb.EOT:
            starts.add(j)
    expect = []
    edges = sorted(starts) + [T]
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = hi - lo
        while n > L:
            expect.append(L)
            n -= L
        if n:
            expect.append(n)
    assert list(lens) == expect


def test_doc_feed_rounds_contiguous(corpus):
    """Consecutive rounds tile the FILTERED corpus with no gap/overlap."""
    st = doc_feed(corpus, T=4096)
    t0, _, _ = st(0)
    t1, _, _ = st(1)
    raw = corpus.read(0, 4 * 8192)
    keep = np.flatnonzero(raw != fineweb.EOT)[:8192]
    both = np.concatenate([t0.numpy(), t1.numpy()])
    assert np.array_equal(both, raw[keep].astype(np.int32))
