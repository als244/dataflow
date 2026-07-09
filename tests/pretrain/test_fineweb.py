"""Gates for the fineweb token stream: header parse, deterministic +
in-range ids, contiguous next-token targets."""
import numpy as np
import pytest
import torch

from dataflow.pretrain import fineweb


@pytest.fixture(scope="module")
def corpus():
    return fineweb.ShardCorpus(fineweb.DEFAULT_ROOT, "train")


def test_header_parse(corpus):
    magic, version, ntok = fineweb.read_header(corpus.paths[0])
    assert magic == fineweb.MAGIC
    assert version == 1
    assert ntok == 100_000_000  # llm.c 100M-token shards
    assert corpus.total_tokens == sum(corpus.shard_ntok)
    assert corpus.total_tokens > 10_000_000_000  # ~10.25B across 103 shards


def test_stream_deterministic(corpus):
    st = fineweb.FinewebStream(corpus, tokens_per_round=8192)
    for k in (0, 1, 7, 100):
        a = st(k)
        b = st(k)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_ids_in_range_and_dtype(corpus):
    st = fineweb.FinewebStream(corpus, tokens_per_round=8192)
    for k in (0, 3, 50):
        tokens, targets = st(k)
        assert tokens.shape == (8192,) and targets.shape == (8192,)
        assert tokens.dtype == torch.int32 and targets.dtype == torch.int32
        # gpt2 ids <= 50256; we train at vocab 50304 -> always in range
        assert int(tokens.min()) >= 0 and int(tokens.max()) < 50304
        assert int(targets.min()) >= 0 and int(targets.max()) < 50304


def test_targets_are_next_token_shift(corpus):
    st = fineweb.FinewebStream(corpus, tokens_per_round=4096)
    tokens, targets = st(0)
    # within a round, target[i] == token[i+1]
    assert torch.equal(targets[:-1], tokens[1:])


def test_rounds_are_contiguous(corpus):
    """Sequential k walks the corpus with stride tokens_per_round, so the
    last target of round k is the first token of round k+1 (standard llm.c
    contiguous packing)."""
    T = 2048
    st = fineweb.FinewebStream(corpus, tokens_per_round=T)
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
