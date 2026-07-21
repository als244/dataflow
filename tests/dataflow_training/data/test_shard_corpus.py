"""Gates for the shard-corpus machinery: header parse,
deterministic + in-range ids, delimiter indexing."""

import numpy as np

import pytest

import torch

from dataflow_training.data.sources import shards as fineweb

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
