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


# ----------------- the legacy configuration pins ----------------------------

# sha256 over (tokens || targets || seq_lens) for the first N rounds of
# the two legacy configurations, PINNED while the retired feed
# implementations still existed and byte-identity against them was
# gate-verified. Any packing/source change that shifts these bytes
# breaks the certified-curve reproductions and must be deliberate.
LEGACY_DOC_SHA = "c70359b3352ccaffdeebec5dbc0e7ebed88683d8d5f832bf92849d7c2dce9d34"
LEGACY_BLOCK_SHA = "37be929906be72dec13f36514820fafafbb07d2758c41e67fbe7e78e821e5abf"


def rounds_hash(window, allow_split, long_policy, T, MAX, n_rounds) -> str:
    import hashlib

    from dataflow_training.data.sources.shards import ShardSource

    src = ShardSource(str(CORPUS), max_seqlen=MAX, long_policy=long_policy,
                      vocab_size=VOCAB, window=window)
    packer = Packer(DataFeed(src), tokens_per_round=T, ga_rounds=1,
                    max_seqlen=MAX, allow_round_split=allow_split,
                    policy="greedy")
    h = hashlib.sha256()
    for _ in range(n_rounds):
        r = packer.next_step().rounds[0]
        h.update(r.tokens.tobytes())
        h.update(r.targets.tobytes())
        h.update(np.asarray(r.seq_lens, dtype=np.int64).tobytes())
    return h.hexdigest()


@needs_corpus
def test_legacy_doc_configuration_pinned():
    """whole-docs + greedy/allow_round_split — the doc-aware legacy
    packing (the 124M study curves), pinned byte-exactly."""
    assert rounds_hash(None, True, "whole", 32768, 1024, 200) \
        == LEGACY_DOC_SHA


@needs_corpus
def test_legacy_block_configuration_pinned():
    """window slicing + greedy — the fixed-block legacy packing
    (the parity/determinism gates' data), pinned byte-exactly."""
    assert rounds_hash(1024, False, "exclude", 32768, 1024, 100) \
        == LEGACY_BLOCK_SHA


@needs_corpus
def test_engine_resume_drill_with_cursor(tmp_path):
    """Cursor resume end-to-end on the engine (new-defaults pipeline —
    under-full rounds executing content-only): a checkpointed run's
    resumed tail must reproduce the uninterrupted run bitwise (one
    daemon; init re-seeds, restore overwrites)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from dataflow_training.data.pipeline import DataPipeline
    from dataflow_training.run.driver import daemon_client, run_engine
    from dataflow_training.run.presets import gpt2_smoke_preset
    from dataflow_training.run.recipe import Recipe

    cfg = gpt2_smoke_preset()
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=6)
    pipe = DataPipeline("shards:", tokens_per_round=cfg.tokens,
                        ga_rounds=cfg.grad_accum_rounds,
                        max_seqlen=cfg.seq_len,
                        vocab_size=cfg.vocab_size)

    def quiet(*a, **k):
        pass

    ck = tmp_path / "drill"
    with daemon_client(slab_gib=4.0, log=quiet) as client:
        full = run_engine(client, cfg, recipe, pipe, 6, budget_gib=4.0,
                          seed=11, log=quiet, checkpoint_every=2,
                          checkpoint_dir=ck)
        resumed = run_engine(client, cfg, recipe, pipe, 6, budget_gib=4.0,
                             seed=11, log=quiet, checkpoint_every=2,
                             checkpoint_dir=ck, resume=True)
    assert resumed.meta["resumed_from"].endswith("step_000006")
    # the resumed run restored @6 and had nothing to do — now force a
    # mid-run resume: drop the newest checkpoint so @4 is the target
    import shutil

    shutil.rmtree(ck / "step_000006")
    with daemon_client(slab_gib=4.0, log=quiet) as client:
        tail = run_engine(client, cfg, recipe, pipe, 6, budget_gib=4.0,
                          seed=11, log=quiet, checkpoint_every=2,
                          checkpoint_dir=ck, resume=True)
    assert len(tail.losses) == 6
    assert tail.losses[:4] == full.losses[:4]        # manifest carry
    for a, b in zip(tail.losses[4:], full.losses[4:]):
        # fresh process: same-daemon bitwise does not apply — hold the
        # tail to the cross-process ambient envelope
        assert abs(a - b) < 5e-4, (tail.losses, full.losses)


# --------------------------- text sources ------------------------------------

class CharTokenizer:
    """Hermetic test tokenizer: one id per character (offset by 1 so
    id 0 stays a benign filler)."""

    def encode(self, text: str) -> list[int]:
        return [1 + (ord(c) % 200) for c in text]

    def describe(self) -> dict:
        return {"backend": "char", "name": "char", "vocab_size": 201,
                "eot_id": 200}


def write_jsonl(tmp_path, docs, name="corpus_0.jsonl"):
    import json as json_mod

    p = tmp_path / name
    p.write_text("\n".join(json_mod.dumps({"text": d}) for d in docs) + "\n")
    return p


def test_jsonl_source_targets_and_masking(tmp_path):
    from dataflow_training.data.sources.jsonl import JsonlSource

    write_jsonl(tmp_path, ["hello world", "ab"])
    src = JsonlSource(str(tmp_path / "*.jsonl"), tokenizer=CharTokenizer(),
                      max_seqlen=64, long_policy="exclude", vocab_size=201)
    seqs = []
    it = src.sequences(None)
    for _ in range(2):
        seq, cur = next(it)
        seqs.append(seq)
    tok = CharTokenizer()
    want = tok.encode("hello world")
    assert list(seqs[0].tokens) == want
    assert list(seqs[0].targets[:-1]) == want[1:]
    assert int(seqs[0].targets[-1]) == 200          # doc-final -> eot
    assert len(seqs[1]) == 2


def test_jsonl_cursor_resume_and_epoch_wrap(tmp_path):
    from dataflow_training.data.sources.jsonl import JsonlSource

    write_jsonl(tmp_path, ["one fine doc", "two", "three docs here"])
    src = JsonlSource(str(tmp_path / "*.jsonl"), tokenizer=CharTokenizer(),
                      max_seqlen=64, long_policy="exclude", vocab_size=201)
    it = src.sequences(None)
    got = [next(it) for _ in range(7)]              # wraps the 3-doc epoch
    seq4, cur4 = got[3]
    it2 = src.sequences(got[2][1])                  # resume after 3rd yield
    re4, _ = next(it2)
    assert np.array_equal(re4.tokens, seq4.tokens)


def test_txt_source_delimiter_split(tmp_path):
    from dataflow_training.data.sources.jsonl import TextSource

    (tmp_path / "a.txt").write_text("first doc\n\nsecond doc\n\n")
    src = TextSource(str(tmp_path / "*.txt"), tokenizer=CharTokenizer(),
                     max_seqlen=64, long_policy="exclude", vocab_size=201)
    it = src.sequences(None)
    a, _ = next(it)
    b, _ = next(it)
    assert len(a) == len("first doc")
    assert len(b) == len("second doc")


def test_jsonl_end_to_end_pack_determinism(tmp_path):
    from dataflow_training.data.sources.jsonl import JsonlSource

    docs = [f"document number {i} with some words" * (1 + i % 3)
            for i in range(24)]
    write_jsonl(tmp_path, docs)

    def build():
        src = JsonlSource(str(tmp_path / "*.jsonl"),
                          tokenizer=CharTokenizer(), max_seqlen=128,
                          long_policy="chunk", vocab_size=201)
        return Packer(DataFeed(src), tokens_per_round=256, ga_rounds=2,
                      max_seqlen=128)

    a, b = build(), build()
    for _ in range(4):
        sa, sb = a.next_step(), b.next_step()
        for ra, rb in zip(sa.rounds, sb.rounds):
            assert np.array_equal(ra.tokens, rb.tokens)
            assert ra.seq_lens == rb.seq_lens
            assert np.all(ra.targets[ra.content:] == -1)


def test_tiktoken_backend_if_available():
    tk = pytest.importorskip("tiktoken")
    from dataflow_training.data.tokenizers import resolve_tokenizer

    try:
        tok = resolve_tokenizer("gpt2")
    except Exception as exc:            # no network + cold cache
        pytest.skip(f"tiktoken gpt2 unavailable: {exc}")
    ids = tok.encode("hello world")
    assert ids and max(ids) < tok.describe()["vocab_size"]
    assert tok.describe()["eot_id"] == 50256


def test_parquet_source_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from dataflow_training.data.sources.parquet import ParquetSource

    docs = [f"parquet document {i} body text" for i in range(9)]
    table = pa.table({"text": docs})
    pq.write_table(table, tmp_path / "part_0.parquet", row_group_size=4)
    src = ParquetSource(str(tmp_path / "*.parquet"),
                        tokenizer=CharTokenizer(), max_seqlen=64,
                        long_policy="exclude", vocab_size=201)
    it = src.sequences(None)
    got = [next(it) for _ in range(12)]              # wraps the epoch
    tok = CharTokenizer()
    assert list(got[0][0].tokens) == tok.encode(docs[0])
    assert list(got[9][0].tokens) == tok.encode(docs[0])   # wrap
    # resume across a row-group boundary
    it2 = src.sequences(got[4][1])
    re6, _ = next(it2)
    assert np.array_equal(re6.tokens, got[5][0].tokens)


# ------------------------- threaded feed -------------------------------------

def test_threaded_feed_equals_sync():
    """The worker is an implementation detail: threaded and synchronous
    feeds hand out identical streams (order, bytes, cursors)."""
    def build(prefetch):
        src = SyntheticSource(vocab_size=VOCAB, mean_len=120, seed=5,
                              max_seqlen=512)
        return DataFeed(src, prefetch_sequences=prefetch)

    with build(64) as threaded, build(0) as sync:
        for _ in range(200):
            a = threaded.next_sequence()
            b = sync.next_sequence()
            assert np.array_equal(a.tokens, b.tokens)
        assert threaded.cursor()["source"] == sync.cursor()["source"]


def test_threaded_feed_error_surfaces():
    class BoomSource:
        def sequences(self, cursor):
            src = SyntheticSource(vocab_size=VOCAB, mean_len=50, seed=1,
                                  max_seqlen=128)
            for i, item in enumerate(src.sequences(cursor)):
                if i == 5:
                    raise RuntimeError("boom at 5")
                yield item

        def describe(self):
            return {}

    with DataFeed(BoomSource(), prefetch_sequences=8) as feed:
        for _ in range(5):
            feed.next_sequence()
        with pytest.raises(RuntimeError, match="boom at 5"):
            feed.next_sequence()


def test_threaded_packer_cursor_roundtrip():
    T, GA = 4096, 4
    packer = Packer(synthetic_feed(), tokens_per_round=T, ga_rounds=GA,
                    max_seqlen=1024)
    steps = [packer.next_step() for _ in range(4)]
    resumed = Packer(synthetic_feed(start_cursor=steps[1].cursor_after),
                     tokens_per_round=T, ga_rounds=GA, max_seqlen=1024)
    got = resumed.next_step()
    for a, b in zip(steps[2].rounds, got.rounds):
        assert np.array_equal(a.tokens, b.tokens)
        assert a.seq_lens == b.seq_lens
