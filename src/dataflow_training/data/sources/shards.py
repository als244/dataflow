"""ShardSource: token-shard corpora (the llm.c .bin format).

Two emission modes over one corpus:
- per-document (default): sequences are the corpus's documents in
  order (delimiter-split; the delimiter token never appears as an
  input, and a document's final position targets the NEXT document's
  leading delimiter — the model learns to emit end-of-text). The
  long policy applies per document.
- ``window=N``: sequences are consecutive N-token windows with the
  global shift-by-one targets (fixed-window slicing; delimiters flow
  through as ordinary tokens).

Long policies: ``exclude`` (drop docs over max_seqlen), ``trim``
(truncate), ``chunk`` (split into max_seqlen pieces, positions
restarting per piece), ``whole`` (emit the full document regardless —
ONLY for packers that split at round edges and chunk round-locally,
the legacy-replication path).

Cursors: per-doc = {"doc": ordinal, "piece": i} (piece only under
``chunk``); window = {"pos": global token position}. Both wrap the
epoch deterministically.

The mmap corpus machinery (ShardCorpus, DocIndex) is shared with the
legacy feed module during the migration window.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from dataflow_training.data.fineweb import (
    DEFAULT_ROOT,
    EOT,
    DocIndex,
    ShardCorpus,
)
from dataflow_training.data.sequence import Sequence, validate_sequence


class ShardSource:
    def __init__(self, root: str = DEFAULT_ROOT, *, split: str = "train",
                 max_seqlen: int, long_policy: str = "exclude",
                 vocab_size: int, window: int | None = None):
        self.corpus = ShardCorpus(root or DEFAULT_ROOT, split)
        self.max_seqlen = int(max_seqlen)
        self.long_policy = long_policy
        self.vocab_size = int(vocab_size)
        self.window = int(window) if window else None
        self.eot_id = EOT
        self.index = None if self.window else DocIndex(self.corpus)

    # ---- per-document mode ----

    def doc_count(self) -> int:
        """Content spans between delimiters, INCLUDING the pre-first-
        delimiter prefix (a corpus need not open with a delimiter) —
        len(eots) + 1 spans, some possibly empty."""
        return int(len(self.index.eots)) + 1

    def doc_span(self, m: int) -> tuple[int, int]:
        """Raw [start, end) of document m's CONTENT (delimiters
        excluded). m = 0 is the prefix before the first delimiter;
        span m ends at delimiter m (corpus end for the last)."""
        eots = self.index.eots
        start = 0 if m == 0 else int(eots[m - 1]) + 1
        end = int(eots[m]) if m < len(eots) else self.corpus.total_tokens
        return start, end

    def doc_pieces(self, m: int) -> list[Sequence]:
        """Document m as 0..k sequences per the long policy."""
        start, end = self.doc_span(m)
        n = end - start
        if n < 1:
            return []                    # empty doc (adjacent delimiters)
        if n > self.max_seqlen and self.long_policy == "exclude":
            return []
        w = self.corpus.read(start, n + 1)     # +1: shifted targets
        toks = w[:-1].astype(np.int32)
        tgts = w[1:].astype(np.int32)
        if n > self.max_seqlen and self.long_policy == "trim":
            n = self.max_seqlen
            toks, tgts = toks[:n], tgts[:n]
        if self.long_policy in ("whole", "exclude", "trim") or n <= self.max_seqlen:
            return [Sequence(tokens=np.ascontiguousarray(toks),
                             targets=np.ascontiguousarray(tgts))]
        out = []
        for lo in range(0, n, self.max_seqlen):    # chunk
            hi = min(lo + self.max_seqlen, n)
            out.append(Sequence(tokens=np.ascontiguousarray(toks[lo:hi]),
                                targets=np.ascontiguousarray(tgts[lo:hi])))
        return out

    # ---- window mode ----

    def window_sequence(self, pos: int) -> Sequence:
        w = self.corpus.read(pos, self.window + 1)
        return Sequence(tokens=w[:-1].astype(np.int32),
                        targets=w[1:].astype(np.int32))

    # ---- the DataSource surface ----

    def sequences(self, cursor: dict | None) -> Iterator[tuple[Sequence, dict]]:
        if self.window:
            total = self.corpus.total_tokens
            pos = int(cursor["pos"]) if cursor else 0
            while True:
                seq = self.window_sequence(pos % total)
                pos = (pos + self.window) % total
                validate_sequence(seq, self.vocab_size)
                yield seq, {"pos": pos}
            return
        ndocs = self.doc_count()
        m = int(cursor["doc"]) if cursor else 0
        first_piece = int(cursor.get("piece", 0)) if cursor else 0
        while True:
            pieces = self.doc_pieces(m % ndocs)
            next_doc = (m + 1) % ndocs
            for i, seq in enumerate(pieces):
                if i < first_piece:
                    continue
                validate_sequence(seq, self.vocab_size)
                if i + 1 < len(pieces):
                    yield seq, {"doc": m % ndocs, "piece": i + 1}
                else:
                    yield seq, {"doc": next_doc, "piece": 0}
            first_piece = 0
            m = next_doc

    def describe(self) -> dict:
        return {"scheme": "shards", "root": self.corpus.root,
                "split": self.corpus.split,
                "total_tokens": self.corpus.total_tokens,
                "vocab_size": self.vocab_size, "eot_id": self.eot_id,
                "tokenizer": None, "deterministic": True,
                "max_seqlen": self.max_seqlen,
                "long_policy": self.long_policy, "window": self.window}
