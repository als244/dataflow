"""ShardSource: token-shard corpora, and the mmap machinery under it.

Shard format: a 1024-byte header (int32[256]; [0] magic 20240520,
[1] version, [2] token count) followed by that many uint16 token ids;
a split is a directory of ``fineweb_{split}_*.bin`` files (the naming
is the corpus convention on disk, not an API). Documents are
delimiter-separated (EOT prepended to every doc); ``DocIndex`` maps
delimiter positions lazily per shard.

Emission modes and policies: see class ShardSource below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from dataflow_training.data.sequence import Sequence, validate_sequence

HEADER_BYTES = 1024
HEADER_INTS = 256
MAGIC = 20240520
TOKEN_DTYPE = np.uint16

# In-repo corpus (a symlink to the llm.c shards). The /mnt fineweb_edu copy
# is llama3-128K-tokenized — the WRONG one; always use this.
DEFAULT_ROOT = str(
    Path(__file__).resolve().parents[4] / "datasets" / "fineweb10B"
)


def read_header(path: str | os.PathLike) -> tuple[int, int, int]:
    """(magic, version, ntok) from a shard header; validates the magic."""
    head = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if head.shape[0] < 3:
        raise ValueError(f"{path}: truncated header")
    magic, version, ntok = int(head[0]), int(head[1]), int(head[2])
    if magic != MAGIC:
        raise ValueError(
            f"{path}: bad magic {magic} (expected {MAGIC}); not an llm.c shard"
        )
    expect = HEADER_BYTES + ntok * np.dtype(TOKEN_DTYPE).itemsize
    actual = os.path.getsize(path)
    if actual != expect:
        raise ValueError(
            f"{path}: size {actual} != header-implied {expect} "
            f"(ntok={ntok})"
        )
    return magic, version, ntok


def list_shards(root: str | os.PathLike = DEFAULT_ROOT,
                split: str = "train") -> list[Path]:
    """Sorted ``fineweb_{split}_*.bin`` shard paths under ``root``."""
    root = Path(root)
    shards = sorted(root.glob(f"fineweb_{split}_*.bin"))
    if not shards:
        raise FileNotFoundError(
            f"no fineweb_{split}_*.bin shards under {root}"
        )
    return shards


class ShardCorpus:
    """Read-only view over the concatenation of a split's shards.

    Shards are memory-mapped lazily (a window read touches only a few KB);
    reads are circular over the total token count, so any global start is
    valid. Token ids come back as ``uint16`` exactly as stored.
    """

    def __init__(self, root: str | os.PathLike = DEFAULT_ROOT,
                 split: str = "train"):
        self.root = str(root)
        self.split = split
        self.paths = list_shards(root, split)
        self.shard_ntok: list[int] = []
        for p in self.paths:
            _, _, ntok = read_header(p)
            self.shard_ntok.append(ntok)
        # cumulative starts: cum[i] .. cum[i+1] is shard i's global range
        self._cum = np.zeros(len(self.paths) + 1, dtype=np.int64)
        self._cum[1:] = np.cumsum(self.shard_ntok)
        self._mmaps: list[np.memmap | None] = [None] * len(self.paths)

    @property
    def total_tokens(self) -> int:
        return int(self._cum[-1])

    def _mm(self, i: int) -> np.memmap:
        mm = self._mmaps[i]
        if mm is None:
            mm = np.memmap(self.paths[i], dtype=TOKEN_DTYPE, mode="r",
                           offset=HEADER_BYTES, shape=(self.shard_ntok[i],))
            self._mmaps[i] = mm
        return mm

    def _locate(self, pos: int) -> tuple[int, int]:
        """(shard_index, within_shard_offset) for a global token position."""
        i = int(np.searchsorted(self._cum, pos, side="right")) - 1
        return i, pos - int(self._cum[i])

    def read(self, start: int, length: int) -> np.ndarray:
        """``length`` tokens beginning at global ``start`` (circular)."""
        total = self.total_tokens
        if length > total:
            raise ValueError(f"read {length} > corpus {total} tokens")
        out = np.empty(length, dtype=TOKEN_DTYPE)
        pos = start % total
        filled = 0
        while filled < length:
            i, within = self._locate(pos)
            mm = self._mm(i)
            take = min(self.shard_ntok[i] - within, length - filled)
            out[filled:filled + take] = mm[within:within + take]
            filled += take
            pos = (pos + take) % total
        return out


EOT = 50256          # gpt2 <|endoftext|> — PREPENDED to every doc in the shards


class DocIndex:
    """Global EOT positions over a ShardCorpus, built lazily per shard
    (one numpy scan of the mmap, ~100 ms per 100M-token shard, cached
    in-process). Provides the filtered->raw coordinate map for the
    doc-aware feed: "filtered" = the corpus with every EOT removed."""

    def __init__(self, corpus: ShardCorpus):
        self.corpus = corpus
        self._per_shard: list[np.ndarray | None] = [None] * len(corpus.paths)
        self._eots: np.ndarray | None = None
        self._g: np.ndarray | None = None

    def _shard_eots(self, i: int) -> np.ndarray:
        e = self._per_shard[i]
        if e is None:
            mm = self.corpus._mm(i)
            e = np.flatnonzero(mm == np.uint16(EOT)).astype(np.int64)
            self._per_shard[i] = e
        return e

    @property
    def eots(self) -> np.ndarray:
        if self._eots is None:
            parts = [self._shard_eots(i) + int(self.corpus._cum[i])
                     for i in range(len(self.corpus.paths))]
            self._eots = (np.concatenate(parts) if parts
                          else np.zeros(0, np.int64))
            # g[m] = filtered tokens strictly before the m-th EOT
            self._g = self._eots - np.arange(len(self._eots), dtype=np.int64)
        return self._eots

    @property
    def filtered_total(self) -> int:
        return self.corpus.total_tokens - len(self.eots)

    def raw_of_filtered(self, f: int) -> int:
        """Raw position of the f-th non-EOT token (f in filtered space)."""
        self.eots
        m = int(np.searchsorted(self._g, f, side="right"))
        return f + m


@dataclass


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
