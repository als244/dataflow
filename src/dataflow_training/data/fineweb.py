"""Deterministic token stream over the llm.c fineweb10B shards.

The corpus is a directory of ``fineweb_{split}_*.bin`` shards in the llm.c
gpt2 format: a 1024-byte header (``int32[256]``; ``[0]`` magic 20240520,
``[1]`` version 1, ``[2]`` token count) followed by that many ``uint16``
gpt2 token ids. gpt2's vocabulary is 50257 (max id 50256); we train at
vocab 50304 (padded for GEMM alignment), so every id is a valid,
in-range target and no ignore-index masking is needed.

The stream is the ONE data seam shared by both training backends (the
pytorch reference and the engine). It is a PURE function of the call
index ``k`` — no RNG, no hidden cursor state — so the reference and the
engine see byte-identical ``(tokens, targets)`` for the same ``k``. The
driver calls it with ``k = step * grad_accum_rounds + round`` exactly as
the in-process loop does, so sequential ``k`` walks the corpus in order.

Packing here is the plain fixed-block scheme (llm.c / nanoGPT): one round
is ``tokens_per_round`` contiguous corpus tokens, reshaped by the consumer
into ``batch`` sequences of ``seq_len`` (uniform segments). Targets are the
contiguous next token (a global shift by one), so the last position of each
sequence predicts the first token of the next contiguous block — the
standard next-token objective. The corpus is treated as circular, so any
``k`` is defined (long runs wrap the epoch deterministically).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_BYTES = 1024
HEADER_INTS = 256
MAGIC = 20240520
TOKEN_DTYPE = np.uint16

# In-repo corpus (a symlink to the llm.c shards). The /mnt fineweb_edu copy
# is llama3-128K-tokenized — the WRONG one; always use this.
DEFAULT_ROOT = str(
    Path(__file__).resolve().parents[3] / "datasets" / "fineweb10B"
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


@dataclass
class FinewebStream:
    """``stream(k) -> (tokens, targets)`` int32 CPU tensors, shape
    ``(tokens_per_round,)``.

    Round ``k`` is corpus tokens ``[k*T : k*T + T]`` with ``targets`` the
    global shift by one (``[k*T + 1 : k*T + T + 1]``). Pure in ``k``.
    ``start_token`` offsets the whole stream (e.g. a held-out val cursor).
    """

    corpus: ShardCorpus
    tokens_per_round: int
    start_token: int = 0

    def window(self, k: int) -> np.ndarray:
        """The ``T+1`` raw uint16 tokens backing round ``k`` (tokens+shift)."""
        T = self.tokens_per_round
        start = self.start_token + k * T
        return self.corpus.read(start, T + 1)

    def __call__(self, k: int):
        import torch

        w = self.window(k)
        # uint16 -> int32 (torch has no uint16); ids are <= 50256, in range.
        tokens = torch.from_numpy(w[:-1].astype(np.int32))
        targets = torch.from_numpy(w[1:].astype(np.int32))
        return tokens, targets


def make_stream(tokens_per_round: int, *, root: str | os.PathLike = DEFAULT_ROOT,
                split: str = "train", start_token: int = 0) -> FinewebStream:
    """Convenience: open the corpus and build a stream in one call."""
    return FinewebStream(ShardCorpus(root, split), tokens_per_round,
                         start_token=start_token)


# ============================ doc-aware packing ==============================

EOT = 50256          # gpt2 <|endoftext|> — PREPENDED to every doc in the shards


class DocIndex:
    """Global EOT positions over a ShardCorpus, built lazily per shard
    (one numpy scan of the mmap, ~100 ms per 100M-token shard, cached
    in-process). Provides the filtered->raw coordinate map for the
    doc-aware stream: "filtered" = the corpus with every EOT removed."""

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
class DocAwareFinewebStream:
    """``stream(k) -> (tokens, targets, seq_lens)``: doc-aware packing.

    Round ``k`` is ``tokens_per_round`` consecutive NON-EOT corpus tokens
    (documents in corpus order — the fixed-block token order minus the
    delimiters). Each position's target is its raw next token, so a
    document's final position predicts the NEXT doc's leading EOT (the
    model learns to emit <|endoftext|>) and EOT NEVER appears as an
    input. ``seq_lens`` splits at every document boundary, at
    ``max_seqlen`` (long docs chunk, positions restarting per chunk —
    the fixed-window behavior, localized), and at round edges (a doc may
    continue into the next round as a fresh segment). sum(seq_lens) ==
    tokens_per_round every round — no padding, every target real. Pure
    in ``k``; circular over the filtered corpus.
    """

    corpus: ShardCorpus
    tokens_per_round: int
    max_seqlen: int
    start_token: int = 0            # offset in FILTERED coordinates

    def __post_init__(self):
        self.index = DocIndex(self.corpus)

    def _raw_piece(self, f_start: int, count: int):
        """(inputs, targets, seg_start_flags) for ``count`` filtered
        tokens beginning at filtered ``f_start`` (no filtered wrap
        inside; raw reads may cross shard ends — corpus.read is
        circular). The window BEGINS at a kept token by construction."""
        idx = self.index
        raw_lo = idx.raw_of_filtered(f_start)
        raw_hi = idx.raw_of_filtered(f_start + count - 1) + 1
        w = self.corpus.read(raw_lo, raw_hi - raw_lo + 1)   # +1: shifted targets
        body = w[:-1]
        keep = body != np.uint16(EOT)
        kept = np.flatnonzero(keep)
        if len(kept) != count:
            raise AssertionError(
                f"doc index inconsistent: window holds {len(kept)} kept "
                f"tokens, expected {count}")
        toks = body[kept].astype(np.int32)
        tgts = w[kept + 1].astype(np.int32)
        # a kept token starts a segment iff its raw predecessor is EOT
        # (doc start); the piece's first token starts one anyway (round
        # edge — its predecessor lies outside the window)
        starts = np.zeros(count, dtype=bool)
        starts[0] = True
        starts[1:] = ~keep[kept[1:] - 1]
        return toks, tgts, starts

    def __call__(self, k: int):
        import torch

        T = self.tokens_per_round
        total = self.index.filtered_total
        if T > total:
            raise ValueError(f"round {T} > filtered corpus {total}")
        f0 = (self.start_token + k * T) % total
        if f0 + T <= total:
            toks, tgts, starts = self._raw_piece(f0, T)
        else:              # epoch wrap: two pieces, boundary = segment edge
            n1 = total - f0
            t1, g1, s1 = self._raw_piece(f0, n1)
            t2, g2, s2 = self._raw_piece(0, T - n1)
            toks = np.concatenate([t1, t2])
            tgts = np.concatenate([g1, g2])
            starts = np.concatenate([s1, s2])
        edges = np.append(np.flatnonzero(starts), T)
        lens: list[int] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            n = int(hi - lo)
            while n > self.max_seqlen:     # long docs chunk at max_seqlen
                lens.append(self.max_seqlen)
                n -= self.max_seqlen
            if n:
                lens.append(n)
        return (torch.from_numpy(toks), torch.from_numpy(tgts), tuple(lens))


def make_doc_stream(tokens_per_round: int, max_seqlen: int, *,
                    root: str | os.PathLike = DEFAULT_ROOT,
                    split: str = "train",
                    start_token: int = 0) -> DocAwareFinewebStream:
    """Doc-aware packed stream (EOT-split varlen rounds)."""
    return DocAwareFinewebStream(ShardCorpus(root, split), tokens_per_round,
                                 max_seqlen, start_token=start_token)
