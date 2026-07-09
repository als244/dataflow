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
