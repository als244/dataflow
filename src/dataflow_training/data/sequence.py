"""The data plane's value objects: Sequence, PackedRound, PackedStep.

A Sequence is the unit of data (one document / one record / one
window): int32 tokens with int32 targets (-1 = masked, skipped by the
loss's ignore-index channel) and optional named per-token extras. The
packer groups sequences into fixed-size rounds; a round's content may
be smaller than its buffer (tasks compute over the content rows; the
tail is dead bytes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Sequence:
    tokens: np.ndarray                  # (n,) int32
    targets: np.ndarray                 # (n,) int32; -1 = masked
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def validate_sequence(seq: Sequence, vocab_size: int) -> None:
    """The source-boundary check: every source yields sequences that
    pass this ONCE; the feed and packer never re-validate."""
    n = len(seq)
    if n < 1:
        raise ValueError("empty sequence")
    for name, arr in (("tokens", seq.tokens), ("targets", seq.targets)):
        if arr.dtype != np.int32 or arr.shape != (n,):
            raise ValueError(f"{name}: want (n,) int32, got "
                             f"{arr.dtype} {arr.shape}")
    if int(seq.tokens.min()) < 0 or int(seq.tokens.max()) >= vocab_size:
        raise ValueError(f"token id outside [0, {vocab_size})")
    if int(seq.targets.min()) < -1 or int(seq.targets.max()) >= vocab_size:
        raise ValueError(f"target outside [-1, {vocab_size})")
    for name, arr in seq.extras.items():
        if arr.shape[0] != n:
            raise ValueError(f"extras[{name}]: length {arr.shape[0]} != {n}")


@dataclass(frozen=True)
class PackedRound:
    """One round's fixed-size buffers. INVARIANTS: sum(seq_lens) ==
    content n <= len(tokens); tokens/targets[:n] obey the Sequence
    invariants. The tail [n:] is tokens 0 / targets -1: under content
    re-view it is dead bytes nobody reads; under execute-padding it
    is a masked segment whose loss contribution is exactly zero."""

    tokens: np.ndarray                  # (tokens_per_round,) int32
    targets: np.ndarray                 # (tokens_per_round,) int32; tail -1
    seq_lens: tuple[int, ...]           # CONTENT segments only
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def content(self) -> int:
        return int(sum(self.seq_lens))

    @property
    def fill_ratio(self) -> float:
        return self.content / int(self.tokens.shape[0])

    def bounds(self) -> list[int]:
        """Cumulative segment bounds [0, b1, ..., n] — the
        run_args["seq_lens"] wire form."""
        out = [0]
        for length in self.seq_lens:
            out.append(out[-1] + int(length))
        return out


@dataclass(frozen=True)
class PackedStep:
    """One optimizer step's rounds + the resume point AFTER them."""

    rounds: tuple[PackedRound, ...]
    cursor_after: dict

    @property
    def content(self) -> int:
        return sum(r.content for r in self.rounds)

    @property
    def valid_rows(self) -> int:
        """The loss denominator: positions with a real (>= 0) target."""
        return int(sum(int((r.targets[:r.content] >= 0).sum())
                       for r in self.rounds))
