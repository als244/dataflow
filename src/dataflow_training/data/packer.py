"""Packer: sequences -> fixed-size rounds -> steps.

Policies:
- ``ffd`` (default): pull a lookahead window (lookahead_mult x the
  step's token budget), first-fit-DECREASING into the step's
  ga_rounds bins (stable sort — ties keep pull order), every round as
  close to full as possible. Sequences that fit nowhere requeue in
  their original pull order and lead the next step. Rounds may be
  UNDER-FULL: seq_lens carries content segments only; buffer tails
  are zero-filled dead bytes.
- ``greedy``: no lookahead — fill each round in feed order; the first
  non-fitting sequence closes the round (and requeues to lead the
  next one). Most corpus-order-preserving.

``allow_round_split=True`` (legacy replication; greedy only): a
sequence crossing the round edge SPLITS — the head fills the round
exactly, the tail continues as the next round's first segment — and
segments chunk at max_seqlen ROUND-LOCALLY (the chunk grid restarts
at each in-round segment start). This reproduces the fixed-token
filtered-walk packing byte-for-byte, so rounds are always exactly
full.

next_step() is deterministic given the feed's hand-out order and the
constructor arguments; cursor_after = feed.cursor() taken AFTER the
requeue, so a resume regenerates the NEXT step exactly.
"""
from __future__ import annotations

import numpy as np

from dataflow_training.data.sequence import PackedRound, PackedStep, Sequence

POLICIES = ("ffd", "greedy")


class DecreasingLength:
    """Stable sort key: longest first, pull order breaking ties."""

    def __init__(self, window):
        self.window = window

    def __call__(self, i: int):
        return (-len(self.window[i]), i)


def concat_round(placed: list[Sequence], tokens_per_round: int,
                 seq_lens: list[int]) -> PackedRound:
    """Materialize one round's fixed-size buffers from placed
    sequences (content = sum of their lengths; zero tail)."""
    tokens = np.zeros(tokens_per_round, dtype=np.int32)
    targets = np.zeros(tokens_per_round, dtype=np.int32)
    extras: dict[str, list[np.ndarray]] = {}
    at = 0
    for seq in placed:
        n = len(seq)
        tokens[at:at + n] = seq.tokens
        targets[at:at + n] = seq.targets
        for key, arr in seq.extras.items():
            extras.setdefault(key, []).append(arr)
        at += n
    packed_extras = {}
    for key, parts in extras.items():
        if len(parts) != len(placed):
            raise ValueError(f"extras[{key}] missing on some sequences "
                             f"in the round")
        packed_extras[key] = np.concatenate(parts)
    return PackedRound(tokens=tokens, targets=targets,
                       seq_lens=tuple(seq_lens), extras=packed_extras)


class Packer:
    def __init__(self, feed, *, tokens_per_round: int, ga_rounds: int,
                 max_seqlen: int, allow_round_split: bool = False,
                 policy: str = "ffd", lookahead_mult: int = 4):
        if policy not in POLICIES:
            raise ValueError(f"policy {policy!r} not in {POLICIES}")
        if allow_round_split and policy != "greedy":
            raise ValueError("allow_round_split requires policy='greedy'")
        if not allow_round_split and max_seqlen > tokens_per_round:
            raise ValueError("max_seqlen > tokens_per_round cannot pack "
                             "without round splitting")
        self.feed = feed
        self.tokens_per_round = int(tokens_per_round)
        self.ga_rounds = int(ga_rounds)
        self.max_seqlen = int(max_seqlen)
        self.allow_round_split = bool(allow_round_split)
        self.policy = policy
        self.lookahead_mult = int(lookahead_mult)

    # ---- greedy (+ optional legacy round-splitting) ----

    def greedy_step(self) -> PackedStep:
        rounds = []
        carry: Sequence | None = None
        for _ in range(self.ga_rounds):
            placed: list[Sequence] = []
            lens: list[int] = []
            room = self.tokens_per_round
            while room > 0:
                seq = carry if carry is not None else self.feed.next_sequence()
                carry = None
                n = len(seq)
                if self.allow_round_split:
                    take = min(n, room)
                    head_t, head_g = seq.tokens[:take], seq.targets[:take]
                    for lo in range(0, take, self.max_seqlen):
                        hi = min(lo + self.max_seqlen, take)
                        placed.append(Sequence(
                            tokens=np.ascontiguousarray(head_t[lo:hi]),
                            targets=np.ascontiguousarray(head_g[lo:hi])))
                        lens.append(hi - lo)
                    room -= take
                    if take < n:
                        carry = Sequence(
                            tokens=np.ascontiguousarray(seq.tokens[take:]),
                            targets=np.ascontiguousarray(seq.targets[take:]))
                else:
                    if n > self.tokens_per_round:
                        raise ValueError(
                            f"sequence of {n} tokens can never fit a "
                            f"{self.tokens_per_round}-token round without "
                            f"round splitting")
                    if n > room:
                        carry = seq          # closes the round under-full
                        break
                    placed.append(seq)
                    lens.append(n)
                    room -= n
            rounds.append(concat_round(placed, self.tokens_per_round, lens))
        if carry is not None:
            self.feed.requeue([carry])
        return PackedStep(rounds=tuple(rounds),
                          cursor_after=self.feed.cursor())

    # ---- first-fit-decreasing over a lookahead window ----

    def ffd_step(self) -> PackedStep:
        budget = self.ga_rounds * self.tokens_per_round
        window: list[Sequence] = []
        pulled = 0
        while pulled < self.lookahead_mult * budget:
            seq = self.feed.next_sequence()
            window.append(seq)
            pulled += len(seq)
        order = sorted(range(len(window)), key=DecreasingLength(window))
        room = [self.tokens_per_round] * self.ga_rounds
        bins: list[list[int]] = [[] for _ in range(self.ga_rounds)]
        placed_ids: set[int] = set()
        for i in order:
            n = len(window[i])
            for b in range(self.ga_rounds):
                if n <= room[b]:
                    bins[b].append(i)
                    room[b] -= n
                    placed_ids.add(i)
                    break
        leftover = [window[i] for i in range(len(window))
                    if i not in placed_ids]
        self.feed.requeue(leftover)
        rounds = []
        for b in range(self.ga_rounds):
            placed = [window[i] for i in bins[b]]
            lens = [len(s) for s in placed]
            rounds.append(concat_round(placed, self.tokens_per_round, lens))
        return PackedStep(rounds=tuple(rounds),
                          cursor_after=self.feed.cursor())

    def next_step(self) -> PackedStep:
        if self.policy == "greedy":
            return self.greedy_step()
        return self.ffd_step()
