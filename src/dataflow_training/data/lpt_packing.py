"""LPT bin-packing primitives (torch-free): a length-aware packing
policy candidate for the Packer (longest-processing-time bins).
The live packing path is data/packer.py; the IGNORE_INDEX constant
here is the shared masked-target sentinel."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE_INDEX = -1          # target value excluded from the loss
_INT = np.int32


@dataclass(frozen=True)
class PackedRound:
    tokens: np.ndarray         # (t_round,) int32, pads = pad_token
    targets: np.ndarray        # (t_round,) int32, pads = IGNORE_INDEX
    cu: np.ndarray             # (s_max+1,) int32 cumulative boundaries,
    #                            sentinel-padded with t_round; the pad
    #                            tail (if any) is the segment
    #                            [valid_count, t_round)
    n_segments: int            # real segments (pad tail excluded)
    valid_count: int           # real tokens in this round
    sum_len_sq: int            # Σ len_i² over real segments (attn-time
    #                            statistic; the multi-plan trigger key)


@dataclass(frozen=True)
class PackedStep:
    rounds: tuple[PackedRound, ...]
    t_round: int
    total_tokens: int          # real tokens across rounds
    pad_fraction: float
    min_fill: float            # min round valid/t_round
    n_splits: int              # sequences that were cut


def pack_batch(seqs, *, t_round: int, n_rounds: int | None = None,
               s_max: int = 512, on_overflow: str = "split",
               pad_token: int = 0) -> PackedStep:
    """Pack (tokens, targets) pairs into rounds.

    seqs: iterable of (tokens, targets) — 1-D int arrays of equal
    length per pair (targets already shifted/masked by the caller).
    n_rounds: exact round count to produce (a program's ga); extra
    rounds are fully padded. None = ceil(total/t_round).
    """
    if on_overflow not in ("split", "error"):
        raise ValueError(f"on_overflow '{on_overflow}'")
    pairs = []
    for i, (tok, tgt) in enumerate(seqs):
        tok = np.asarray(tok, dtype=_INT).ravel()
        tgt = np.asarray(tgt, dtype=_INT).ravel()
        if tok.shape != tgt.shape:
            raise ValueError(f"seq {i}: tokens {tok.shape} != "
                             f"targets {tgt.shape}")
        if len(tok) == 0:
            continue
        pairs.append((i, tok, tgt))

    total = sum(len(t) for _, t, _ in pairs)
    need = -(-total // t_round) if total else 0
    k = need if n_rounds is None else n_rounds
    if k < need:
        raise ValueError(f"{total} tokens need {need} rounds of "
                         f"{t_round}; n_rounds={n_rounds}")
    if on_overflow == "error":
        for i, tok, _ in pairs:
            if len(tok) > t_round:
                raise ValueError(
                    f"seq {i} ({len(tok)} tokens) exceeds t_round "
                    f"{t_round} and on_overflow='error'")

    # deterministic LPT: length desc, tie by original index
    order = sorted(pairs, key=lambda p: (-len(p[1]), p[0]))
    loads = [0] * k
    placed: list[list[tuple[np.ndarray, np.ndarray]]] = \
        [[] for _ in range(k)]
    n_splits = 0
    for i, tok, tgt in order:
        fits = [r for r in range(k) if loads[r] + len(tok) <= t_round]
        if fits:
            r = min(fits, key=lambda r: (loads[r], r))
            placed[r].append((tok, tgt))
            loads[r] += len(tok)
            continue
        if on_overflow == "error":
            raise ValueError(
                f"seq {i} ({len(tok)} tokens) fits no round whole "
                f"and on_overflow='error'")
        # split: consume capacity from emptiest rounds first
        n_splits += 1
        off = 0
        for r in sorted(range(k), key=lambda r: (loads[r], r)):
            gap = t_round - loads[r]
            if gap <= 0 or off >= len(tok):
                continue
            take = min(gap, len(tok) - off)
            placed[r].append((tok[off:off + take], tgt[off:off + take]))
            loads[r] += take
            off += take
        if off < len(tok):
            raise AssertionError("split placement lost tokens")

    rounds = []
    for r in range(k):
        segs = placed[r]
        if len(segs) + 1 > s_max:
            raise ValueError(
                f"round {r}: {len(segs)} segments + pad tail exceeds "
                f"s_max {s_max}; raise s_max or t_round")
        toks = np.full(t_round, pad_token, dtype=_INT)
        tgts = np.full(t_round, IGNORE_INDEX, dtype=_INT)
        cu = np.full(s_max + 1, t_round, dtype=_INT)
        cu[0] = 0
        pos = 0
        ssq = 0
        for j, (tok, tgt) in enumerate(segs):
            toks[pos:pos + len(tok)] = tok
            tgts[pos:pos + len(tok)] = tgt
            pos += len(tok)
            cu[j + 1] = pos
            ssq += len(tok) * len(tok)
        # pad tail = its own segment (defined numerics everywhere;
        # exclusion is the deferred prefix-truncation optimization)
        if pos < t_round:
            cu[len(segs) + 1] = t_round
        rounds.append(PackedRound(
            tokens=toks, targets=tgts, cu=cu, n_segments=len(segs),
            valid_count=pos, sum_len_sq=ssq))

    fills = [r.valid_count / t_round for r in rounds] or [1.0]
    return PackedStep(
        rounds=tuple(rounds), t_round=t_round, total_tokens=total,
        pad_fraction=1.0 - (total / (k * t_round) if k else 1.0),
        min_fill=min(fills), n_splits=n_splits)
