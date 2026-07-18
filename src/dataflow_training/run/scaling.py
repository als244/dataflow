"""Scaling-study analysis: throughput table + a loss-vs-size fit over the
saved run curves.

The scaling experiments train the ladder (125M -> 1B) at the SAME recipe /
token budget as the 1B parity run; each produces a loss curve. Over a short
65M-token budget the models are undertrained, so the fit is illustrative
(the trend, not a converged scaling law): we report a smoothed final loss per
size and a log-linear fit L ≈ a - b·log10(N_non_embedding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .driver import RunResult, load_result


def _tail_mean(xs, frac: float = 0.1) -> float:
    """Mean of the last ``frac`` of a curve (a stable end-of-run loss)."""
    if not xs:
        return float("nan")
    k = max(1, int(len(xs) * frac))
    return sum(xs[-k:]) / k


@dataclass
class ThroughputRow:
    label: str
    backend: str
    budget_gib: float | None
    tokens_per_step: int
    steady_tok_per_s: float
    final_loss: float


def throughput_table(results: dict[str, RunResult]) -> list[ThroughputRow]:
    rows = []
    for label, r in results.items():
        tail = r.tok_per_s[1:] or r.tok_per_s
        rows.append(ThroughputRow(
            label=label, backend=r.backend, budget_gib=r.budget_gib,
            tokens_per_step=int(r.meta.get("tokens_per_step", 0)),
            steady_tok_per_s=(sum(tail) / len(tail) if tail else 0.0),
            final_loss=_tail_mean(r.losses),
        ))
    return rows


@dataclass
class ScalingFit:
    points: list[tuple[float, float, str]]   # (N_non_embed, final_loss, label)
    a: float                                 # L ≈ a - b·log10(N)
    b: float
    r2: float

    def predict(self, n: float) -> float:
        return self.a - self.b * math.log10(n)


def fit_scaling(results: dict[str, RunResult]) -> ScalingFit:
    """Log-linear fit of the smoothed final loss vs non-embedding params."""
    pts = []
    for label, r in results.items():
        params = r.meta.get("params")
        if not params:
            continue
        n = float(params["non_embedding"])
        pts.append((n, _tail_mean(r.losses), label))
    pts.sort()
    if len(pts) < 2:
        return ScalingFit(points=pts, a=float("nan"), b=float("nan"), r2=float("nan"))
    xs = [math.log10(n) for n, _, _ in pts]
    ys = [loss for _, loss, _ in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0        # dL/dlog10(N) (negative)
    a = my - slope * mx
    b = -slope
    ss_res = sum((y - (a - b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return ScalingFit(points=pts, a=a, b=b, r2=r2)


def load_all(results_dir, patterns=("*.json",)) -> dict[str, RunResult]:
    """Load saved RunResults keyed by filename stem."""
    from pathlib import Path

    out = {}
    d = Path(results_dir)
    for pat in patterns:
        for p in sorted(d.glob(pat)):
            try:
                out[p.stem] = load_result(p)
            except Exception:
                pass
    return out
