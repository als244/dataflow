"""Compare two loss curves (reference vs engine, or engine at two budgets)
and judge alignment.

The two backends use identical init + data + recipe but different execution
engines (pytorch eager vs the dataflow service), so curves are expected to
track within bf16 kernel-order noise, NOT bit-exactly. We report the step-0
gap (a pure-forward check on identical weights), the max/mean/final absolute
CE gaps, and an EMA of the gap (robust to per-step noise).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Band:
    step0_abs: float = 0.05     # forward on identical init should match tightly
    max_abs: float = 0.20       # worst per-step gap over the run
    final_abs: float = 0.10     # end-of-run gap
    ema_abs: float = 0.05       # smoothed gap (the trajectory-tracking metric)


@dataclass
class ParityReport:
    n: int
    step0_abs: float
    max_abs: float
    max_at: int
    mean_abs: float
    final_abs: float
    ema_abs: float
    a_final: float
    b_final: float
    a_label: str
    b_label: str
    passed: bool
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        ok = "ALIGNED" if self.passed else "DIVERGED"
        return (f"[{ok}] {self.a_label} vs {self.b_label} over {self.n} steps: "
                f"step0 Δ={self.step0_abs:.4f}  max Δ={self.max_abs:.4f}"
                f"@{self.max_at}  mean Δ={self.mean_abs:.4f}  "
                f"final Δ={self.final_abs:.4f} ({self.a_final:.4f} vs "
                f"{self.b_final:.4f})  ema Δ={self.ema_abs:.4f}"
                + ("" if self.passed else "  FAILS: " + "; ".join(self.failures)))


def compare(a_losses, b_losses, *, band: Band | None = None,
            a_label: str = "reference", b_label: str = "engine",
            ema_alpha: float = 0.2) -> ParityReport:
    import math

    band = band or Band()
    n = min(len(a_losses), len(b_losses))
    if n == 0:
        raise ValueError("empty curves")
    bad_a = sum(0 if math.isfinite(float(x)) else 1 for x in a_losses[:n])
    bad_b = sum(0 if math.isfinite(float(x)) else 1 for x in b_losses[:n])
    diffs = [abs(float(a_losses[i]) - float(b_losses[i])) for i in range(n)]
    ema = diffs[0]
    for d in diffs[1:]:
        ema = ema_alpha * d + (1 - ema_alpha) * ema
    max_abs = max(diffs)
    rep = ParityReport(
        n=n, step0_abs=diffs[0], max_abs=max_abs, max_at=diffs.index(max_abs),
        mean_abs=sum(diffs) / n, final_abs=diffs[-1], ema_abs=ema,
        a_final=float(a_losses[n - 1]), b_final=float(b_losses[n - 1]),
        a_label=a_label, b_label=b_label, passed=True,
    )
    fails = []
    if bad_a or bad_b:
        # NaN comparisons are all False — without this gate a non-finite
        # curve sails through every band check
        fails.append(f"non-finite losses ({a_label}:{bad_a} {b_label}:{bad_b})")
    if rep.step0_abs > band.step0_abs:
        fails.append(f"step0 {rep.step0_abs:.4f}>{band.step0_abs}")
    if rep.max_abs > band.max_abs:
        fails.append(f"max {rep.max_abs:.4f}>{band.max_abs}")
    if rep.final_abs > band.final_abs:
        fails.append(f"final {rep.final_abs:.4f}>{band.final_abs}")
    if rep.ema_abs > band.ema_abs:
        fails.append(f"ema {rep.ema_abs:.4f}>{band.ema_abs}")
    rep.passed = not fails
    rep.failures = fails
    return rep


def curves_healthy(losses, *, expect_start: float, start_tol: float = 0.5,
                   min_drop: float = 0.2) -> tuple[bool, str]:
    """Sanity: the curve starts near ln(V) and makes real downward progress
    (no NaN/blowup). ``expect_start`` is typically ln(vocab)."""
    import math

    if any(not math.isfinite(x) for x in losses):
        return False, "non-finite loss (NaN/inf)"
    start = losses[0]
    if abs(start - expect_start) > start_tol:
        return False, f"start {start:.3f} far from ln(V)={expect_start:.3f}"
    drop = losses[0] - min(losses)
    if drop < min_drop:
        return False, f"insufficient drop {drop:.3f} < {min_drop}"
    return True, f"start {start:.3f}, min {min(losses):.3f}, drop {drop:.3f}"
