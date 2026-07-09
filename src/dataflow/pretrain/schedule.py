"""Learning-rate schedule: linear warmup then cosine decay.

The engine's optimizer scales its base LR by ``LRSchedule.scale(step)``
(``dataflow.tasks.optim``), a pure function of the 1-indexed optimizer
step, and ``AdamWStep`` applies it as ``hyper.lr * scale(run_args["step"]
+ 1)``. For byte-faithful parity the reference backend must compute the
IDENTICAL lr(step), so this module DELEGATES to the same ``LRSchedule``
rather than reimplementing the cosine (a second copy would drift — the
cosine progress is measured from the 1-indexed step, an easy off-by-one).

``CosineSchedule(step)`` returns the absolute LR for the 0-indexed driver
step ``step`` (the value both backends use for that optimizer step).
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow.tasks.optim import LRSchedule


@dataclass(frozen=True)
class CosineSchedule:
    peak_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.total_steps <= 0:
            raise ValueError("warmup_steps >= 0 and total_steps > 0 required")
        if self.min_lr < 0 or self.peak_lr < self.min_lr:
            raise ValueError("require 0 <= min_lr <= peak_lr")

    @property
    def min_lr_frac(self) -> float:
        return self.min_lr / self.peak_lr if self.peak_lr else 0.0

    def lrschedule(self) -> LRSchedule:
        """The engine-side schedule object (the authority). ``peak_lr`` is
        carried separately as the base LR; this scales it in [frac, 1]."""
        return LRSchedule(
            kind="cosine",
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_frac=self.min_lr_frac,
        )

    def __call__(self, step: int) -> float:
        """Absolute LR for the 0-indexed optimizer ``step``. Mirrors the
        engine exactly: ``AdamWStep`` reads ``run_args['step']`` (=step) and
        applies ``hyper.lr * schedule.scale(step + 1)``."""
        return self.peak_lr * self.lrschedule().scale(step + 1)
