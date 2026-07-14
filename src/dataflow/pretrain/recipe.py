"""The training recipe — the ONE optimizer + schedule object both the
reference and engine backends read, so the only variable between them is
the execution engine.

Locked (2026-07-09): AdamW β(0.9, 0.95), weight decay 0.1 applied
UNIFORMLY (the engine's ``adamw`` policy decays every field including norm
gains; the reference mirrors that for exact parity — not the textbook
norm-exclusion), NO gradient clipping, cosine LR with linear warmup. Peak
LR 3e-4 over 1000 steps, warmup 100, decaying to 0.1×. Precision is
all-bf16 (param/grad/opt) — the engine's default ``DTypePolicy`` — so no
dtype override is needed on the config; the reference stores bf16 params
and bf16 AdamW moments to match.
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow.tasks.base_blocks import AdamWHyper

from .schedule import CosineSchedule


@dataclass(frozen=True)
class Recipe:
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    total_steps: int = 1000
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    momentum: float = 0.95          # muon/sgdm momentum (muon runs)
    muon_lr: float | None = None    # None: muon shares the scheduled lr
                                    # (the Moonshot scale's design)
    grad_clip: float | None = None  # v1: no clipping, both sides

    def schedule(self) -> CosineSchedule:
        return CosineSchedule(
            peak_lr=self.peak_lr, min_lr=self.min_lr,
            warmup_steps=self.warmup_steps, total_steps=self.total_steps,
        )

    def lr_at(self, step: int) -> float:
        """Absolute LR for the 0-indexed optimizer ``step`` (both backends)."""
        return self.schedule()(step)

    def base_hyper(self) -> AdamWHyper:
        """The engine resolver's hyper: base LR = peak, with the cosine
        ``LRSchedule`` baked in (``AdamWStep`` scales by
        ``schedule.scale(step+1)`` per run). Also the spec the reference
        optimizer reads (betas/eps/wd)."""
        return AdamWHyper(
            lr=self.peak_lr, beta1=self.beta1, beta2=self.beta2,
            eps=self.eps, weight_decay=self.weight_decay,
            momentum=self.momentum, muon_lr=self.muon_lr,
            schedule=self.schedule().lrschedule(),
        )

    def hyper_spec(self) -> dict:
        """JSON-able hyper for the wire resolver spec
        (``register_program(resolver={..., "hyper": recipe.hyper_spec()})``);
        ``bridge.resolver_for`` rebuilds an ``AdamWHyper`` + ``LRSchedule``
        from it. Kept in lock-step with ``base_hyper``."""
        return {
            "lr": self.peak_lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "muon_lr": self.muon_lr,
            "schedule": {
                "kind": "cosine",
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
                "min_lr_frac": self.min_lr / self.peak_lr if self.peak_lr else 0.0,
            },
        }


DEFAULT_RECIPE = Recipe()
