"""Correctness at very tight budgets: runtime vs plain-torch golden model.

Runs the full annotated program through the REAL engine (check_model_step)
on a plain-torch-trainable config at descending fast-memory budgets, and
records the per-tensor relative-L2 error of loss + every final weight
against the golden eager-torch trajectory. The point: memory pressure
(offload/prefetch/recompute churn) must not perturb the math at all —
errors stay at bf16 noise regardless of budget.

Usage: python tools/verify/pressure_correctness.py [--out artifacts/correctness.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataflow_training.model_families.llama3 import ShapedLlamaConfig
from dataflow_training.testing.gradcheck import check_model_step

GIB = 1024**3

CFG = ShapedLlamaConfig(
    n_layers=16, d_model=2048, n_heads=16, n_kv_heads=4, d_ff=8192,
    vocab_size=32768, seq_len=4096, batch=1,
)

# (budget_gib, recompute) — descending pressure; the tightest points force
# recompute of every block context
POINTS = [
    (6.0, False),
    (4.0, False),
    (3.0, False),
    (2.5, True),
    (2.0, True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/correctness.json"))
    args = parser.parse_args()

    from dataflow_training.model_families.llama3 import lower_llama3

    levels_all = {
        rw.object_id: 1 for rw in lower_llama3(CFG).recompute_rewrites
    }
    import torch

    rows = []
    for gib, recompute in POINTS:
        levels = levels_all if recompute else None
        # each point leaves the golden model's autograd allocations in the
        # torch cache; release them or they crowd out the next engine slab
        torch.cuda.empty_cache()
        try:
            report = check_model_step(
                CFG, fast_memory_capacity=int(gib * GIB),
                recompute_levels=levels, seed=7,
            )
        except Exception as exc:
            status = f"{type(exc).__name__}: {exc}"
            rows.append({"budget_gib": gib, "recompute": recompute,
                         "status": status})
            print(f"budget {gib:g} GiB (recompute={recompute}): {status}")
            continue
        worst_name, worst_err = report.worst()
        rows.append({
            "budget_gib": gib,
            "recompute": recompute,
            "status": "ok" if report.ok else "FAILED",
            "worst_tensor": worst_name,
            "worst_rel_l2": worst_err,
            "errors": report.errors,
            "tol": report.tol,
        })
        print(f"budget {gib:g} GiB (recompute={recompute}): "
              f"{'ok' if report.ok else 'FAILED'} — worst {worst_name} "
              f"rel-L2 {worst_err:.2e} (tol {report.tol:g})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "n_layers": CFG.n_layers, "d_model": CFG.d_model,
            "n_heads": CFG.n_heads, "n_kv_heads": CFG.n_kv_heads,
            "d_ff": CFG.d_ff, "vocab_size": CFG.vocab_size,
            "seq_len": CFG.seq_len, "batch": CFG.batch,
        },
        "points": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
