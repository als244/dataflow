#!/usr/bin/env python
"""Peek at an in-flight (or finished) long run's loss curve.

Solo-engine checkpoints embed the complete per-step loss curve so far
in every manifest (client_meta.losses — the same record resume
stitches from). This reads the newest complete checkpoint of a run
and writes a plottable partial-curve json + prints a summary.

    python tools/curve_peek.py l3_1b_engine_t512k_adamw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPTS = ROOT / "results" / "pretrain" / "checkpoints"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name (the --out stem; a directory "
                                "under results/pretrain/checkpoints/)")
    ap.add_argument("--ema", type=float, default=0.98)
    args = ap.parse_args()

    run_dir = CKPTS / args.run
    manifests = sorted(run_dir.glob("step_*/manifest.json"))
    if not manifests:
        print(f"no complete checkpoints under {run_dir}", file=sys.stderr)
        return 1
    manifest = json.loads(manifests[-1].read_text())
    meta = manifest.get("client_meta", {})
    losses = [float(x) for x in meta.get("losses", [])]
    if not losses:
        print(f"{manifests[-1]} carries no loss curve", file=sys.stderr)
        return 1

    ema_v = losses[0]
    for x in losses:
        ema_v = args.ema * ema_v + (1 - args.ema) * x
    out = ROOT / "results" / "pretrain" / f"{args.run}_partial.json"
    out.write_text(json.dumps({
        "backend": "engine", "partial_through_step": int(meta["step"]),
        "losses": losses, "meta": {"source": str(manifests[-1])},
    }, indent=2))
    print(f"{args.run}: {len(losses)} steps recorded "
          f"(through step {meta['step']})")
    print(f"  last loss {losses[-1]:.4f}   EMA({args.ema}) {ema_v:.4f}   "
          f"min {min(losses):.4f}")
    print(f"  partial curve -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
