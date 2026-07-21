#!/usr/bin/env python
"""nsys capture of a solo ENGINE run — the profiling rig's single-box
entry point.

Wraps ``tools/train/train_solo.py engine`` in ``nsys profile`` with the
fleet's canonical trace set and cudaProfilerApi capture range: the run
brackets the requested step window via the daemon's ``profiler_control``
verb (annotator start/stop -> cudaProfilerStart/Stop), so the report
holds exactly those steps — warmed, no boot noise.

    python tools/bench/nsys_profile.py --preset gpt2_124m --ga-rounds 8 \
        --batch 64 --data doc --budget 16 --slab 16 \
        --steps 10 --start 5 --stop 8 --out gpt2_124m_ga8

writes results/pretrain/logs/<out>.nsys-rep. Extra train_solo engine
flags append verbatim as trailing arguments.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

# the fleet's canonical trace set (distributed/hostops.py NSYS_TRACE):
# nsys 2025.5 rejects a 'nccl' trace value — NCCL activity arrives via
# cuda kernels + its NVTX ranges
NSYS_TRACE = "cuda,nvtx,osrt,cublas,cudnn"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="l3_1b")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--start", type=int, default=5,
                    help="capture starts BEFORE this step")
    ap.add_argument("--stop", type=int, default=8,
                    help="capture stops AFTER this step")
    ap.add_argument("--ga-rounds", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--data", default=None,
                    help="data source spec (docs/data_feeds.md); default: "
                         "train_solo's default feed. Plan-comparable "
                         "captures want the uniform-window config: "
                         "--data 'shards:,window=SEQ' --packing-policy "
                         "greedy (via the passthrough args)")
    ap.add_argument("--opt", choices=["adamw", "muon"], default=None)
    ap.add_argument("--budget", type=float, default=14.0)
    ap.add_argument("--slab", type=float, default=16.0)
    ap.add_argument("--out", default=None,
                    help="report stem under results/pretrain/logs/ "
                         "(default: <preset>)")
    ap.add_argument("--nsys", default="nsys", help="nsys binary")
    ap.add_argument("extra", nargs="*",
                    help="extra train_solo engine flags, passed through")
    args = ap.parse_args()

    if shutil.which(args.nsys) is None:
        print(f"error: {args.nsys!r} not on PATH", file=sys.stderr)
        return 2
    if not 0 <= args.start <= args.stop < args.steps:
        print(f"error: need 0 <= start <= stop < steps "
              f"(got {args.start}/{args.stop}/{args.steps})",
              file=sys.stderr)
        return 2

    out_dir = _ROOT / "results" / "pretrain" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (args.out or args.preset)
    run_json = out_dir / f"{out.name}_profile_run.json"

    train = [sys.executable, "-u", str(_ROOT / "tools" / "train" / "train_solo.py"),
             "engine", "--preset", args.preset,
             "--steps", str(args.steps),
             "--budget", str(args.budget), "--slab", str(args.slab),
             "--profile",
             "--profile-start-before-step", str(args.start),
             "--profile-stop-after-step", str(args.stop),
             "--out", str(run_json)]
    if args.ga_rounds:
        train += ["--ga-rounds", str(args.ga_rounds)]
    if args.batch:
        train += ["--batch", str(args.batch)]
    if args.data:
        train += ["--data", args.data]
    if args.opt:
        train += ["--opt", args.opt]
    train += list(args.extra)

    cmd = [args.nsys, "profile", f"--trace={NSYS_TRACE}",
           "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
           "--gpu-metrics-devices=0",
           "-o", str(out), "--force-overwrite", "true"] + train
    print("[nsys_profile]", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=_ROOT).returncode
    rep = out.with_suffix(".nsys-rep")
    if rc == 0 and rep.exists():
        print(f"[nsys_profile] report -> {rep}")
    else:
        print(f"[nsys_profile] FAILED (rc {rc}); report "
              f"{'present' if rep.exists() else 'missing'}",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
