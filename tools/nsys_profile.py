"""Run a training point under Nsight Systems with device metrics + NVTX.

Wraps the exact sweep pipeline (tools/m4_train.py) in `nsys profile` with:

- ``--trace=cuda,nvtx,osrt`` and ``--cuda-memory-usage=true``
- ``--gpu-metrics-devices=all`` (SM occupancy/throughput/PCIe counters on
  the timeline; needs profiling permission — see the hint printed on
  failure, or pass --no-gpu-metrics)
- NVTX ranges from the runtime's annotator abstraction
  (``DATAFLOW_NVTX=1``): one range per task (task ids like
  ``block_bwd_0_3_16``), one per transfer (``from_slow:A_0_3_16``), one per
  optimizer step (``step:N``) — nsys's projection view attributes them onto
  the streams that executed the work. The annotator is vendor-portable
  (device/annotate.py): an AMD roctx implementation plugs into the same
  three calls.
- by default, capture is limited to the training steps
  (``--capture-range=nvtx`` on the ``train_steps`` range), so planning /
  profiling / setup do not bloat the report.

Profiles and PCIe measurements come from the disk caches, so the traced
process spends its time in the part you care about.

Usage (the M4.4 headline point):
    python tools/nsys_profile.py                       # bs8/ga8 @ 24 GiB
    python tools/nsys_profile.py --config 8b-s1k-bs2ga32 --budget 16
    python tools/nsys_profile.py --capture full --stats
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_nsys_cmd(args, report: Path) -> list[str]:
    cmd = [
        "nsys", "profile",
        f"--output={report}",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "--cuda-memory-usage=true",
    ]
    if args.gpu_metrics:
        cmd += [
            "--gpu-metrics-devices=all",
            f"--gpu-metrics-frequency={args.gpu_metrics_frequency}",
        ]
    if args.capture == "steps":
        # start at the runtime's train_steps NVTX range; stop when it pops.
        # NSYS_NVTX_PROFILER_REGISTER_ONLY=0: we push plain (unregistered)
        # NVTX strings; without this, the trigger silently never matches and
        # nsys generates NO report.
        cmd += [
            "--capture-range=nvtx",
            "--nvtx-capture=train_steps",
            "--capture-range-end=stop",
            "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
        ]
    cmd += [
        sys.executable, "tools/m4_train.py",
        "--config", args.config,
        "--budgets", f"{args.budget:g}",
        "--steps", str(args.steps),
        "--recompute",
        "--backing-gib", f"{args.backing_gib:g}",
        "--out", str(args.out / "run-artifacts"),
    ]
    if args.annotated is not None:
        cmd += ["--annotated", str(args.annotated)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="nsys report for one training point (device metrics + NVTX)"
    )
    parser.add_argument("--config", default="8b-s1k-bs8ga8")
    parser.add_argument("--budget", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--backing-gib", type=float, default=100.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/nsys"))
    parser.add_argument("--capture", choices=["steps", "full"], default="steps",
                        help="steps: only the training loop (default); "
                             "full: the whole process incl. planning")
    parser.add_argument("--no-gpu-metrics", dest="gpu_metrics", action="store_false",
                        help="skip --gpu-metrics-devices (no perf-counter permission)")
    parser.add_argument("--gpu-metrics-frequency", type=int, default=10000)
    parser.add_argument("--stats", action="store_true",
                        help="run `nsys stats` summaries after profiling")
    parser.add_argument("--annotated", type=Path, default=None,
                        help="profile an exact SAVED plan (passed through to "
                             "m4_train --annotated; --config must match)")
    args = parser.parse_args()

    if shutil.which("nsys") is None:
        sys.exit("nsys not found on PATH (install NVIDIA Nsight Systems)")
    if not Path("tools/m4_train.py").exists():
        sys.exit("run from the repository root")

    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / f"{args.config}-{args.budget:g}gib-{args.steps}steps"
    cmd = build_nsys_cmd(args, report)
    env = {**os.environ, "DATAFLOW_NVTX": "1"}

    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        if args.gpu_metrics:
            print(
                "\nnsys failed. If the error mentions GPU performance counter "
                "permission (ERR_NVGPUCTRPERM), either rerun with "
                "--no-gpu-metrics or enable counters for all users:\n"
                "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' "
                "| sudo tee /etc/modprobe.d/nvidia-prof.conf && reboot",
                file=sys.stderr,
            )
        sys.exit(proc.returncode)

    rep_file = report.with_suffix(".nsys-rep")
    if not rep_file.exists():
        sys.exit(
            f"nsys exited 0 but {rep_file} was not generated — with "
            f"--capture steps this means the train_steps NVTX range never "
            f"triggered capture. Try --capture full to bisect."
        )
    print(f"\nreport: {rep_file}")
    if args.stats:
        subprocess.run([
            "nsys", "stats",
            "--report", "nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum",
            str(rep_file),
        ])


if __name__ == "__main__":
    main()
