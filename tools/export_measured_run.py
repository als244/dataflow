"""Export a real run as a webapp-uploadable measured-run file.

Consumes the output of tools/gap_analysis.py (trace.json + annotated.json)
and produces one self-contained JSON:

    {
      "format": "dataflow-measured-run/v1",
      "meta":        {...run identity: config, budget, kernel set, device},
      "log":         EventLog of MEASURED intervals + memory trace,
      "summary":     Summary computed from the real run,
      "sim_log":     EventLog the simulator predicts for the same plan,
      "sim_summary": Summary of that prediction
    }

Both logs use the webapp's native EventLog shape (task_intervals,
memory_trace, peak_fast_memory_bytes), so the UI renders a real run with
the exact panels used for simulations, and can diff the two without any
server round-trip.

Usage:
    python tools/export_measured_run.py --gap-dir artifacts/m4/gap-bs4ga4-18 \
        --meta config=8b-bs4ga4 budget_gib=18 --out bs4ga4-18.measured.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataflow.core import load_program
from dataflow.core.convert import to_sim_chain


def _summary(intervals: list[dict], memory_peak_bytes: int, flops: float,
             tokens: float, hardware_tflops: float = 0.0) -> dict:
    makespan = max((iv["end"] for iv in intervals), default=0.0)
    compute = [iv for iv in intervals if iv["track"] == "compute"]
    busy = sum(iv["end"] - iv["start"] for iv in compute)
    recompute = sum(
        iv["end"] - iv["start"] for iv in compute if iv["task_id"].startswith("r_")
    )
    util = {
        t: sum(iv["end"] - iv["start"] for iv in intervals if iv["track"] == t)
        for t in ("from_slow", "to_slow")
    }
    return {
        "makespan_us": makespan,
        "total_flops": flops,
        "total_effective_flops": flops,
        "tokens_per_second": tokens / (makespan / 1e6) if makespan else 0.0,
        "primary_unit": "tokens",
        "primary_count": tokens,
        "primary_rate_per_second": tokens / (makespan / 1e6) if makespan else 0.0,
        "effective_tflops": (flops / makespan / 1e6) if makespan else 0.0,
        "hardware_tflops": hardware_tflops,
        "peak_fast_memory_gb": memory_peak_bytes / 1e9,
        "idle_pct": (1 - busy / makespan) * 100 if makespan else 0.0,
        "recompute_pct": (recompute / busy) * 100 if busy else 0.0,
        "from_slow_util_pct": util["from_slow"] / makespan * 100 if makespan else 0.0,
        "to_slow_util_pct": util["to_slow"] / makespan * 100 if makespan else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-dir", type=Path, required=True,
                        help="output dir of tools/gap_analysis.py")
    parser.add_argument("--meta", nargs="*", default=[],
                        help="key=value pairs merged into meta")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads((args.gap_dir / "trace.json").read_text())
    program = load_program(args.gap_dir / "annotated.json")
    analysis = json.loads((args.gap_dir / "analysis.json").read_text())

    total_flops = float(sum(getattr(t, "flops", 0) or 0 for t in program.tasks))
    meta: dict = {
        "name": program.name,
        "kernel_set": "fused-v1 (triton)",
        "device": "RTX 5090",
        "real_tokens_per_s": analysis.get("real_tokens_per_s"),
        "sim_tokens_per_s": analysis.get("sim_tokens_per_s"),
        "replay_gap_pct": analysis.get("replay_gap_pct"),
    }
    for kv in args.meta:
        k, _, v = kv.partition("=")
        try:
            meta[k] = float(v) if "." in v or v.isdigit() else v
        except ValueError:
            meta[k] = v
    tokens = float(meta.get("tokens_per_step", 0)) or (
        analysis["real_tokens_per_s"] * trace["makespan_us"] / 1e6
    )

    # measured log: intervals verbatim; memory trace as one band (role-split
    # reconstruction is a follow-up — the capacity/pressure picture is intact)
    measured_log = {
        "task_intervals": trace["intervals"],
        "events": [],
        "peak_fast_memory_bytes": trace["peak_fast_bytes"],
        "memory_trace": [
            {"t": t, "fast_bytes_by_band": {"other": used}}
            for t, used in trace["memory_trace"]
        ],
    }

    # sim prediction for the same annotated plan
    from dataflow_sim.engine.simulator import run as sim_run

    log = sim_run(to_sim_chain(program), snapshots=False, memory_trace=True)
    sim_intervals = [
        {"task_id": iv.task_id, "start": iv.start, "end": iv.end, "track": iv.track}
        for iv in log.task_intervals
    ]
    sim_log = {
        "task_intervals": sim_intervals,
        "events": [],
        "peak_fast_memory_bytes": log.peak_fast_memory_bytes,
        "memory_trace": [
            {"t": p.t, "fast_bytes_by_band": dict(p.fast_bytes_by_band)}
            for p in log.memory_trace
        ],
    }

    payload = {
        "format": "dataflow-measured-run/v1",
        "meta": meta,
        "log": measured_log,
        "summary": _summary(trace["intervals"], trace["peak_fast_bytes"],
                            total_flops, tokens),
        "sim_log": sim_log,
        "sim_summary": _summary(sim_intervals, log.peak_fast_memory_bytes,
                                total_flops, tokens),
    }
    args.out.write_text(json.dumps(payload) + "\n")
    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({mb:.1f} MB): "
          f"{len(trace['intervals'])} measured intervals, "
          f"{len(sim_intervals)} sim intervals")


if __name__ == "__main__":
    main()
