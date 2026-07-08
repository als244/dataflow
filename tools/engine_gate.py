"""Engine gate: real-GPU synthetic execution vs simulator prediction.

Flow: measure real PCIe bandwidths -> build + plan a shaped program whose
transfer model uses the measured numbers -> simulate (prediction) -> execute
on the CUDA backend with calibrated spin kernels (planned runtimes become
physically true) -> compare makespan/intervals, quantify host-side overhead
(GPU idle gaps between consecutive compute tasks), and measure overlap.

Usage:
    python tools/engine_gate.py --config small --fast-gib 2
    python tools/engine_gate.py --config 8b --fast-gib 16 [--recompute]
    python tools/engine_gate.py ... --completion-mode hostfn   # compare token modes
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

from dataflow.core.convert import to_sim_chain
from dataflow.runtime import Engine
from dataflow.runtime.device.cuda import CudaBackend, _check
from dataflow.runtime.device.cuda_spin import make_spin_resolver
from dataflow.training.planning import plan_program
from dataflow.training.llama3 import ShapedLlamaConfig, build_shaped_llama3
from dataflow.training.shaped_program import ShapedHardware

GIB = 1024**3


def build_config(name: str) -> ShapedLlamaConfig:
    if name == "8b":
        return ShapedLlamaConfig.llama3_8b()
    if name == "small":
        return ShapedLlamaConfig(
            n_layers=8, d_model=2048, n_heads=16, n_kv_heads=4, d_ff=8192,
            vocab_size=32768, seq_len=2048, batch=1,
        )
    if name == "mini":
        return ShapedLlamaConfig(
            n_layers=4, d_model=1024, n_heads=8, n_kv_heads=2, d_ff=4096,
            vocab_size=16384, seq_len=1024, batch=1,
        )
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["mini", "small", "8b"], default="small")
    parser.add_argument("--fast-gib", type=float, required=True)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--completion-mode", choices=["poll", "hostfn"], default="poll")
    parser.add_argument(
        "--bw-mode", choices=["bidi", "uni"], default="bidi",
        help="Plan transfers with bidirectional (concurrent-load) or unidirectional "
             "measured bandwidth. bidi is the honest default: under memory pressure "
             "both directions run concurrently most of the time.",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/m2"))
    args = parser.parse_args()

    backend = CudaBackend(completion_mode=args.completion_mode)
    pcie = backend.measure_pcie()
    print(f"measured pinned PCIe GB/s: uni h2d={pcie.uni_h2d / 1e3:.1f} d2h={pcie.uni_d2h / 1e3:.1f}"
          f" | bidi h2d={pcie.bidi_h2d / 1e3:.1f} d2h={pcie.bidi_d2h / 1e3:.1f}")
    if args.bw_mode == "bidi":
        h2d_bpus, d2h_bpus = pcie.bidi_h2d, pcie.bidi_d2h
    else:
        h2d_bpus, d2h_bpus = pcie.uni_h2d, pcie.uni_d2h

    cfg = build_config(args.config)
    hw = ShapedHardware()  # compute-cost knobs are irrelevant: spin makes runtimes true

    def build(levels=None):
        program = build_shaped_llama3(cfg, hw=hw, recompute_levels=levels)
        return replace(program, bandwidth_from_slow=h2d_bpus, bandwidth_to_slow=d2h_bpus)

    cap = int(args.fast_gib * GIB)
    t0 = time.perf_counter()
    planned = plan_program(
        build(), fast_memory_capacity=cap, recompute=args.recompute,
        build_variant=(lambda levels: build(levels)) if args.recompute else None,
    )
    program = planned.program
    print(f"planned in {time.perf_counter() - t0:.2f}s: predicted makespan "
          f"{planned.makespan_us / 1e3:.2f}ms, peak {planned.peak_fast_bytes / GIB:.2f}GiB, "
          f"recompute {sum(1 for v in planned.recompute_levels.values() if v)}/"
          f"{len(planned.recompute_levels)}")

    resolver = make_spin_resolver(backend)
    print(f"spin accuracy (wall-true globaltimer): ratio={resolver.spin_accuracy_ratio:.4f}")

    # fake dry run -> exact buffer demand -> prewarm (no vendor allocs mid-run)
    from dataflow.runtime.device.fake import FakeBackend

    dry = Engine(FakeBackend()).execute(program)
    t0 = time.perf_counter()
    result = Engine(backend).execute(program, resolver=resolver, pool_prewarm=dry.pool_demand)
    wall_s = time.perf_counter() - t0
    real_makespan = result.makespan_us

    # --- analysis -------------------------------------------------------------
    sim_makespan = planned.makespan_us
    gap_pct = (real_makespan - sim_makespan) / sim_makespan * 100

    # Scheduler-fidelity replay: re-simulate with every MEASURED duration as
    # an override (tasks and transfers alike). If the runtime schedules like
    # the simulator, makespans match up to dispatch overhead — bandwidth-model
    # error is factored out entirely.
    from dataflow.training.replay import replay_gap_pct as _rg

    replay_gap_pct = _rg(program, result.trace, result.makespan_us)

    compute = sorted((iv for iv in result.trace.intervals if iv.track == "compute"),
                     key=lambda iv: iv.start)
    idle_gaps = [b.start - a.end for a, b in zip(compute, compute[1:])]
    planned_by_id = {t.id: t.runtime_us for t in program.tasks}
    spin_err_pct = [
        (iv.end - iv.start - planned_by_id[iv.task_id]) / planned_by_id[iv.task_id] * 100
        for iv in compute
    ]
    transfers = [iv for iv in result.trace.intervals if iv.track != "compute"]
    overlap_us = 0.0
    for tr in transfers:
        for c in compute:
            lo, hi = max(tr.start, c.start), min(tr.end, c.end)
            if hi > lo:
                overlap_us += hi - lo
    transfer_us = sum(iv.end - iv.start for iv in transfers)

    summary = {
        "config": args.config,
        "completion_mode": args.completion_mode,
        "bw_mode": args.bw_mode,
        "fast_gib": args.fast_gib,
        "recompute_chosen": sum(1 for v in planned.recompute_levels.values() if v),
        "task_count": len(program.tasks),
        "measured_pcie_gbs": {
            "uni_h2d": pcie.uni_h2d / 1e3, "uni_d2h": pcie.uni_d2h / 1e3,
            "bidi_h2d": pcie.bidi_h2d / 1e3, "bidi_d2h": pcie.bidi_d2h / 1e3,
        },
        "planned_with": {"h2d_gbs": h2d_bpus / 1e3, "d2h_gbs": d2h_bpus / 1e3},
        "sim_makespan_ms": sim_makespan / 1e3,
        "real_makespan_ms": real_makespan / 1e3,
        "gap_pct_vs_plan": gap_pct,
        "gap_pct_vs_replay": replay_gap_pct,
        "wall_s": wall_s,
        "peak_fast_gib_sim": planned.peak_fast_bytes / GIB,
        "peak_fast_gib_real": result.peak_fast_bytes / GIB,
        "gpu_idle_between_tasks_us": {
            "p50": statistics.median(idle_gaps) if idle_gaps else 0.0,
            "p95": statistics.quantiles(idle_gaps, n=20)[-1] if len(idle_gaps) >= 20 else max(idle_gaps, default=0.0),
            "max": max(idle_gaps, default=0.0),
            "total_ms": sum(idle_gaps) / 1e3,
        },
        "spin_duration_error_pct": {
            "p50": statistics.median(spin_err_pct),
            "p95": statistics.quantiles(spin_err_pct, n=20)[-1] if len(spin_err_pct) >= 20 else max(spin_err_pct),
        },
        "transfer_time_ms": transfer_us / 1e3,
        "transfer_overlap_with_compute_pct": (overlap_us / transfer_us * 100) if transfer_us else 0.0,
        "buffers_allocated": result.buffers_allocated,
        "slab_overflows": result.slab_overflows,
        "buffers_reused": result.buffers_reused,
        "events_created": backend.events_created,
    }
    print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{program.name}-{args.fast_gib:g}gib-{args.completion_mode}"
    (args.out / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / f"{stem}.trace.json").write_text(json.dumps(
        {
            "intervals": [iv.__dict__ for iv in result.trace.intervals],
            "sim_intervals": [
                {"task_id": iv.task_id, "start": iv.start, "end": iv.end, "track": iv.track}
                for iv in __import__("dataflow_sim.engine.simulator", fromlist=["run"]).run(
                    to_sim_chain(program), snapshots=False
                ).task_intervals
            ],
        }, indent=2) + "\n")
    print(f"wrote {args.out}/{stem}.{{summary,trace}}.json")


if __name__ == "__main__":
    main()
