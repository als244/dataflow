"""Decompose real-vs-sim throughput gap for one (config, budget) point.

Runs the exact sweep pipeline (profile -> plan on measured costs -> train a
few steps), then attributes the gap:

  real vs sim  =  scheduling fidelity (replay gap: re-sim with measured
                  durations)  +  cost-model error (planned vs measured
                  per-task durations, aggregated by task family)
                  +  transfer-model error (planned vs achieved bandwidth)

Outputs a markdown report + JSON (per-family planned/measured totals,
per-direction achieved bandwidth, stall exposure) and saves the raw trace
for the webapp exporter.

Usage:
    python tools/gap_analysis.py --config 8b-bs4ga4 --budget 18 --steps 3 \
        --out artifacts/m4/gap-bs4ga4-18
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from dataflow.runtime.device.cuda import CudaBackend
from dataflow.training.families import resolve_family
from dataflow.training.planning import plan_program
from dataflow.training.profiling import apply_measured_costs, cached_pcie, load_or_profile
from dataflow.training.replay import replay_gap_pct
from dataflow.training.train_loop import train

from m4_train import CONFIGS  # noqa: E402  (tools/ sibling import)

GIB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), default="8b-bs4ga4")
    parser.add_argument("--budget", type=float, default=18.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--recompute", action="store_true", default=True)
    parser.add_argument("--backing-gib", type=float, default=100.0)
    parser.add_argument("--contend", action="store_true",
                        help="profile under saturated bidirectional PCIe traffic "
                             "(bounds the cost model from the pessimistic side)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/m4/gap"))
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    fam = resolve_family(cfg)  # family-generic since the MoE families (was llama-hardcoded)
    dims = fam.dims_of(cfg)
    build_resolver = fam.build_resolver
    tokens_per_step = float(cfg.tokens * cfg.grad_accum_rounds)
    backend = CudaBackend()
    pcie = cached_pcie(backend)

    def build_raw(levels=None):
        return replace(
            fam.lower(cfg, recompute_levels=levels),
            bandwidth_from_slow=pcie.bidi_h2d,
            bandwidth_to_slow=pcie.bidi_d2h,
            backing_memory_capacity=int(args.backing_gib * GIB),
        )

    program = build_raw()
    profiles = load_or_profile(
        program, build_resolver(dims), backend, contend_pcie=args.contend,
    )
    rc_all = {rw.object_id: 1 for rw in program.recompute_rewrites}
    profiles.update(load_or_profile(
        build_raw(rc_all), build_resolver(dims), backend, contend_pcie=args.contend,
    ))
    measured = apply_measured_costs(program, profiles)

    planned = plan_program(
        measured, fast_memory_capacity=int(args.budget * GIB), recompute=True,
        build_variant=lambda levels: apply_measured_costs(build_raw(levels), profiles),
    )
    sim_ms = planned.makespan_us / 1e3
    sim_tok = tokens_per_step / (planned.makespan_us / 1e6)

    torch.cuda.empty_cache()
    report = train(planned.program, cfg, backend, steps=args.steps, seed=11)
    real_us = report.steady_state_makespan_us
    real_tok = tokens_per_step / (real_us / 1e6)
    trace = report.last_trace

    # persist raw artifacts first: analysis bugs must not cost the GPU run
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "trace.json").write_text(json.dumps({
        "intervals": [
            {"task_id": iv.task_id, "start": iv.start, "end": iv.end, "track": iv.track}
            for iv in trace.intervals
        ],
        "memory_trace": trace.memory_trace,
        "peak_fast_bytes": trace.peak_fast_bytes,
        "makespan_us": trace.makespan_us(),
    }) + "\n")
    from dataflow.core import save_program
    save_program(planned.program, args.out / "annotated.json")

    # non-fatal like m4_train (0336654): imposing real timings on the sim's
    # reserve-at-start accounting can be infeasible for plans the real
    # engine ran fine — the diagnostic must not cost the analysis
    try:
        replay_gap = replay_gap_pct(planned.program, trace, report.step_makespan_us[-1])
    except Exception as exc:  # noqa: BLE001
        print(f"replay-fidelity diagnostic infeasible (non-fatal): {exc}")
        replay_gap = None

    # --- per-task-family cost error --------------------------------------------
    tasks_by_id = {t.id: t for t in planned.program.tasks}
    fam_planned = defaultdict(float)
    fam_measured = defaultdict(float)
    fam_count = defaultdict(int)
    for iv in trace.intervals:
        if iv.track != "compute":
            continue
        spec = tasks_by_id.get(iv.task_id)
        if spec is None:
            continue
        fam = spec.compute_block_key or "unknown"
        fam_planned[fam] += float(spec.runtime_us)
        fam_measured[fam] += iv.end - iv.start
        fam_count[fam] += 1

    # --- transfer bandwidth achieved vs planned ---------------------------------
    sizes = {o.id: o.size_bytes for o in planned.program.initial_objects}
    for t in planned.program.tasks:
        for out in t.outputs:
            sizes[out.id] = out.size_bytes
    xfer = {"from_slow": {"bytes": 0, "us": 0.0, "n": 0},
            "to_slow": {"bytes": 0, "us": 0.0, "n": 0}}
    for iv in trace.intervals:
        if iv.track not in xfer:
            continue
        oid = iv.task_id.split(":", 1)[1].split("#", 1)[0]
        size = sizes.get(oid, 0)
        xfer[iv.track]["bytes"] += size
        xfer[iv.track]["us"] += iv.end - iv.start
        xfer[iv.track]["n"] += 1

    # --- compute-lane exposure ---------------------------------------------------
    compute_ivs = sorted(
        (iv for iv in trace.intervals if iv.track == "compute"), key=lambda i: i.start
    )
    busy = sum(iv.end - iv.start for iv in compute_ivs)
    gaps = []
    for prev, nxt in zip(compute_ivs, compute_ivs[1:]):
        if nxt.start > prev.end:
            gaps.append((prev.end, nxt.start - prev.end, prev.task_id, nxt.task_id))
    idle = sum(g[1] for g in gaps)
    gaps.sort(key=lambda g: -g[1])

    # --- report -------------------------------------------------------------------
    total_planned = sum(fam_planned.values())
    total_measured = sum(fam_measured.values())
    lines = []
    lines.append(f"# Gap analysis: {args.config} @ {args.budget:g} GiB "
                 f"({args.steps} steps, kernel set "
                 f"{sorted(set(build_resolver(dims).kernel_set.describe().values()))})\n")
    lines.append(f"- sim: {sim_ms:.1f} ms/step ({sim_tok:.0f} tok/s); "
                 f"real steady: {real_us / 1e3:.1f} ms/step ({real_tok:.0f} tok/s); "
                 f"real vs sim {(real_tok / sim_tok - 1) * 100:+.2f}%")
    lines.append(
        "- replay-fidelity gap (scheduling, cost error removed): "
        + (f"{replay_gap:+.2f}%" if replay_gap is not None
           else "n/a (replay reserve-infeasible under real timings — itself a fragility signal)")
    )
    lines.append(f"- compute lane: busy {busy / 1e3:.1f} ms, exposed idle "
                 f"{idle / 1e3:.1f} ms ({idle / (busy + idle) * 100:.1f}%), "
                 f"{len(gaps)} gaps")
    lines.append(f"- total task time: planned {total_planned / 1e3:.1f} ms vs "
                 f"measured {total_measured / 1e3:.1f} ms "
                 f"({(total_measured / total_planned - 1) * 100:+.2f}%)\n")
    lines.append("## Per-family planned vs measured (last step)\n")
    lines.append("| family | n | planned ms | measured ms | error | share of step |")
    lines.append("|:--|--:|--:|--:|--:|--:|")
    for fam in sorted(fam_measured, key=lambda f: -(fam_measured[f] - fam_planned[f])):
        p, m = fam_planned[fam], fam_measured[fam]
        lines.append(f"| {fam} | {fam_count[fam]} | {p / 1e3:.1f} | {m / 1e3:.1f} | "
                     f"{(m / p - 1) * 100:+.1f}% | {m / real_us * 100:.1f}% |")
    lines.append("\n## Transfers: planned vs achieved bandwidth\n")
    lines.append("| direction | n | GB moved | planned GB/s | achieved GB/s |")
    lines.append("|:--|--:|--:|--:|--:|")
    planned_bw = {"from_slow": pcie.bidi_h2d, "to_slow": pcie.bidi_d2h}
    for direction, agg in xfer.items():
        if agg["n"] == 0:
            continue
        achieved = agg["bytes"] / agg["us"] / 1e3 if agg["us"] else 0.0
        lines.append(f"| {direction} | {agg['n']} | {agg['bytes'] / 1e9:.1f} | "
                     f"{planned_bw[direction] / 1e3:.1f} | {achieved:.1f} |")
    lines.append("\n## Largest exposed compute gaps (top 10)\n")
    lines.append("| at (ms) | length (ms) | after task | before task |")
    lines.append("|--:|--:|:--|:--|")
    for at, length, prev_t, nxt_t in gaps[:10]:
        lines.append(f"| {at / 1e3:.1f} | {length / 1e3:.2f} | {prev_t} | {nxt_t} |")

    (args.out / "analysis.md").write_text("\n".join(lines) + "\n")
    payload = {
        "config": args.config, "budget_gib": args.budget,
        "sim_tokens_per_s": sim_tok, "real_tokens_per_s": real_tok,
        "replay_gap_pct": replay_gap,
        "families": {f: {"n": fam_count[f], "planned_us": fam_planned[f],
                         "measured_us": fam_measured[f]} for f in fam_measured},
        "transfers": xfer,
        "exposed_idle_us": idle,
    }
    (args.out / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}/analysis.md, analysis.json, trace.json, annotated.json")


if __name__ == "__main__":
    main()
