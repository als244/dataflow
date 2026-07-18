"""Multi-step planning-window analysis: what does a planner that can SEE
across step boundaries do differently from our replayed 1-step plan?

The 1-step replay is correct by the step-boundary invariant
.md); this tool measures what that invariant still COSTS by
using jointly-planned k-step windows (k = 1..K) as an oracle, with every
variable pinned except the window size:

- identical measured task costs (profile cache) and PCIe bandwidths;
- identical budget;
- recompute levels LOCKED to the k=1 plan's choice, replicated per step
  (pass --free-recompute to let the planner adapt per window instead —
  reported as its own finding, never silently mixed into seam numbers).

Per window size it reports, all machine-checked from the annotated chain
and the simulator's event log (never eyeballed plan diffs):

  periodicity   are interior steps IDENTICAL as plans? Each step's tasks
                and directives are canonicalized (step indices stripped
                from step-scoped names; W_*/O_* stay global) and compared
                exactly; the first divergence is printed. A periodic
                interior step is a mechanically extractable replay unit.
  seam state    at each interior cut (end of step s): which objects the
                planner keeps resident on fast ACROSS the boundary
                (structural walk of directives), and which transfers are
                in flight at the cut instant (sim transfer intervals
                spanning it). The resident set is the planner's own
                answer to "what should carry over" — the R that a
                resident-invariant B(R) would pin.
  marginal cost interior-step durations within one plan and the marginal
                makespan of growing the window; their limit is the
                steady-state step cost with full cross-boundary overlap.
  replay regret k=1 sim makespan minus the oracle's interior-step
                duration: the ceiling on what ANY seam mechanism
                (resident carryover, cross-seam dispatch, unit replay)
                could still recover. Post-fix the 1-step plan's own seam
                is ~0.04 s of wall, so sim-to-sim is the fair comparison.

Usage:
    python tools/window_plans.py --config 8b-s1k-bs8ga8 --budgets 12,16,20 --max-steps 4
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

GIB = 1024**3

# task-id shapes emitted by the lowering (step index is always the first
# integer group); parsing asserts full coverage so renamed families fail loudly
_TASK_RE = re.compile(
    r"^(embed_fwd|block_fwd|head_loss|block_recompute|block_bwd"
    r"|embed_bwd|optimizer_embed|optimizer_head|optimizer)_(\d+)((?:_\d+)*)$"
)
# step-scoped object families (second group = step index); W_*/O_* are global
_OBJ_RE = re.compile(
    r"^(tokens|targets|y_embed|y|A|dy_embed|dy|loss|dW_embed|dW_head|dW)"
    r"_(\d+)((?:_\d+)*)$"
)


def task_step(task_id: str) -> int:
    m = _TASK_RE.match(task_id)
    if m is None:
        raise ValueError(f"unrecognized task id shape: {task_id!r}")
    return int(m.group(2))


def canon_task(task_id: str) -> str:
    m = _TASK_RE.match(task_id)
    assert m is not None
    return f"{m.group(1)}_*{m.group(3)}"


def canon_obj(obj_id: str) -> str:
    m = _OBJ_RE.match(obj_id)
    if m is None:
        return obj_id  # global (W_*, O_*)
    return f"{m.group(1)}_*{m.group(3)}"


def canon_directives(task) -> tuple:
    return (
        canon_task(task.id),
        tuple(canon_obj(o) for o in task.inputs),
        tuple((canon_obj(o.id), o.size_bytes) for o in task.outputs),
        tuple(canon_obj(o) for o in task.mutates),
        tuple(sorted(canon_obj(o) for o in task.releases_after)),
        tuple(sorted(canon_obj(t.object_id) for t in task.offload_after)),
        tuple(sorted(canon_obj(t.object_id) for t in task.prefetch_after)),
    )


def replicate_levels(levels_step0: dict[str, int], num_steps: int) -> dict[str, int]:
    """Replicate a k=1 recompute-level map (keys A_0_{r}_{i}) across steps."""
    out = {}
    for key, lvl in levels_step0.items():
        m = _OBJ_RE.match(key)
        assert m is not None and m.group(1) == "A", key
        for s in range(num_steps):
            out[f"A_{s}{m.group(3)}"] = lvl
    return out


def analyze_window(program, log) -> dict:
    """Structured seam/periodicity analysis of one annotated k-step plan."""
    tasks = program.tasks
    steps = sorted({task_step(t.id) for t in tasks})
    k = len(steps)
    pos_of = {t.id: i for i, t in enumerate(tasks)}
    comp = {iv.task_id: (iv.start, iv.end) for iv in log.task_intervals if iv.track == "compute"}
    sizes = {o.id: o.size_bytes for o in program.initial_objects}
    for t in tasks:
        for o in t.outputs:
            sizes[o.id] = o.size_bytes

    # step windows: last task position + compute-end time per step
    last_pos = {s: max(i for i, t in enumerate(tasks) if task_step(t.id) == s) for s in steps}
    cut_time = {s: comp[tasks[last_pos[s]].id][1] for s in steps}

    # --- periodicity: canonicalized step plans compared exactly -------------
    step_plans = {
        s: tuple(canon_directives(t) for t in tasks if task_step(t.id) == s) for s in steps
    }
    # interior steps are 1..k-2: step 0 sees the cold head, step k-1 has no
    # future to prefetch for — comparing either to an interior step would
    # report spurious aperiodicity. Interior pairs exist only for k >= 4.
    periodic = all(step_plans[steps[i]] == step_plans[steps[i + 1]] for i in range(1, k - 2))
    first_diff = None
    for i in range(1, k - 2):
        a, b = step_plans[steps[i]], step_plans[steps[i + 1]]
        if a != b:
            for j, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    first_diff = {"steps": (steps[i], steps[i + 1]), "task_index": j,
                                  "a": x[0], "a_dirs": x[4:], "b": y[0], "b_dirs": y[4:]}
                    break
            else:
                first_diff = {"steps": (steps[i], steps[i + 1]),
                              "task_index": min(len(a), len(b)), "detail": "length differs"}
            break

    # --- seam-resident sets: structural walk of fast-copy lifetimes ---------
    has_fast = {o.id for o in program.initial_objects if o.location == "fast"}
    resident_at_cut: dict[int, list] = {s: [] for s in steps[:-1]}
    cut_positions = {last_pos[s]: s for s in steps[:-1]}
    for i, t in enumerate(tasks):
        for o in t.outputs:
            if o.location == "fast":
                has_fast.add(o.id)
        for trig in t.prefetch_after:
            has_fast.add(trig.object_id)
        for oid in t.releases_after:
            has_fast.discard(oid)
        for trig in t.offload_after:
            has_fast.discard(trig.object_id)  # sim offload = move to backing
        if i in cut_positions:
            s = cut_positions[i]
            resident_at_cut[s] = sorted(has_fast)

    # --- in-flight transfers at each cut instant (sim intervals) ------------
    inflight_at_cut = {}
    for s in steps[:-1]:
        ct = cut_time[s]
        spans = []
        for iv in log.task_intervals:
            if iv.track in ("from_slow", "to_slow") and iv.start < ct < iv.end:
                oid = iv.task_id.split(":", 1)[1].split("#", 1)[0]
                spans.append({"dir": iv.track, "object": oid, "bytes": sizes.get(oid, 0)})
        inflight_at_cut[s] = spans

    # --- cross-seam prefetches: fired in step s, first used in step > s -----
    next_use: dict[str, list[int]] = {}
    for i, t in enumerate(tasks):
        for oid in t.inputs:
            next_use.setdefault(oid, []).append(i)
    cross_bytes = {s: 0 for s in steps[:-1]}
    for i, t in enumerate(tasks):
        s = task_step(t.id)
        if s not in cross_bytes:
            continue
        for trig in t.prefetch_after:
            uses = [u for u in next_use.get(trig.object_id, []) if u > i]
            if uses and task_step(tasks[uses[0]].id) > s:
                cross_bytes[s] += sizes.get(trig.object_id, 0)

    def _sum(objs):
        return sum(sizes.get(o, 0) for o in objs)

    makespan = max(e for _, e in comp.values())
    step_durations = {
        steps[i]: (cut_time[steps[i]] - (cut_time[steps[i - 1]] if i else 0.0))
        for i in range(k)
    }
    return {
        "num_steps": k,
        "makespan_us": makespan,
        "step_durations_us": {str(s): step_durations[s] for s in steps},
        "interior_periodic": periodic if k > 3 else None,
        "first_interior_diff": first_diff,
        "seams": {
            str(s): {
                "resident_fast_objects": resident_at_cut[s],
                "resident_fast_gib": _sum(resident_at_cut[s]) / GIB,
                "resident_global_gib": _sum(
                    o for o in resident_at_cut[s] if _OBJ_RE.match(o) is None
                ) / GIB,
                "inflight": inflight_at_cut[s],
                "inflight_gib": sum(x["bytes"] for x in inflight_at_cut[s]) / GIB,
                "cross_seam_prefetch_gib": cross_bytes[s] / GIB,
            }
            for s in steps[:-1]
        },
    }


def main() -> None:
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.model_families.llama3_blocks import build_resolver
    from dataflow_training.model_families.llama3 import dims_of, lower_llama3
    from dataflow_training.lowering.planning import plan_program, simulate_program
    from dataflow_training.run.profiling import apply_measured_costs, cached_pcie, load_or_profile
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    from bench_train import CONFIGS  # same config registry as the sweeps

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", choices=sorted(CONFIGS), default="8b-s1k-bs8ga8")
    parser.add_argument("--budgets", type=str, default="12,16,20")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument(
        "--free-recompute", action="store_true",
        help="let the recompute planner adapt per window size (default: lock "
             "levels to the k=1 choice so the window is the ONLY variable)",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/window-plans"))
    args = parser.parse_args()

    base_cfg = CONFIGS[args.config]
    backend = CudaBackend()
    pcie = cached_pcie(backend)
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"config": args.config, "free_recompute": args.free_recompute, "budgets": {}}

    for gib in [float(x) for x in args.budgets.split(",")]:
        cap = int(gib * GIB)
        rows = []
        locked_levels: dict[str, int] | None = None
        for k in range(1, args.max_steps + 1):
            cfg = replace(base_cfg, num_steps=k)
            dims = dims_of(cfg)

            def build_raw(levels=None, cfg=cfg):
                return replace(
                    lower_llama3(cfg, recompute_levels=levels),
                    bandwidth_from_slow=pcie.bidi_h2d,
                    bandwidth_to_slow=pcie.bidi_d2h,
                )

            program = build_raw()
            profiles = load_or_profile(program, build_resolver(dims), backend)
            rc_all = {rw.object_id: 1 for rw in program.recompute_rewrites}
            profiles.update(load_or_profile(build_raw(rc_all), build_resolver(dims), backend))

            t0 = time.perf_counter()
            if k == 1 or args.free_recompute:
                planned = plan_program(
                    apply_measured_costs(program, profiles), fast_memory_capacity=cap,
                    recompute=True,
                    build_variant=lambda lv: apply_measured_costs(build_raw(lv), profiles),
                )
                levels = planned.recompute_levels
                if k == 1:
                    locked_levels = dict(levels)
            else:
                assert locked_levels is not None
                levels = replicate_levels(locked_levels, k)
                planned = plan_program(
                    apply_measured_costs(build_raw(levels), profiles),
                    fast_memory_capacity=cap,
                )
            plan_s = time.perf_counter() - t0

            log = simulate_program(planned.program)
            row = analyze_window(planned.program, log)
            row.update({
                "budget_gib": gib,
                "planning_s": plan_s,
                "recompute_per_step": sum(1 for v in levels.values() if v)
                // max(row["num_steps"], 1),
            })
            rows.append(row)

            seam_line = ""
            if row["seams"]:
                s0 = row["seams"][min(row["seams"])]
                seam_line = (
                    f" | seam0: resident {s0['resident_fast_gib']:.2f} GiB "
                    f"(global {s0['resident_global_gib']:.2f}), inflight "
                    f"{s0['inflight_gib']:.2f}, x-seam prefetch "
                    f"{s0['cross_seam_prefetch_gib']:.2f}"
                )
            durs = list(row["step_durations_us"].values())
            print(
                f"{gib:>5g} GiB k={k}: makespan {row['makespan_us']/1e6:7.3f}s "
                f"steps {[round(d/1e6, 3) for d in durs]} "
                f"periodic={row['interior_periodic']}{seam_line} "
                f"(planned in {plan_s:.0f}s)",
                flush=True,
            )
        # marginal + regret
        for i in range(1, len(rows)):
            rows[i]["marginal_step_us"] = rows[i]["makespan_us"] - rows[i - 1]["makespan_us"]
        k1 = rows[0]["makespan_us"]
        for row in rows[1:]:
            interior = [d for s, d in row["step_durations_us"].items()
                        if 0 < int(s) < row["num_steps"] - 1] or list(
                        row["step_durations_us"].values())[1:]
            row["replay_regret_us"] = k1 - min(interior)
        best = rows[-1].get("replay_regret_us", 0.0)
        print(f"{gib:>5g} GiB: replay regret (k=1 sim vs best oracle interior step) "
              f"= {best/1e3:.0f} ms/step ({best/k1:+.1%})\n", flush=True)
        report["budgets"][f"{gib:g}"] = rows

    out = args.out / f"window-{args.config}-{'free' if args.free_recompute else 'locked'}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
