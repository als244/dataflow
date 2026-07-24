#!/usr/bin/env python
"""Deep-dive ONE cell: measured-cost sim prediction vs the real engine run, to
locate the ample-budget fidelity gap (ratio ~1.2-1.33). Plans with PROFILED
costs (matching the measure-leg prediction), runs K real steps through the
daemon keeping the last step's trace, then decomposes makespan into
compute-busy / idle-gaps / transfer for BOTH sim and real. Emits the webapp
bundle (real vs sim) + prints where they diverge.
"""
import argparse
import json
import os
import sys
from dataclasses import replace


def find_root(s):
    d = s
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src", "dataflow_training")) and \
           os.path.isdir(os.path.join(d, "tools", "export")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("no root")


ROOT = find_root(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "tools", "export"), os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import trace_real_run as TR                                       # noqa: E402
from dataflow.core.jsonio import program_to_dict                 # noqa: E402
from dataflow.core.convert import to_webapp_program              # noqa: E402
from dataflow_training.run import presets as P                   # noqa: E402
from dataflow_training.run.presets import cfg_dict, resolver_family  # noqa: E402
from dataflow_training.run.driver import (engine_client, init_model,  # noqa: E402
                                          plan_at_budget)
from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.run.recipe import Recipe                   # noqa: E402


def union_len(intervals):
    """total covered time of a set of (start,end) — collapses overlap."""
    out = 0.0
    for s, e in sorted(intervals):
        if not out or s > cur_e:
            out += e - s
            cur_s, cur_e = s, e
        elif e > cur_e:
            out += e - cur_e
            cur_e = e
    return out


def tracks(log):
    d = {}
    for iv in log["task_intervals"]:
        d.setdefault(iv["track"], []).append((iv["start"], iv["end"]))
    return d


def makespan(log):
    return max((iv["end"] for iv in log["task_intervals"]), default=0.0)


def summarize(name, log):
    tk = tracks(log)
    ms = makespan(log)
    busy = {t: union_len(iv) for t, iv in tk.items()}
    compute = max(busy.values()) if busy else 0.0
    compute_track = max(busy, key=busy.get) if busy else None
    idle = ms - busy.get(compute_track, 0.0)
    print(f"  {name:>6}: makespan {ms/1e3:8.1f}ms | "
          f"busy {{" + ", ".join(f'{t}:{v/1e3:.0f}ms' for t, v in sorted(busy.items(), key=lambda kv: -kv[1])) + "}}"
          f" | idle-on-compute {idle/1e3:.1f}ms")
    return ms, compute_track, busy, idle


def gaps_on_track(log, track):
    ivs = sorted((iv["start"], iv["end"], iv["task_id"])
                 for iv in log["task_intervals"] if iv["track"] == track)
    out = []
    for i in range(1, len(ivs)):
        gap = ivs[i][0] - ivs[i - 1][1]
        if gap > 0:
            out.append((gap, ivs[i][2], ivs[i - 1][2]))
    return sorted(out, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--t-round", dest="tr", type=int, default=32768)
    ap.add_argument("--t-step", dest="ts", type=int, default=131072)
    ap.add_argument("--budget", type=float, default=64.0)
    ap.add_argument("--opt", default="adamw")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--backing-gib", dest="backing", type=float, default=130.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = replace(P.resolve_preset("llama3_8b"), opt_policy=a.opt)
    cfg = replace(base, seq_len=a.seq, grad_accum_rounds=a.ts // a.tr, batch=a.tr // a.seq)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=max(1, a.steps // 3),
                    total_steps=a.steps)
    print(f"cell seq{a.seq} tr{a.tr} ts{a.ts} b{a.budget:g} {a.opt}  "
          f"(batch {cfg.batch}, ga {cfg.grad_accum_rounds})")

    planned = plan_at_budget(cfg, a.budget, measured=True)   # profiled-cost plan
    cd = cfg_dict(cfg)
    R = cfg.grad_accum_rounds
    with engine_client(backing_gib=a.backing, log=lambda *x, **k: None) as client:
        init_model(client, resolver_family(cfg), cd, seed=11)
        stepper = legacy_block_pipeline(cfg)(None)
        valid0, lens0 = TR.put_packed_step(client, stepper, cfg.max_tokens)
        reg = client.register_program(
            program_to_dict(planned.program),
            resolver={"kind": "model_family", "family": resolver_family(cfg),
                      "cfg": cd, "hyper": recipe.hyper_spec()})
        if reg["bindings"]["missing_inputs"]:
            raise SystemExit("unbound: " + str(reg["bindings"]["missing_inputs"]))
        last = None
        for step in range(a.steps):
            valid, lens = (valid0, lens0) if step == 0 else \
                TR.put_packed_step(client, stepper, cfg.max_tokens)
            out = client.run(reg["prog_id"],
                             args={"step": step, "valid_rows": valid, "seq_lens": lens},
                             fetch=[f"loss_0_{r}" for r in range(R)],
                             trace=(step == a.steps - 1))
            if out.get("state") != "done":
                raise SystemExit(f"step {step}: {out}")
            if "trace" in out:
                last = out["trace"]
    if last is None:
        raise SystemExit("no trace")

    measured = TR.measured_log_from_trace(last)
    sim = TR.sim_log_for(planned.program)
    print("\n=== makespan decomposition (compute-busy vs idle-gaps vs transfer) ===")
    ms_r, ct_r, busy_r, idle_r = summarize("REAL", measured)
    ms_s, ct_s, busy_s, idle_s = summarize("SIM", sim)
    print(f"\n  ratio real/sim = {ms_r/ms_s:.2f}   extra = {(ms_r-ms_s)/1e3:.1f}ms")
    print(f"  of which  compute-busy Δ = {(busy_r.get(ct_r,0)-busy_s.get(ct_s,0))/1e3:+.1f}ms"
          f"   idle-gaps Δ = {(idle_r-idle_s)/1e3:+.1f}ms")
    print("\n=== biggest idle gaps on REAL compute track (accidental syncs?) ===")
    for gap, tid, prev in gaps_on_track(measured, ct_r)[:12]:
        print(f"  {gap/1e3:7.2f}ms before {tid}   (after {prev})")

    stem = a.out or f"dd_seq{a.seq}_tr{a.tr}_ts{a.ts}_b{a.budget:g}_{a.opt}"
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
    os.makedirs(outdir, exist_ok=True)
    json.dump({"format": "dataflow-measured-run/v1",
               "meta": {"cell": [a.seq, a.tr, a.ts, a.budget, a.opt], "steps": a.steps},
               "webapp": to_webapp_program(planned.program),
               "log": measured, "sim_log": sim},
              open(os.path.join(outdir, stem + ".measured.json"), "w"), indent=2)
    print(f"\nwrote traces/{stem}.measured.json  (webapp real-vs-sim bundle)")
    print(f"[parity] {TR.parity_line(measured, sim)}")


if __name__ == "__main__":
    main()
