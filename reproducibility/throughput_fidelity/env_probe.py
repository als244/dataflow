#!/usr/bin/env python
"""Decide what THIS machine can sweep, and write it to env.json.

The same study runs on very different boxes (an 80 GiB datacentre card with a
140 GiB allocation, a 32 GiB desktop card with 188 GiB of host RAM, a 24 GiB
card with 126 GiB), so the grid cannot be hard-coded. This probe reads the real
limits — device memory, and the HOST limit that actually applies (a cgroup cap
under a batch scheduler, else physical RAM) — then picks:

  * preset       the largest model whose offloaded state fits the host limit,
                 so every box runs the same experiment at the scale it can hold
  * budgets      a fast-memory ladder from a floor that can hold one task up to
                 most of the device, which is the axis the study is about
  * backing_gib  the host ceiling handed to the planner
  * seqs / t_rounds / t_steps   geometry axes, scaled to the device class

Cell-level feasibility is NOT decided here: the prediction pass plans every
candidate and records the ones the planner cannot fit as INFEASIBLE rows, and
the measure subset is drawn from what actually survived. This probe only bounds
the candidate space so that pass is not mostly wasted work.
"""
import argparse
import json
import os
import sys


def find_root(start):
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src", "dataflow_training")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("repo root not found")


ROOT = find_root(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

GIB = 1024 ** 3


def numbers(text):
    """"1024,4096" -> [1024, 4096]; "4,5.7" -> [4, 5.7]."""
    return [float(x) if "." in x else int(x) for x in text.split(",")]

# largest first: the biggest model a box can hold is the most informative, since
# the whole point is training that does NOT fit in fast memory
PRESET_LADDER = ["llama3_8b", "l3_1b", "l3_760m", "l3_350m", "l3_125m"]

# Persistent state (weights + optimizer + gradients) is the HARD floor: it lives
# in host memory for the whole run and no planner choice can shrink it. Saved
# activations on top of it are ELASTIC — the backing ceiling is a planner input,
# so a smaller host allowance simply makes the recompute planner keep fewer
# contexts and re-derive more. That is why a box with less RAM still runs the
# same model: it recomputes its way into the space it has, which is exactly the
# regime this runtime exists for.
PERSISTENT_HEADROOM = 1.15   # staging/copies alongside the persistent floor
HOST_SHARE = 0.8             # of the applicable host limit
BUDGET_STEP = 2 ** 0.5       # ratio between budget rungs
DEVICE_SHARE = 0.85          # of device memory the largest budget may use

# The top of the backing ladder is what the host can spare, not a multiple of
# the persistent floor. An unconstrained plan of this shape wants roughly twice
# that floor in saved contexts, so a tighter ceiling would silently force
# recompute and hide the cells that get FASTER with more host memory. Whether
# the top rung is still improving is itself a result: it says the box is
# host-limited rather than device-limited.


def host_limit_bytes():
    """The host memory that actually applies: what a batch scheduler granted
    this job, else a cgroup cap, else physical RAM. The scheduler's own
    variable is checked FIRST because a compute node reports its full physical
    RAM through both /proc/meminfo and (often) the cgroup, while the job may
    only own a slice of it — sizing a pinned slab from the node total gets the
    job killed."""
    for var, unit in (("SLURM_MEM_PER_NODE", 1024 ** 2),
                      ("SLURM_MEM_PER_CPU", 1024 ** 2)):
        raw = os.environ.get(var)
        if raw and raw.isdigit():
            total = int(raw) * unit
            if var == "SLURM_MEM_PER_CPU":
                total *= int(os.environ.get("SLURM_CPUS_ON_NODE", "1"))
            return total, var
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw and raw != "max":
                val = int(raw)
                # an unset v1 limit reads as a huge sentinel
                if 0 < val < (1 << 62):
                    return val, f"cgroup ({os.path.basename(path)})"
        except (OSError, ValueError):
            pass
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024, "MemTotal"
    raise SystemExit("cannot determine host memory")


def persistent_bytes(preset, opt):
    """Bytes the run must keep for the whole step: parameters, optimizer state
    and gradients. Lowering only — no device needed."""
    from dataclasses import replace

    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.run import presets as P

    cfg = replace(P.resolve_preset(preset), opt_policy=opt)
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    sizes = program.object_sizes()
    return sum(sizes[o.id] for o in program.initial_objects), cfg


def budget_ladder(device_bytes, floor_gib, step=None):
    """HALF-OCTAVE steps from a floor that can hold one task up to most of the
    device. Doubling is too coarse where it matters: on a large card the whole
    interesting transition (offload-bound to compute-bound) can hide between
    16 and 64 GiB, and three points cannot show a knee. sqrt(2) spacing keeps
    the ladder short while resolving that region, and the device cap is always
    included so the ample end is a real measurement rather than an
    extrapolation."""
    cap = DEVICE_SHARE * device_bytes / GIB
    step = step or BUDGET_STEP
    out, b = [], float(floor_gib)
    while b <= cap * 1.001:
        out.append(round(b, 1) if b < 10 else round(b))
        b *= step
    if not out:
        return [round(cap, 1)]
    if cap > out[-1] * 1.1:
        out.append(round(cap, 1))
    return sorted(set(out))


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="override preset selection")
    ap.add_argument("--opt", default="adamw", help="optimizer the sizing assumes")
    ap.add_argument("--seqs", default=None, help="sequence lengths, comma separated")
    ap.add_argument("--t-rounds", dest="t_rounds", default=None,
                    help="tokens per round, comma separated")
    ap.add_argument("--t-steps", dest="t_steps", default=None,
                    help="tokens per optimizer step, comma separated")
    ap.add_argument("--budgets", default=None,
                    help="GPU memory budgets in GiB, comma separated")
    ap.add_argument("--budget-step", dest="budget_step", type=float, default=None,
                    help="ratio between budget rungs (default sqrt(2))")
    ap.add_argument("--host-share", dest="host_share", type=float, default=None,
                    help="fraction of the host limit offered (default 0.8)")
    ap.add_argument("--backing-gib", dest="backing_gib", type=float, default=None,
                    help="host allowance outright, ignoring --host-share")
    ap.add_argument("--steps", type=int, default=6,
                    help="steps per measured cell, recorded for downstream use")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("env_probe must run on the target device")
    props = torch.cuda.get_device_properties(0)
    device_bytes = props.total_memory
    host_bytes, host_src = host_limit_bytes()

    # The link rate the planner actually prices transfers at. The engine
    # measures this itself, with BOTH directions in flight, and caches it —
    # which is the number plans consume, and lower than either direction
    # benchmarked alone. Re-benchmarking it here would report a different,
    # prettier figure that nothing uses.
    try:
        from dataflow.runtime.device.cuda import CudaBackend
        from dataflow_training.run.profiling import cached_pcie
        pcie = cached_pcie(CudaBackend())
        # the engine carries these as bytes per microsecond
        link = {"bidi_h2d_gbs": round(pcie.bidi_h2d / 1000, 1),
                "bidi_d2h_gbs": round(pcie.bidi_d2h / 1000, 1)}
    except Exception as exc:
        link = {"error": str(exc)[:80]}

    chosen = None
    for preset in ([args.preset] if args.preset else PRESET_LADDER):
        persist, cfg = persistent_bytes(preset, args.opt)
        if args.preset or persist * PERSISTENT_HEADROOM <= HOST_SHARE * host_bytes:
            chosen = (preset, persist, cfg)
            break
    if chosen is None:
        raise SystemExit("no preset fits this host")
    preset, persist, cfg = chosen
    # what this host can spare; the planner takes less when it wants less
    backing = (args.backing_gib * GIB if args.backing_gib
               else (args.host_share or HOST_SHARE) * host_bytes)

    # a task needs its own inputs+outputs resident; that bounds the useful floor
    from dataflow_training.model_families.families import resolve_family
    program = resolve_family(cfg).lower(cfg)
    sizes = program.object_sizes()
    floor = max(sum(sizes[i] for i in t.inputs) + sum(o.size_bytes for o in t.outputs)
                for t in program.tasks)
    floor_gib = 2
    while floor_gib * GIB < floor:
        floor_gib *= 2

    big = device_bytes / GIB >= 40
    base = 131072 if big else 65536
    env = {
        "host": os.uname().nodename,
        "device": props.name,
        "device_gib": round(device_bytes / GIB, 1),
        "link": link,
        "host_limit_gib": round(host_bytes / GIB, 1),
        "host_limit_source": host_src,
        "preset": preset,
        "opt_default": args.opt,
        "persistent_gib": round(persist / GIB, 1),
        "backing_gib": round(backing / GIB, 1),
        "task_floor_gib": round(floor / GIB, 2),
        "budgets": (numbers(args.budgets) if args.budgets else
                    budget_ladder(device_bytes, floor_gib, args.budget_step)),
        "seqs": (numbers(args.seqs) if args.seqs else
                 [s for s in (1024, 2048, 4096, 8192) if s <= cfg.seq_len * 2]),
        "t_rounds": (numbers(args.t_rounds) if args.t_rounds
                     else [8192, 16384, 32768, 65536]),
        "t_steps": (numbers(args.t_steps) if args.t_steps
                    else [base // 2, base, base * 2]),
        "steps_per_cell": args.steps,
    }
    dst = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "env.json")
    with open(dst, "w") as fh:
        json.dump(env, fh, indent=2)
    for k, v in env.items():
        print(f"  {k:20} {v}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
