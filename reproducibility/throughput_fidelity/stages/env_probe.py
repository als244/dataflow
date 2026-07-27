#!/usr/bin/env python
"""Decide what THIS machine can sweep, and write it to env.json.

The same study has to mean something on a datacentre card under a scheduler's
memory grant and on a desktop card with whatever RAM the workstation has, so
the grid cannot be hard-coded. This probe reads the real
limits — device memory, and the HOST limit that actually applies (a cgroup cap
under a batch scheduler, else physical RAM) — then picks:

  * preset       the largest model whose offloaded state fits the host limit,
                 so every box runs the same experiment at the scale it can hold
  * budgets      a fast-memory ladder from the smallest budget that actually
                 plans up to most of the device, which is the axis the study is
                 about -- holding one task is the floor, not the entry price
  * backing_gib  the host ceiling handed to the planner
  * seqs / t_rounds / t_steps   geometry axes; rounds start at one sequence
                 each and double, so a small card's feasible region is inside
                 the ladder rather than below it

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
ROUND_RUNGS = 7              # doublings of tokens-per-round to offer

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
    # MemAvailable, not MemTotal: the allowance becomes a PINNED slab, and
    # memory already spoken for by other tenants cannot be pinned no matter
    # what the machine's nameplate says.
    fields = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            name, _, rest = line.partition(":")
            fields[name] = int(rest.split()[0]) * 1024
    if "MemAvailable" in fields:
        return fields["MemAvailable"], "MemAvailable"
    if "MemTotal" in fields:
        return fields["MemTotal"], "MemTotal"
    raise SystemExit("cannot determine host memory")


def pins_ok(size_bytes):
    """Whether the host will hand out a pinned buffer of this size, right now.

    Asks the same way the ENGINE asks. torch's pin_memory goes through
    cudaHostAlloc, which rounds the request up to a power of two, so a probe
    built on it reports the largest power of two the host can hold rather than
    the largest slab -- 63.8 GiB on a box that grants 82. The number this
    returns becomes the allowance the daemon is then asked to pin, so it has to
    be measured with the daemon's own allocator.

    The refusal arrives as an exception and is the answer being asked for, not
    a fault: the only way to find the ceiling is to ask."""
    from dataflow.runtime.device.cuda import pin_region, unpin_region

    try:
        ptr = pin_region(int(size_bytes))
    except Exception:
        return False
    unpin_region(ptr, int(size_bytes))
    return True


def largest_pinnable(want_bytes, floor_bytes, tolerance_bytes=GIB):
    """The biggest pinned allocation this host will actually give us, at or
    below ``want_bytes``.

    Sizing the allowance by arithmetic on a memory total is wishful: pinned
    pages must be resident and non-swappable, so the ceiling is set by what is
    free right now, by whatever else has already pinned memory, and by the
    kernel's commit limit -- which defaults to half of RAM and has nothing to
    do with what MemAvailable reports. Asking costs a few seconds at startup
    and turns a run-time allocation failure -- which surfaces deep inside the
    first measured cell, after the whole prediction pass has been paid for --
    into a number chosen before any work begins.

    Stepping down alone answers the wrong question, though. A 0.8 step that
    overshoots reports the last size that happened to fit rather than the
    largest that does: on a 125 GiB box whose commit limit was 64.8 GiB, the
    steps landed 90.7 -> 72.6 -> 58.1 and stopped, giving up 6 GiB of host
    allowance to nothing but the size of the step. So the step only brackets
    the ceiling, and a bisect finds it."""
    size = int(want_bytes)
    if pins_ok(size):
        return size
    refused = size
    accepted = None
    while size > floor_bytes:
        size = int(size * 0.8)
        if pins_ok(size):
            accepted = size
            break
        refused = size
    if accepted is None:
        return int(floor_bytes)
    while refused - accepted > tolerance_bytes:
        mid = (accepted + refused) // 2
        if pins_ok(mid):
            accepted = mid
        else:
            refused = mid
    return accepted


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


def smallest_plannable_gib(cfg, floor_gib, backing_bytes, seq, t_step):
    """Bisect for the smallest budget that actually produces a plan.

    Holding one task is necessary but nowhere near sufficient: the planner also
    overlaps that task with the traffic feeding the next one, and measured on
    an 8B model it needs about twice the largest task's working set before it
    can make progress at all. A ladder starting at the floor therefore opens
    with rungs that cannot plan for any geometry -- they are not a finding
    about the machine, just wasted cells that read as "this box cannot do it".

    Bisected against the cheapest variant (everything recomputed, so no saved
    activation can be what fails) at the grid's most permissive REAL cell: one
    round per step, which is the smallest step run whole. Accumulating a step
    over many rounds costs memory even when nothing is saved, so a probe at a
    round size the grid never pairs with that step would report a lower entry
    price than any cell can actually pay. Analytic costs: this only has to
    bound where the ladder starts, and the prediction pass re-decides every
    cell for real."""
    from dataclasses import replace

    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.model_families.families import resolve_family

    probe = replace(cfg, seq_len=seq, batch=max(1, t_step // seq),
                    grad_accum_rounds=1)
    fam = resolve_family(probe)
    program = fam.lower(probe)
    pins = {rw.object_id: rw.options[-1].level for rw in program.recompute_rewrites}
    cheapest = fam.lower(probe, recompute_levels=pins)

    lo, hi = float(floor_gib), float(floor_gib) * 8
    try:
        plan_program(cheapest, fast_memory_capacity=int(hi * GIB),
                     backing_capacity=backing_bytes, recompute=False)
    except ValueError:
        return round(hi, 1)          # even 8x the floor will not plan; say so
    for _ in range(6):
        mid = (lo + hi) / 2
        try:
            plan_program(cheapest, fast_memory_capacity=int(mid * GIB),
                         backing_capacity=backing_bytes, recompute=False)
            hi = mid
        except ValueError:
            lo = mid
    return round(hi, 1)


def budget_ladder(device_bytes, start_gib, step=None):
    """HALF-OCTAVE steps from the smallest plannable budget up to most of the
    device. Doubling is too coarse where it matters: on a large card the whole
    interesting transition (offload-bound to compute-bound) can hide between
    16 and 64 GiB, and three points cannot show a knee. sqrt(2) spacing keeps
    the ladder short while resolving that region, and the device cap is always
    included so the ample end is a real measurement rather than an
    extrapolation."""
    cap = DEVICE_SHARE * device_bytes / GIB
    step = step or BUDGET_STEP
    out, b = [], float(start_gib)
    while b <= cap * 1.001:
        out.append(round(b, 1) if b < 10 else round(b))
        b *= step
    if not out:
        return [round(cap, 1)]
    if cap > out[-1] * 1.1:
        out.append(round(cap, 1))
    return sorted(set(out))


def round_ladder(seqs, rungs=ROUND_RUNGS):
    """Powers of two from ONE SEQUENCE PER ROUND upwards.

    A round's saved activations scale with the tokens in it, so the largest
    round a box can plan falls as the card gets smaller: an 8B model plans a
    64K-token round on an 80 GiB card and nothing above 4K on a 24 GiB one.
    A ladder that starts above a small card's ceiling reports that card as
    unable to train the model at all, which is false.

    Anchoring at the smallest sequence length is the one bound that holds
    everywhere: a round cannot be shorter than a single sequence, and it must
    be a whole number of them. Which rungs a given box can actually plan is
    left to the prediction pass, whose job is exactly that."""
    base = min(seqs)
    return [base * 2 ** k for k in range(rungs)]


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
    # what this host can spare, then what the ENGINE will accept, then what the
    # host will actually pin. The engine's own reserve has to bound this: the
    # daemon refuses a slab that would pin into the last SYSTEM_RESERVE_GiB,
    # and it refuses it at boot -- which is after the whole prediction pass has
    # been paid for. Asking for more than it will grant turns every measured
    # cell into a failure at the end of the run.
    from dataflow.service.hostmem import PinnedSlab, meminfo_available_bytes

    backing = (args.backing_gib * GIB if args.backing_gib
               else (args.host_share or HOST_SHARE) * host_bytes)
    engine_max = meminfo_available_bytes() - int(PinnedSlab.SYSTEM_RESERVE_GIB * GIB)
    backing = min(backing, engine_max)
    backing = largest_pinnable(backing, persist * PERSISTENT_HEADROOM)

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
    seqs = (numbers(args.seqs) if args.seqs else
            [s for s in (1024, 2048, 4096, 8192) if s <= cfg.seq_len * 2])
    t_rounds = (numbers(args.t_rounds) if args.t_rounds else round_ladder(seqs))
    t_steps = (numbers(args.t_steps) if args.t_steps
               else [base // 2, base, base * 2])
    start_gib = smallest_plannable_gib(cfg, floor_gib, backing,
                                       min(seqs), min(t_steps))
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
        "smallest_plannable_gib": start_gib,
        "budgets": (numbers(args.budgets) if args.budgets else
                    budget_ladder(device_bytes, start_gib, args.budget_step)),
        "seqs": seqs,
        "t_rounds": t_rounds,
        "t_steps": t_steps,
        "steps_per_cell": args.steps,
    }
    dst = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "env.json")   # the default results root
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(env, fh, indent=2)
    for k, v in env.items():
        print(f"  {k:20} {v}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
