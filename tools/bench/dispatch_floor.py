"""Isolated inter-task dispatch overhead: the floor benchmark.

A linear chain of K trivial tasks through the real engine on the real
CUDA backend — every input resident in fast memory, no transfers — so
the device-timeline gap between consecutive kernels is exactly
[completion detect + retire task N + dispatch task N+1]: the
dispatcher's floor, isolated from any model. Each task consumes one
4 KiB object, produces the next, and releases its input after use, so
the retire path does real table/ledger/pool work per task.

The reported number is the same quantity the production NVTX traces
show as the inter-task gap (device-idle between kernel end and next
kernel start), measured from the run's own trace intervals — no
profiler required. Runs in seconds; use before/after every dispatcher
change.

Usage:
  python tools/bench/dispatch_floor.py [--tasks 400] [--size 4096]
      [--repeats 3] [--stats] [--json PATH]

--stats additionally enables DATAFLOW_DISPATCH_STATS for the phase
breakdown (note: the stats instrumentation itself costs ~20 us/task,
so the clean gap number comes from the default mode).
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def build_program(n_tasks, size_bytes, shape):
    """minimal: 1 in / 1 out / 1 release — the machinery floor.
    block: 3 in (chain + 2 persistent weights) / 2 out / 2 releases —
    the width of a real gpt2 block's bookkeeping (transfer directives
    excluded on purpose: lanes idle => the gap is pure dispatcher)."""
    from dataflow.core import ObjectSpec, OutputSpec, Program, TaskSpec

    n_weights = 8
    tasks = []
    initial = [ObjectSpec(id="x0", size_bytes=size_bytes, location="fast")]
    if shape == "block":
        initial += [ObjectSpec(id=f"w{j}", size_bytes=size_bytes,
                               location="fast") for j in range(n_weights)]
    for i in range(n_tasks):
        if shape == "minimal":
            inputs = (f"x{i}",)
            outputs = (OutputSpec(id=f"x{i + 1}", size_bytes=size_bytes,
                                  location="fast"),)
            releases = (f"x{i}",)
        else:
            inputs = (f"x{i}", f"w{i % n_weights}",
                      f"w{(i + 1) % n_weights}")
            outputs = (OutputSpec(id=f"x{i + 1}", size_bytes=size_bytes,
                                  location="fast"),
                       OutputSpec(id=f"a{i}", size_bytes=size_bytes,
                                  location="fast"))
            releases = (f"x{i}",) if i == 0 else (f"x{i}", f"a{i - 1}")
        tasks.append(TaskSpec(
            id=f"t{i}", inputs=inputs, outputs=outputs,
            releases_after=releases, runtime_us=1.0,
        ))
    return Program(
        name="dispatch-floor",
        initial_objects=tuple(initial),
        tasks=tuple(tasks),
        fast_memory_capacity=max(64 * 1024 * 1024,
                                 8 * n_tasks * size_bytes),
    )


class FloorExecutable:
    """``enqueues`` tiny fill kernels on the compute stream — real
    enqueues, real completion event, negligible device time. More than
    one enqueue reproduces the launch-tail dynamic (the dispatcher is
    still enqueueing when short device work finishes)."""

    def __init__(self, enqueues):
        self.enqueues = enqueues

    def launch(self, ctx):
        import torch

        from dataflow.runtime.interop import external_stream, torch_view

        out_spec = ctx.task.outputs[0]
        buf = ctx.outputs[out_spec.id]
        es = external_stream(ctx.stream)
        chunk = max(64, buf.size_bytes // self.enqueues)
        with torch.cuda.stream(es):
            view = torch_view(buf, (buf.size_bytes,), torch.uint8)
            for k in range(self.enqueues):
                lo = (k * chunk) % buf.size_bytes
                view[lo:lo + 64].fill_(1)


def gaps_from_trace(trace):
    """Device-timeline gaps between consecutive compute intervals —
    the exact quantity the production traces show between blocks."""
    compute = sorted((iv for iv in trace.intervals if iv.track == "compute"),
                     key=lambda iv: iv.start)
    return [nxt.start - prev.end
            for prev, nxt in zip(compute, compute[1:])]


def run_once(n_tasks, size_bytes, shape, enqueues):
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend

    executable = FloorExecutable(enqueues)

    def floor_resolver(task):
        return executable

    program = build_program(n_tasks, size_bytes, shape)
    fake = FakeBackend()
    dry_values = {o.id: fake.alloc(o.location, o.size_bytes)
                  for o in program.initial_objects}
    dry = Engine(FakeBackend()).execute(program, resolver=None,
                                        initial_buffers=dry_values)
    backend = CudaBackend()
    values = {o.id: backend.alloc(o.location, o.size_bytes)
              for o in program.initial_objects}
    from dataflow.service.execution import DisplaySub

    # production fleet runs always carry a display renamer (per-step
    # NVTX names); include it so rename-path costs are measured
    renamer = DisplaySub(r"^t(\d+)$", "task_\\g<1>")
    t0 = time.perf_counter()
    result = Engine(backend).execute(
        program, resolver=floor_resolver, initial_buffers=values,
        pool_prewarm=dry.pool_demand, annotate_rename=renamer)
    wall = time.perf_counter() - t0
    if not result.outcome.is_success:
        raise RuntimeError(f"floor run failed: {result.outcome.message}\n"
                           f"{result.outcome.traceback_text}")
    gaps = gaps_from_trace(result.trace)
    result.close()
    dry.close()
    return wall, gaps


def calibrate():
    """The three machine primitives the floor is built from — run on
    each box so cross-machine gap ratios decompose into causes:
    interpreter speed (dict/method op cost), completion-poll cost
    (cudaEventQuery), and enqueue cost (tiny kernel launch)."""
    import torch

    from dataflow.runtime.device.cuda import CudaBackend

    backend = CudaBackend()
    d = {}
    t0 = time.perf_counter()
    n = 200_000
    for i in range(n):
        d[i & 1023] = i
        d.get(i & 511)
    py_ns = (time.perf_counter() - t0) / (2 * n) * 1e9
    stream = backend.create_stream("cal")
    ev = backend.record_event(stream)
    t0 = time.perf_counter()
    for _ in range(2000):
        backend.event_complete(ev)
    q_us = (time.perf_counter() - t0) / 2000 * 1e6
    x = torch.ones(64, device="cuda", dtype=torch.uint8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2000):
        x.fill_(1)
    torch.cuda.synchronize()
    enq_us = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"calibration: python dict op {py_ns:6.0f} ns   "
          f"eventQuery {q_us:5.1f} us   enqueue {enq_us:5.1f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shape", choices=("minimal", "block"),
                    default="block")
    ap.add_argument("--enqueues", type=int, default=24,
                    help="kernel enqueues per task (block-launch tail)")
    ap.add_argument("--stats", action="store_true",
                    help="enable DATAFLOW_DISPATCH_STATS (adds its own "
                         "~20 us/task; use for attribution, not the number)")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="print machine primitives (interpreter, "
                         "eventQuery, enqueue) and exit")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return

    if args.stats:
        os.environ["DATAFLOW_DISPATCH_STATS"] = "1"

    all_gaps = []
    walls = []
    for r in range(args.repeats):
        wall, gaps = run_once(args.tasks, args.size, args.shape,
                              args.enqueues)
        walls.append(wall)
        if r > 0:  # first repeat warms CUDA context/pools; exclude
            all_gaps.extend(gaps)
    q = statistics.quantiles(all_gaps, n=100)
    p50, p90, p99 = q[49], q[89], q[98]
    best_wall = min(walls[1:]) if len(walls) > 1 else walls[0]
    print(f"dispatch floor ({args.shape} shape, {args.tasks} tasks x "
          f"{args.size} B, {args.enqueues} enqueues/task, "
          f"{args.repeats - 1} measured repeats):")
    print(f"  inter-task device gap: p50 {p50:7.1f} us   "
          f"p90 {p90:7.1f}   p99 {p99:7.1f}   max {max(all_gaps):7.1f}")
    print(f"  wall/task: {best_wall / args.tasks * 1e6:7.1f} us")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"tasks": args.tasks, "size": args.size,
                       "shape": args.shape, "enqueues": args.enqueues,
                       "gap_p50_us": p50, "gap_p90_us": p90,
                       "gap_p99_us": p99, "gap_max_us": max(all_gaps),
                       "wall_per_task_us": best_wall / args.tasks * 1e6},
                      fh, indent=1)
        print(f"  json -> {args.json_out}")


main()
