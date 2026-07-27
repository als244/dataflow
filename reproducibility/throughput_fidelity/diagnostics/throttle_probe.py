#!/usr/bin/env python
"""Sustained-load test for the block_bwd 40->60ms gap.

Runs ONE block_bwd executable two ways and records per-iteration GPU time:
  * sync-between  : record-launch-record-cudaEventSynchronize per iter (the
                    profiler's method) -> the ~40.9ms isolated baseline.
  * back-to-back  : launch N times with NO sync between, one event per iter,
                    sync only at the end -> sustained SM load.

If the gap is clock/power throttling, back-to-back per-iter time should RAMP
from ~40ms toward ~60ms as the GPU saturates and power-caps. If it stays flat
at ~40ms, throttle is NOT the cause. Emits per-iter arrays (JSON) for plotting.
"""
import argparse
import json
import os
import statistics
import sys
from dataclasses import replace


def find_root(s):
    d = s
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src", "dataflow_training")) and \
           os.path.isdir(os.path.join(d, "tools", "bench")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("no root")


ROOT = find_root(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                                                     # noqa: E402
from cuda.bindings import runtime as cudart                      # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend, _check     # noqa: E402
from dataflow.runtime.executable import TaskContext              # noqa: E402
from dataflow.runtime.interop import torch_view                  # noqa: E402
from dataflow_training.run import presets as P                   # noqa: E402
from dataflow_training.run.profiling import thermal_soak         # noqa: E402
from dataflow_training.model_families.families import resolve_family  # noqa: E402
from dataflow_training.data.segments import uniform_segments     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--t-round", dest="tr", type=int, default=32768)
    ap.add_argument("--t-step", dest="ts", type=int, default=131072)
    ap.add_argument("--opt", default="adamw")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--soak", type=float, default=1.0)
    ap.add_argument("--sets", type=int, default=1,
                    help="rotate over N distinct buffer sets (pipeline touches a different\n                          layer's buffers each task; 1 = reuse one hot set)")
    ap.add_argument("--fill", default="uninit", choices=["uninit","zeros","randn"],
                    help="float input payload: profiler-style uninit, explicit zeros, or real-scale randn")
    ap.add_argument("--out", default="throttle.json")
    a = ap.parse_args()

    cfg = replace(replace(P.resolve_preset("llama3_8b"), opt_policy=a.opt),
                  seq_len=a.seq, grad_accum_rounds=a.ts // a.tr, batch=a.tr // a.seq)
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    dims = fam.derive_dims(cfg)
    resolver = fam.build_resolver(dims)
    backend = CudaBackend()
    sizes = program.object_sizes()
    metas = {o.id: o.tensor for o in program.initial_objects}
    for t in program.tasks:
        for o in t.outputs:
            metas[o.id] = o.tensor
    stream = backend.create_stream("compute")
    segs = uniform_segments(dims, program)
    if getattr(backend, "physical", False):
        one = next(iter(segs.values())).on(f"cuda:{backend.device}")
        segs = {r: one for r in segs}
    run_args = {"segments": segs}

    task = next(t for t in program.tasks
                if t.group == "backward" and t.compute_block_key.endswith("_bwd"))
    local = []

    def buf(sz):
        b = backend.alloc("fast", sz)
        local.append(b)
        return b

    def make_inputs():
        """One full set of input buffers for the task."""
        d = {}
        for obj in task.inputs:
            b = buf(sizes[obj])
            m = metas.get(obj)
            if m is not None and m.dtype == "int32":
                torch_view(b, (sizes[obj] // 4,), torch.int32).fill_(0)
            elif a.fill != "uninit":
                v = torch_view(b, (sizes[obj] // 2,), torch.bfloat16)
                if a.fill == "randn":
                    v.normal_(0.0, 1.0)
                else:
                    v.zero_()
            d[obj] = b
        return d

    inb = {}
    for obj in task.inputs:
        b = buf(sizes[obj])
        m = metas.get(obj)
        if m is not None and m.dtype == "int32":
            torch_view(b, (sizes[obj] // 4,), torch.int32).fill_(0)
        elif a.fill != "uninit":
            # float payloads: the profiler leaves these UNINITIALIZED; compare
            # against explicit zeros and REAL-scale random data (switching
            # activity -> power draw -> sustained clock).
            v = torch_view(b, (sizes[obj] // 2,), torch.bfloat16)
            if a.fill == "randn":
                v.normal_(0.0, 1.0)
            else:
                v.zero_()
        inb[obj] = b
    outb = {o.id: buf(o.size_bytes) for o in task.outputs}
    mutb = {m: inb[m] for m in task.mutates}
    ex = resolver(task)
    ctx = TaskContext(task=task, stream=stream, inputs=inb, outputs=outb,
                      mutates=mutb, backend=backend, run_args=run_args)
    ctxs = [ctx]
    for _ in range(max(0, a.sets - 1)):
        ib = make_inputs()
        ob = {o.id: buf(o.size_bytes) for o in task.outputs}
        ctxs.append(TaskContext(task=task, stream=stream, inputs=ib, outputs=ob,
                                mutates={m: ib[m] for m in task.mutates},
                                backend=backend, run_args=run_args))
    fill = getattr(ex, "profile_fill", None)
    if fill:
        for c in ctxs:
            fill(c)
        torch.cuda.synchronize()

    print(f"task {task.id}  seq{a.seq} tr{a.tr} ts{a.ts} {a.opt}  iters={a.iters}  FILL={a.fill}  SETS={a.sets}", flush=True)
    thermal_soak(a.soak)

    # A) profiler-style: sync after every launch  -> ~40ms baseline
    torch.cuda.synchronize()
    syncbtw = []
    for i in range(150):
        ea = backend.record_event(stream)
        ex.launch(ctxs[i % len(ctxs)])
        eb = backend.record_event(stream)
        _check(cudart.cudaEventSynchronize(eb.raw))
        syncbtw.append(backend.event_time_us(eb) - backend.event_time_us(ea))
    print(f"  [sync-between] median {statistics.median(syncbtw)/1e3:.2f}ms", flush=True)

    # B) back-to-back: NO sync between launches -> sustained load
    torch.cuda.synchronize()
    evs = [backend.record_event(stream)]
    for i in range(a.iters):
        ex.launch(ctxs[i % len(ctxs)])
        evs.append(backend.record_event(stream))
    torch.cuda.synchronize()
    b2b = [backend.event_time_us(evs[i + 1]) - backend.event_time_us(evs[i])
           for i in range(a.iters)]
    print(f"  [back-to-back] first20 {statistics.mean(b2b[:20])/1e3:.2f}ms  "
          f"mid {statistics.mean(b2b[len(b2b)//2-50:len(b2b)//2+50])/1e3:.2f}ms  "
          f"last200 {statistics.mean(b2b[-200:])/1e3:.2f}ms", flush=True)

    json.dump({"cell": [a.seq, a.tr, a.ts, a.opt], "task": task.id, "iters": a.iters, "fill": a.fill, "sets": a.sets,
               "sync_between_us": syncbtw, "back_to_back_us": b2b}, open(a.out, "w"))
    print(f"wrote {a.out}")
    for b in local:
        backend.free(b)


if __name__ == "__main__":
    main()
