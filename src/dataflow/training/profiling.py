"""Profiling harness: measured runtimes + workspace, written back into programs.

The plan's measurement-over-estimation principle, mechanized:

- **Runtime**: each unique task signature `(compute_block_key, sorted io
  sizes)` is executed in isolation on a scratch stream, timed with CUDA
  events (warmup + median of repeats).
- **Workspace**: the torch caching-allocator peak delta around one launch —
  exactly the scratch-lane bytes the executable used beyond runtime-owned
  buffers (runtime buffers come from our pool, invisible to torch's
  allocator, so the delta isolates op-internal scratch).

`apply_measured_costs` returns a program with measured `runtime_us` per task
and metadata `{"measured": {"runtime_us", "workspace_bytes", ...}}`; re-plan
it with `plan_program` before headline runs (final planning on measured
costs).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from typing import Callable

from dataflow.core import Program, TaskSpec


@dataclass(frozen=True)
class TaskProfile:
    runtime_us: float
    workspace_bytes: int
    repeats: int


def _signature(task: TaskSpec, sizes: dict[str, int]) -> tuple:
    return (
        task.compute_block_key,
        tuple(sorted(sizes[i] for i in task.inputs)),
        tuple(sorted(o.size_bytes for o in task.outputs)),
        bool(task.mutates),
    )


def profile_program(
    program: Program,
    resolver: Callable[[TaskSpec], object],
    backend,
    *,
    warmup: int = 2,
    repeats: int = 5,
    int32_fill: int = 0,
) -> dict[tuple, TaskProfile]:
    """Measure every unique task signature on the real device."""
    import torch

    from cuda.bindings import runtime as cudart

    from dataflow.runtime.device.cuda import _check
    from dataflow.runtime.executable import TaskContext
    from dataflow.tasks.interop import torch_view

    sizes = program.object_sizes()
    metas = {o.id: o.tensor for o in program.initial_objects}
    for t in program.tasks:
        for o in t.outputs:
            metas[o.id] = o.tensor

    stream = backend.create_stream("compute")
    profiles: dict[tuple, TaskProfile] = {}

    for task in program.tasks:
        sig = _signature(task, sizes)
        if sig in profiles:
            continue
        # Distinct buffers per role slot, allocated for THIS signature and
        # freed right after: caching across signatures accumulated more than
        # VRAM once grad-accum variants and batched sizes appeared.
        local: list = []

        def buf(size: int):
            b = backend.alloc("fast", size)
            local.append(b)
            return b

        try:
            in_buffers = {}
            for obj in task.inputs:
                b = buf(sizes[obj])
                meta = metas.get(obj)
                if meta is not None and meta.dtype == "int32":
                    torch_view(b, (sizes[obj] // 4,), torch.int32).fill_(int32_fill)
                in_buffers[obj] = b
            out_buffers = {o.id: buf(o.size_bytes) for o in task.outputs}
            mut_buffers = {m: in_buffers[m] for m in task.mutates}
            ctx = TaskContext(
                task=task, stream=stream, inputs=in_buffers, outputs=out_buffers,
                mutates=mut_buffers, backend=backend,
            )
            executable = resolver(task)

            # workspace: allocator peak delta around one launch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            executable.launch(ctx)
            torch.cuda.synchronize()
            workspace = max(0, torch.cuda.max_memory_allocated() - base)

            times = []
            for i in range(warmup + repeats):
                a = backend.record_event(stream)
                executable.launch(ctx)
                b = backend.record_event(stream)
                _check(cudart.cudaEventSynchronize(b.raw))
                if i >= warmup:
                    times.append(backend.event_time_us(b) - backend.event_time_us(a))
            profiles[sig] = TaskProfile(
                runtime_us=statistics.median(times), workspace_bytes=workspace, repeats=repeats,
            )
        finally:
            for b in local:
                backend.free(b)
    return profiles


def apply_measured_costs(program: Program, profiles: dict[tuple, TaskProfile]) -> Program:
    sizes = program.object_sizes()
    new_tasks = []
    for task in program.tasks:
        p = profiles[_signature(task, sizes)]
        new_tasks.append(replace(
            task,
            runtime_us=p.runtime_us,
            metadata={
                **task.metadata,
                "measured": {
                    "runtime_us": p.runtime_us,
                    "workspace_bytes": p.workspace_bytes,
                    "repeats": p.repeats,
                    "estimate_runtime_us": task.runtime_us,
                },
            },
        ))
    return replace(program, tasks=tuple(new_tasks))
