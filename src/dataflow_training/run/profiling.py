"""Profiling harness: measured runtimes + workspace, written back into programs.

The plan's measurement-over-estimation principle, mechanized:

- **Runtime**: each unique task signature `(compute_block_key, sorted io
  sizes)` is executed in isolation on a scratch stream, timed with CUDA
  events (warmup + median of repeats), AFTER a sustained thermal soak —
  a cold GPU measures tasks on transient boost clocks and under-prices
  them ~5-10% vs a training run at steady-state clocks (observed on the
  bs4/ga4 gap analysis before the soak existed). Mean/stdev/min/max ride
  along in the profile metadata for distribution visibility.
- **Payload realism**: inputs are seeded with real-scale random values
  (`fill_realistic`), because a task's runtime depends on operand VALUES and
  not only on shapes — near-zero operands barely switch the datapath, so the
  device draws far less power and holds a much higher clock than it ever can
  under real activations. This is the same failure mode as an unsoaked GPU
  and it is larger: measured on H100/llama3_8b, timing on zero-valued buffers
  under-priced the attention backward by 1.27x (9.11 ms vs 11.62 ms) and the
  whole compute-bound regime by ~1.25x. Composite operands (saved-activation
  contexts, packed weight layouts) carry NO element type and are the biggest
  buffers a block reads, so they are seeded as bf16; integer payloads stay
  deterministic and any discrete field is restored by the executable's own
  `profile_fill` hook.
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
    runtime_us: float          # median of repeats
    workspace_bytes: int
    repeats: int
    mean_us: float = 0.0
    stdev_us: float = 0.0
    min_us: float = 0.0
    max_us: float = 0.0


def _signature(task: TaskSpec, sizes: dict[str, int],
               resolver=None) -> tuple:
    """Cost-equivalence key. Sizes alone under-discriminate FROZEN
    plans: two trainable-field subsets with equal byte totals (wq vs
    wk) would share a timing while skipping DIFFERENT wgrad GEMMs. The
    freeze FINGERPRINT — the backward's trainable dW field names, read
    from the executable's own policy-filtered grad layout — separates
    them. It is EMPTY (and the signature byte-identical to the
    historical form) whenever nothing in the task's weight layout is
    frozen, so default-policy profile caches never invalidate."""
    fp: tuple = ()
    if resolver is not None and task.group == "backward" \
            and task.compute_block_key.endswith("_bwd"):
        try:
            ex = resolver(task)
            gl = ex.task_grad_layout(task)
            wl = ex.task_weight_layout(task)
            if len(gl.fields) < len(wl.fields):
                fp = tuple(f.name for f in gl.fields)
        except Exception:
            fp = ()
    base = (
        task.compute_block_key,
        tuple(sorted(sizes[i] for i in task.inputs)),
        tuple(sorted(o.size_bytes for o in task.outputs)),
        bool(task.mutates),
    )
    return base + ((fp,) if fp else ())


PROFILE_FILL_SEED = 20260724


def fill_realistic(buffer, size_bytes: int, dtype_name: str, generator) -> bool:
    """Seed one float input with real-scale random values. Returns whether it filled.

    Kernel time depends on operand VALUES, not only on shapes. Near-zero operands
    barely switch the datapath, so the device draws far less power and holds a much
    higher sustained clock than it can under real activations — profiling on
    uninitialized (effectively zero) buffers therefore under-prices every compute
    task. Measured on H100 / llama3_8b, flash attention backward: 9.11 ms on zeros
    vs 11.62 ms on N(0,1) — 484 W @ 1811 MHz vs 681 W @ 1448 MHz — and the N(0,1)
    figure matches the same kernel inside a real training step to 0.1%. Without
    this the whole cost model runs ~1.25x optimistic wherever compute is the
    critical path.

    INTEGER payloads are left to the caller's deterministic fill: they are indices
    (routing slots, positions, segment bounds) and garbage there is an illegal
    memory access, not a slower kernel.
    """
    import torch

    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view

    if dtype_name is None:
        # COMPOSITE payloads (saved-activation contexts, packed weight layouts)
        # declare no single element type — they are the largest operands a block
        # reads, so leaving them zero is exactly what mispriced the model. Seed
        # them as bf16, the element type of every float field in those layouts.
        # Any discrete field inside (MoE routing slots, sparse-attention indices)
        # is restored right after by the executable's own ``profile_fill`` hook,
        # which every family that makes discrete choices implements.
        dtype = torch.bfloat16
    else:
        dtype = TORCH_DTYPE_BY_NAME.get(dtype_name)
        if dtype is None or not dtype.is_floating_point:
            return False
    elem = torch.finfo(dtype).bits // 8
    n = size_bytes // elem
    if n == 0:
        return False
    view = torch_view(buffer, (n,), dtype)
    try:
        view.normal_(0.0, 1.0, generator=generator)
    except (RuntimeError, TypeError):
        # narrow float types (fp8) have no RNG kernel of their own — stage bf16
        stage = torch.empty(n, device=view.device, dtype=torch.bfloat16)
        view.copy_(stage.normal_(0.0, 1.0, generator=generator))
    return True


def thermal_soak(seconds: float = 1.0) -> None:
    # 1s default (was 10): with the PCIe contender on, profiling itself
    # keeps the die busy, so the soak only needs to lift clocks off idle
    # before the FIRST signature; validated by comparing per-signature
    # medians at soak=1 vs soak=10 (see commit).
    """Pull the GPU to sustained-load clocks before any timing: back-to-back
    large GEMMs with no host syncs in the loop. Without this, measurements
    ride the transient boost window and under-price real training steps."""
    import time

    import torch

    if seconds <= 0:
        return
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(200):
            a = a @ b
            a = a / a.norm().clamp_min(1e-3)  # keep values finite
        torch.cuda.synchronize()
    del a, b


class _PcieContender:
    """Keeps bidirectional PCIe DMA grinding while tasks are timed.

    Real training overlaps kernels with prefetch/offload traffic that
    competes for DRAM bandwidth; timing on an idle bus under-prices
    memory-bound kernels. Measured on bs4/ga4 @ 18 GiB (fused kernels):
    idle-bus profiling -> tasks +5..7% slower in-run; SATURATED bidi
    contention (this mode) -> tasks 3..6% FASTER in-run, i.e. the bound
    from the other side (the real bus duty cycle was ~34%/21%, not 100%).
    DEFAULT ON (locked 2026-07-06): between the two available biases,
    saturated contention is the better default — the error is smaller
    (+3..6% vs -5..-7% per task) and CONSERVATIVE (sim under-promises,
    real over-delivers), and the planner internalizes contention, which
    profiling showed is the direction reality rewards (recompute
    keeps winning at generous budgets BECAUSE it avoids unpriced
    contention). The unbiased fix remains duty-cycle-matched contention
    (2-pass: plan -> re-profile at the plan's duty cycle), not yet built.
    Scheduling fidelity is unaffected either way (replay gap ~0.4%)."""

    CHUNK = 256 * 1024 * 1024

    def __init__(self, backend) -> None:
        self.backend = backend
        self.h2d = backend.create_stream("h2d")
        self.d2h = backend.create_stream("d2h")
        self.pinned = backend.alloc("backing", self.CHUNK)
        self.dev_in = backend.alloc("fast", self.CHUNK)
        self.dev_out = backend.alloc("fast", self.CHUNK)
        self.chunk_us = self.CHUNK / (30e9 / 1e6)  # ~30 GB/s per direction

    def cover(self, expected_us: float) -> None:
        n = max(4, int(expected_us / self.chunk_us * 1.5) + 2)
        for _ in range(n):
            self.backend.memcpy_async(self.dev_in, self.pinned, self.CHUNK, self.h2d)
            self.backend.memcpy_async(self.pinned, self.dev_out, self.CHUNK, self.d2h)

    def close(self) -> None:
        import torch

        torch.cuda.synchronize()
        for b in (self.pinned, self.dev_in, self.dev_out):
            self.backend.free(b)


def profile_program(
    program: Program,
    resolver: Callable[[TaskSpec], object],
    backend,
    *,
    warmup: int = 2,
    repeats: int = 9,
    soak_seconds: float = 1.0,
    contend_pcie: bool = True,
    int32_fill: int = 0,
) -> dict[tuple, TaskProfile]:
    """Measure every unique task signature on the real device."""
    import torch

    from cuda.bindings import runtime as cudart

    from dataflow.runtime.device.cuda import _check
    from dataflow.runtime.executable import TaskContext
    from dataflow.runtime.interop import torch_view

    sizes = program.object_sizes()
    metas = {o.id: o.tensor for o in program.initial_objects}
    for t in program.tasks:
        for o in t.outputs:
            metas[o.id] = o.tensor

    thermal_soak(soak_seconds)
    # one seeded generator for every payload fill: costs must be reproducible
    # across cache refreshes (a re-profile that shifts task costs re-plans)
    fill_gen = torch.Generator(device="cuda")
    fill_gen.manual_seed(PROFILE_FILL_SEED)
    stream = backend.create_stream("compute")
    contender = _PcieContender(backend) if contend_pcie else None
    profiles: dict[tuple, TaskProfile] = {}

    # attention blocks resolve the round's Segments workload-side
    # (resolve_segments); the profiler drives executables directly, so build
    # + materialize a uniform descriptor once here (from any block
    # executable's dims — all tasks share dims)
    _run_args = None
    for _t in program.tasks:
        _d = getattr(resolver(_t), "dims", None)
        if _d is not None:
            from ..data.segments import uniform_segments

            segs = uniform_segments(_d, program)
            if getattr(backend, "physical", False):
                one = next(iter(segs.values())).on(
                    f"cuda:{backend.device}")
                segs = {r: one for r in segs}
            _run_args = {"segments": segs}
            break

    for task in program.tasks:
        sig = _signature(task, sizes, resolver)
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
                else:
                    # every other payload carries REAL-SCALE values: operand
                    # content drives switching activity -> power -> sustained
                    # clock, so zero-valued inputs time a machine that never
                    # exists in a real step (see fill_realistic). Objects with
                    # no TensorMeta at all are the composite ones (A contexts,
                    # packed weights) and are the BIGGEST operands a block reads.
                    fill_realistic(b, sizes[obj], getattr(meta, "dtype", None),
                                   fill_gen)
                in_buffers[obj] = b
            out_buffers = {o.id: buf(o.size_bytes) for o in task.outputs}
            mut_buffers = {m: in_buffers[m] for m in task.mutates}
            executable = resolver(task)
            ctx = TaskContext(
                task=task, stream=stream, inputs=in_buffers, outputs=out_buffers,
                mutates=mut_buffers, backend=backend, run_args=_run_args,
            )

            # Executables may declare a deterministic buffer-seeding hook
            # (MoE blocks do): valid routing indices in packed contexts —
            # garbage int32 ctx fields are an illegal memory access in the
            # gathers — plus seeded float fills so data-dependent routing
            # costs are near-balanced and REPRODUCIBLE across cache
            # refreshes (uninitialized logits route everything to K experts,
            # an anti-conservative distribution-dependent bias).
            fill = getattr(executable, "profile_fill", None)
            if fill is not None:
                fill(ctx)
                torch.cuda.synchronize()

            # workspace: allocator peak delta around one launch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            executable.launch(ctx)
            torch.cuda.synchronize()
            workspace = max(0, torch.cuda.max_memory_allocated() - base)

            if contender is not None:
                contender.cover(float(task.runtime_us) * (warmup + repeats))
            times = []
            for i in range(warmup + repeats):
                a = backend.record_event(stream)
                executable.launch(ctx)
                b = backend.record_event(stream)
                _check(cudart.cudaEventSynchronize(b.raw))
                if i >= warmup:
                    times.append(backend.event_time_us(b) - backend.event_time_us(a))
            profiles[sig] = TaskProfile(
                runtime_us=statistics.median(times),
                workspace_bytes=workspace,
                repeats=repeats,
                mean_us=statistics.fmean(times),
                stdev_us=statistics.stdev(times) if len(times) > 1 else 0.0,
                min_us=min(times),
                max_us=max(times),
            )
        finally:
            for b in local:
                backend.free(b)
    if contender is not None:
        contender.close()
    return profiles


def apply_measured_costs(program: Program, profiles: dict[tuple, TaskProfile],
                         resolver=None) -> Program:
    sizes = program.object_sizes()
    new_tasks = []
    for task in program.tasks:
        p = profiles[_signature(task, sizes, resolver)]
        new_tasks.append(replace(
            task,
            runtime_us=p.runtime_us,
            metadata={
                **task.metadata,
                "measured": {
                    "runtime_us": p.runtime_us,
                    "workspace_bytes": p.workspace_bytes,
                    "repeats": p.repeats,
                    "mean_us": p.mean_us,
                    "stdev_us": p.stdev_us,
                    "min_us": p.min_us,
                    "max_us": p.max_us,
                    "estimate_runtime_us": task.runtime_us,
                },
            },
        ))
    return replace(program, tasks=tuple(new_tasks))


# bump when task-internals change measured behavior (runtime or workspace):
# the cache key cannot see code, so this is the manual invalidation lever.
# rev 2: BlockRecompute stops at w1/w3 (down-proj/swiglu/y removed).
def host_backing_cap_bytes(*, reserve_gib: float = 10.0) -> int:
    """Planning cap for pinned-host backing, derived from the host's
    CURRENTLY AVAILABLE memory (MemAvailable) minus a flat leeway
    (default 10 GiB) for the OS, torch host buffers, and profiling
    scratch.

    This is a PLANNING bound only: it keeps PressureFit from emitting
    plans whose offload footprint could never be pinned (which would
    otherwise fail at run time, mid-pin). The runtime itself pins by
    plan DEMAND (pool prewarm) when program.backing_memory_capacity is
    None — callers should plan WITH this cap and execute with the
    capacity stripped, because a set capacity makes the engine pin the
    FULL capacity as one up-front slab (engine.add_slab)."""
    import os

    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if avail is None:  # non-Linux fallback
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return int(max(0, avail - reserve_gib * 1024 ** 3))


PROFILE_CACHE_REV = "4"  # bump whenever kernel/task costs change (invalidates all cached profiles)
#   "3": float inputs seeded with real-scale values (fill_realistic) — every
#        earlier profile timed zero-valued operands and ran ~1.25x optimistic
#   "4": that seeding extended to COMPOSITE operands (saved-activation contexts,
#        packed weights carry no element type); rev 3 left them zero, which is
#        most of the bytes a block reads


def load_or_profile(
    program: Program,
    resolver,
    backend,
    *,
    cache_dir=None,
    kernel_set: dict[str, str] | None = None,
    refresh: bool = False,
    **kwargs,
) -> dict[tuple, TaskProfile]:
    """Disk-cached profile_program.

    Costs are measurements of a specific (task signatures, kernel set,
    profiling environment, device) — the cache key covers all four, so a
    kernel swap or a contended-mode toggle re-measures instead of silently
    reusing stale numbers. One cache hit skips soak + all timing: startup
    becomes cheap for every repeat run of the same config.
    """
    import hashlib
    import json
    from pathlib import Path

    import torch

    sizes = program.object_sizes()
    signatures = sorted({repr(_signature(t, sizes, resolver)) for t in program.tasks})
    if kernel_set is None and hasattr(resolver, "kernel_set"):
        kernel_set = resolver.kernel_set.describe()
    env = {
        "signatures": signatures,
        "kernel_set": kernel_set or {},
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "soak_seconds": kwargs.get("soak_seconds", 1.0),
        "contend_pcie": kwargs.get("contend_pcie", True),
        "repeats": kwargs.get("repeats", 9),
        "torch": torch.__version__,
        "code_rev": PROFILE_CACHE_REV,
    }
    key = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir) if cache_dir is not None else Path("artifacts/profile-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"profiles-{key}.json"

    if path.exists() and not refresh:
        raw = json.loads(path.read_text())
        print(f"profile cache HIT {path.name} ({len(raw['profiles'])} signatures)")
        return {eval(k): TaskProfile(**v) for k, v in raw["profiles"].items()}

    profiles = profile_program(program, resolver, backend, **kwargs)
    path.write_text(json.dumps({
        "env": env,
        "profiles": {repr(k): vars(v) for k, v in profiles.items()},
    }, indent=2) + "\n")
    print(f"profile cache MISS -> wrote {path.name}")
    return profiles


def cached_pcie(backend, *, cache_dir=None, refresh: bool = False):
    """Disk-cached backend.measure_pcie(): bandwidths are device properties,
    and re-measuring per invocation makes plans non-reproducible (a few
    percent of measurement noise is enough to tip the recompute planner to a
    different variant, which changes lifetimes, packing, and even placement
    feasibility). Pin them once; --refresh to re-measure."""
    import json
    from pathlib import Path

    import torch

    cache_dir = Path(cache_dir) if cache_dir is not None else Path("artifacts/profile-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = torch.cuda.get_device_name().replace(" ", "-") if torch.cuda.is_available() else "cpu"
    path = cache_dir / f"pcie-{name}.json"
    if path.exists() and not refresh:
        d = json.loads(path.read_text())
        print(f"pcie cache HIT {path.name}: "
              f"bidi {d['bidi_h2d'] / 1e3:.1f}/{d['bidi_d2h'] / 1e3:.1f} GB/s")
        from types import SimpleNamespace

        return SimpleNamespace(**d)
    pcie = backend.measure_pcie()
    path.write_text(json.dumps(pcie.__dict__, indent=2) + "\n")
    print(f"pcie cache MISS -> wrote {path.name}")
    return pcie
