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
  `profile_fill` hook. WEIGHT objects then get a second, per-FIELD pass with
  the init PRODUCTION uses (`fill_weight_fields`: N(0, 0.02) matrices,
  `*_norm_w` ones), because "real-scale" is a per-field property and a
  blanket N(0,1) is ~50x off for weights: it widened the head's logits from
  std 1.28 to 63.3, saturated its CE softmax, left 99.9% of dlogits exactly
  zero, and under-priced head_loss ~20% on H100 (155 ms profiled vs 187 ms in
  a real step) — a near-idle datapath holding a clock the run never sees.
- **Workspace**: the torch caching-allocator peak delta around one launch —
  exactly the scratch-lane bytes the executable used beyond runtime-owned
  buffers (runtime buffers come from our pool, invisible to torch's
  allocator, so the delta isolates op-internal scratch).

`apply_measured_costs` returns a program with measured `runtime_us` per task
and metadata `{"measured": {"runtime_us", "workspace_bytes", ...}}`; re-plan
it with `plan_program` before headline runs (final planning on measured
costs).

MEMORY LIFECYCLE CONTRACT (device):
- The profiler is IN-PROCESS and engine-free: it drives task
  executables directly. Operand buffers are raw per-signature
  backend.alloc/free (exact fit, no pool — there is no plan or budget
  to pool against yet); kernel-internal scratch goes through torch's
  caching allocator per the task contract.
- All profile passes share ONE process-lifetime stream trio
  (dataflow.runtime.streams), so torch-cached scratch REUSES across
  signatures and tables instead of stranding per pass.
- A table's scratch dies with the table: profile_program returns
  torch's cache to the driver on exit (cache-hit calls never touch
  the device at all).
- Raw operand allocation self-heals once on DeviceMemoryError
  (synchronize + empty torch cache + retry): raw cudaMalloc cannot
  see torch's cached-idle blocks, and torch only self-heals its own
  allocations.
- Costs are disk-cached PER SIGNATURE per (kernel set, profiling
  environment, device); PROFILE_CACHE_REV is the manual invalidation
  lever. A signature measured once is never re-measured — across
  budgets, t_step variants, optimizers, and runs.
"""
from __future__ import annotations

import os as _os
import statistics
from dataclasses import dataclass, replace
from typing import Callable

from dataflow.core import Program, TaskSpec


@dataclass(frozen=True)
class TaskProfile:
    runtime_us: float          # median of repeats
    workspace_bytes: int
    repeats: int
    # total DEVICE time this signature was sampled over. A profile is only
    # as good as the load it was taken under, so the number rides along.
    sampled_us: float = 0.0
    # the sampling floor this profile was TAKEN UNDER, which is what makes it
    # usable as a production price. Checking sampled_us instead would punish
    # fast signatures: they exit on MAX_BRACKETS before the budget, so a
    # legitimate production profile can show less elapsed time than the floor.
    # -1.0 means "written before this field existed" and is never refused.
    sample_floor_s: float = -1.0
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
    frozen, so default-policy profile caches never invalidate.

    Sizes also under-discriminate GEOMETRY that leaves buffer bytes unchanged.
    A round of T tokens occupies the same buffers whether it is one long
    sequence or many short ones, so batch x seq_len combinations with equal
    token counts collide — while attention costs scale with the sequence, not
    the token total (8x the attention work at seq 8192 vs 1024 for the same T).
    The lowering declares such geometry in the task's ``cost_key`` and this
    reads whatever is there, so a new kind of cost-relevant geometry starts
    separating profiles without this function learning about it. Tasks whose
    cost does not depend on any leave it absent.

    Both extra components are read from the TASK, never from the resolver: a
    signature has to be a pure function of the task and its sizes, or the same
    task keys differently depending on who is asking. ``profile_program`` passes
    a resolver and ``apply_measured_costs`` does not, so a resolver-derived
    component turns every later lookup into a miss."""
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
    cost_key = (task.metadata or {}).get("cost_key") or {}
    geometry = tuple(sorted(cost_key.items()))
    base = (
        task.compute_block_key,
        tuple(sorted(sizes[i] for i in task.inputs)),
        tuple(sorted(o.size_bytes for o in task.outputs)),
        bool(task.mutates),
    )
    return base + ((fp,) if fp else ()) + (geometry if geometry else ())


PROFILE_FILL_SEED = 20260724


def fill_weight_fields(buffer, layout, generator) -> None:
    """Production per-FIELD weight init (lowering/emit) applied to a
    profiling buffer — re-exported here so the profile loop states its
    dependency on the init policy explicitly."""
    from dataflow_training.lowering.emit import fill_weight_fields as impl

    impl(buffer, layout, generator)


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


# one timed bracket aims for this much device time, so a sub-ms task
# amortizes its host sync over many back-to-back launches and a 170 ms
# task still gets its own bracket
BRACKET_US = 50_000.0
# the inner cap only binds for VERY short tasks, and when it binds the
# bracket falls short of BRACKET_US, so the budget runs out of brackets
# before it runs out of time: an 8.5 us prologue capped at 512 launches
# made 4.35 ms brackets and sampled 0.88 s against a 2.5 s ask. Sized so
# a task down to ~12 us still fills a bracket.
MAX_INNER = 4096
MAX_BRACKETS = 400


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


def profiling_streams(backend):
    """The profiler's process-lifetime stream trio (see
    dataflow.runtime.streams for the stream-lifecycle rule this
    follows: cached kernel scratch is stream-keyed, so profile passes
    must share streams or strand it)."""
    from dataflow.runtime.streams import shared_streams

    return shared_streams(backend, "profiling")


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
        streams = profiling_streams(backend)
        self.h2d = streams[1]
        self.d2h = streams[2]
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


# Sampling floor for PRODUCTION pricing. A task measured in a short burst
# reads faster than the same task under sustained load — block_fwd times
# 22.16 ms in a burst against 23.56 ms sustained, and the sustained figure is
# the one production reproduces (23.06-23.27 ms). Clock and power settle over
# roughly a second, so a price that feeds real planning samples for at least
# this long per signature.
#
# It is NOT the default, because it costs 2.5 s x every unique signature and
# most callers (tests, docs, probes) want a cost table, not a price: paying it
# everywhere added ~9 min to the suite in two files that build cold tables.
# Production pricing passes it EXPLICITLY — see driver.measured_pricing_inputs
# and measured_grouped_program.
PRODUCTION_SAMPLE_SECONDS = float(_os.environ.get(
    "DATAFLOW_SAMPLE_FLOOR_S", "2.5"))
# ...overridable so the floor's WORTH can be measured rather than assumed:
# profile a cell set at 2.5 / 1.0 / 0.0, plan from each, and compare both the
# plans chosen and the sim-vs-real agreement. The floor is in the profile
# cache key, so each setting keeps its own cache and cannot contaminate
# another. Leave it unset for anything that ships a number.

# The cheap default: stop as soon as `repeats` brackets are in hand. Callers
# that need a defensible PRICE opt in to PRODUCTION_SAMPLE_SECONDS above.
DEFAULT_SAMPLE_SECONDS = 0.0


def profile_program(
    program: Program,
    resolver: Callable[[TaskSpec], object],
    backend,
    *,
    warmup: int = 2,
    repeats: int = 9,
    min_sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    soak_seconds: float = 1.0,
    contend_pcie: bool = True,
    int32_fill: int = 0,
    have: dict[tuple, TaskProfile] | None = None,
) -> dict[tuple, TaskProfile]:
    """Measure every unique task signature on the real device.

    ``have`` carries signatures already measured under this same environment;
    they are returned untouched and never re-run. A signature IS the
    cost-equivalence key, so reusing one across programs is the reuse this
    already does within a program — it just stops two programs that share most
    of their work (the same model under a different optimizer, say) from paying
    for the shared part twice."""
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

    profiles: dict[tuple, TaskProfile] = dict(have or {})
    if all(_signature(t, sizes, resolver) in profiles for t in program.tasks):
        return profiles          # nothing new: no soak, no device work at all

    thermal_soak(soak_seconds)
    # one seeded generator for every payload fill: costs must be reproducible
    # across cache refreshes (a re-profile that shifts task costs re-plans)
    fill_gen = torch.Generator(device="cuda")
    fill_gen.manual_seed(PROFILE_FILL_SEED)
    stream = profiling_streams(backend)[0]
    contender = _PcieContender(backend) if contend_pcie else None

    # attention blocks resolve the round's Segments workload-side
    # (resolve_segments); the profiler drives executables directly, so build
    # + materialize a uniform descriptor once here (from any block
    # executable's dims — all tasks share dims)
    _run_args = None
    _run_values = None
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
            # production publishes each round's CONTENT token count via
            # the prologue into run_values; profiled launches must read
            # the SAME channel with the SAME numbers (uniform segments
            # = full rounds), not the max_tokens fallback — for full
            # rounds the value is identical, but the channel divergence
            # would silently overprice num_tokens consumers on any
            # under-filled round profiled in the future
            _run_values = {"num_tokens_by_round":
                           {str(r): _d.max_tokens for r in segs}}
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
            from dataflow.runtime.device.cuda import DeviceMemoryError

            try:
                b = backend.alloc("fast", size)
            except DeviceMemoryError:
                # raw cudaMalloc cannot see torch's cached-but-idle
                # blocks; give the headroom back once and retry
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(f"  [profile] freed torch cache to fit a "
                      f"{size / 2 ** 30:.1f} GiB operand", flush=True)
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

            # WEIGHT objects are then re-seeded per FIELD with the init
            # PRODUCTION uses (N(0, 0.02) matrices, *_norm_w ones): the
            # blanket N(0,1) above is right for activations but wrong for
            # weights by ~50x, and operand scale decides cost. It made
            # the head's softmax saturate, zeroed 99.9% of its dlogits,
            # and under-priced it 20% against a real step (see
            # _Base.profile_weight_layouts). Fixed seed: profiles must be
            # reproducible across cache refreshes.
            layouts = getattr(executable, "profile_weight_layouts", None)
            if layouts is not None:
                weight_gen = torch.Generator().manual_seed(0)
                for oid, layout in layouts(task).items():
                    wbuf = in_buffers.get(oid)
                    if wbuf is None:
                        continue
                    extent = max((f.offset_bytes + f.nbytes)
                                 for f in layout.fields)
                    if extent > wbuf.size_bytes:
                        # A FAMILY-NEUTRAL executable (the shared optimizer
                        # step) inherits the GENERIC weight layout, which
                        # does not describe every family's packing: gpt2
                        # fuses qkv and carries biases, so the inherited
                        # llama-shaped layout addresses 18.9 MB of a 14.2 MB
                        # object. The program's object size is the truth
                        # (packed layouts = size truth), so a layout that
                        # does not fit is the wrong description of this
                        # buffer — writing it would run off the end. The
                        # generic real-scale fill above stands for it; the
                        # optimizer measures 0.0% fill-sensitive anyway
                        # (task_values_probe).
                        continue
                    fill_weight_fields(wbuf, layout, weight_gen)
            ctx = TaskContext(
                task=task, stream=stream, inputs=in_buffers, outputs=out_buffers,
                mutates=mut_buffers, backend=backend, run_args=_run_args,
                run_values=_run_values,
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
                # initial guess only — topped up per repeat below from
                # MEASURED durations, so neither the plan's estimate nor
                # the contender's assumed drain rate can starve the bus
                # mid-bench (an exhausted contender made a ~187 ms head
                # measure ~155: the tail repeats ran on an idle bus)
                contender.cover(float(task.runtime_us) * (warmup + 2))
            # SAMPLE FOR A DURATION, not a count, and launch
            # BACK-TO-BACK inside each timed bracket.
            #
            # A fixed repeat count benches a short task in a burst the
            # real step never runs in: 11 launches of a 23 ms block is
            # 0.25 s of load and the die holds boost clocks throughout.
            # Measured on H100, block_fwd reads 22.16 ms that way against
            # 23.56 ms under sustained load — and only the sustained
            # figure matches what the engine does in a step (23.06-23.27
            # ms). The one-shot thermal_soak lifts clocks before the
            # FIRST signature and cannot hold them down through every
            # later one; the PCIe contender is DMA, not compute (it moves
            # the head 0.2%). So each signature generates its own load.
            #
            # Syncing per launch would defeat that (the die idles in every
            # gap) and would cost thousands of syncs on sub-ms tasks, so
            # each bracket runs `inner` launches with no host sync inside
            # and divides. Profiling is cheap next to planning and
            # measurement, which is what buys the extra seconds.
            for _ in range(warmup):
                a = backend.record_event(stream)
                executable.launch(ctx)
                b = backend.record_event(stream)
                _check(cudart.cudaEventSynchronize(b.raw))
            one_us = max(backend.event_time_us(b) - backend.event_time_us(a),
                         1.0)
            inner = max(1, min(MAX_INNER, int(BRACKET_US / one_us)))
            times = []
            elapsed_us = 0.0
            budget_us = float(min_sample_seconds) * 1e6
            for _ in range(MAX_BRACKETS):
                a = backend.record_event(stream)
                for _ in range(inner):
                    executable.launch(ctx)
                b = backend.record_event(stream)
                _check(cudart.cudaEventSynchronize(b.raw))
                span = backend.event_time_us(b) - backend.event_time_us(a)
                times.append(span / inner)
                elapsed_us += span
                if contender is not None:
                    contender.cover(span * 2.0)
                if len(times) >= repeats and elapsed_us >= budget_us:
                    break
            profiles[sig] = TaskProfile(
                runtime_us=statistics.median(times),
                workspace_bytes=workspace,
                repeats=len(times),          # ACTUAL brackets, not the ask
                sampled_us=elapsed_us,       # how long this signature ran
                sample_floor_s=float(min_sample_seconds),
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
    # a table's scratch dies with the table: hand torch's cached kernel
    # workspaces back to the driver so every table STARTS from ~zero
    # reserved (shared streams already make scratch reusable WITHIN the
    # table; this bounds what outlives it). Cache-hit calls never reach
    # here — they return before any device work.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return profiles


def recompute_level_pins(program) -> list[dict[str, int]]:
    """One level assignment per distinct recompute level in the rewrite table.

    The recompute search prices programs the base lowering never contains. A
    block that recomputes emits a task the base program has no equivalent of,
    and its forward stops emitting the saved-activation object — which changes
    that forward's cost signature too. Costs are looked up BY signature, so a
    variant built from levels nobody profiled cannot be priced at all, and the
    search loses it.

    Pinning every rewrite to each level in turn produces the programs whose
    signatures the search can encounter: a mixed assignment only ever combines
    tasks one of these pins already contains. Nothing here reads what a level
    means — the rewrite table is the whole input."""
    levels = sorted({o.level for rw in program.recompute_rewrites for o in rw.options})
    return [{rw.object_id: min(level, rw.options[-1].level)
             for rw in program.recompute_rewrites}
            for level in levels]


def measured_profile_table(fam, cfg, resolver, backend, *, recompute: bool = True,
                           refresh: bool = False,
                           **kwargs) -> dict[tuple, TaskProfile]:
    """The cost table for ``cfg``, covering every variant that will be priced.

    Profiling only the base lowering leaves the recompute search unable to
    price any variant it evaluates, which it cannot distinguish from a variant
    that does not fit — so it quietly keeps the save-everything plan and calls
    the rest infeasible. Every caller that plans with measured costs needs the
    same coverage, so they share this rather than each assembling a table:
    two copies of that assembly is how the gap survived being fixed once."""
    bare = fam.lower(cfg)
    # ``refresh`` applies to the FIRST lowering only: it empties the store
    # once, and every variant after it re-measures exactly what the emptied
    # store no longer covers. Passing it to every call would leave the file
    # holding only the LAST lowering's signatures (each refresh restarts
    # the store), losing the base program's costs.
    profiles = load_or_profile(bare, resolver, backend, refresh=refresh,
                               **kwargs)
    if recompute:
        for pins in recompute_level_pins(bare):
            profiles = load_or_profile(fam.lower(cfg, recompute_levels=pins),
                                       resolver, backend, **kwargs)
    return profiles


def measured_program(fam, cfg, profiles, resolver, pcie, levels=None) -> Program:
    """One program priced entirely by measurement.

    Measurement covers two things, and using one without the other silently
    biases the plan: profiled task costs say how long compute takes, and the
    box's own link bandwidths say how long the transfers feeding it take. A
    program given measured compute but the lowering's default bandwidths
    prices its offloads against a link the machine does not have, and every
    plan built from it comes out optimistic by however far that default sits
    from the truth. Both callers take their programs from here so neither can
    apply half the measurement."""
    program = (fam.lower(cfg) if levels is None
               else fam.lower(cfg, recompute_levels=levels))
    return replace(apply_measured_costs(
                       program, profiles, resolver,
                       require_sample_seconds=PRODUCTION_SAMPLE_SECONDS),
                   bandwidth_from_slow=pcie.bidi_h2d,
                   bandwidth_to_slow=pcie.bidi_d2h)


def measured_grouped_program(cfg, dp_group, resolver, pcie, backend,
                             *, require_cached: bool,
                             levels=None, parallel=None,
                             zero1rs_world=None) -> Program:
    """Grouped twin of measured_program: the SAME both-halves contract
    (profiled task costs + this box's measured link bandwidths) over a
    lower_with_group lowering — the single production pricing path for
    any world size. Coverage is ENSURED here, not assumed: the grouped
    lowering runs through load_or_profile, so shard/tp tasks whose
    cost signatures the store never measured are measured NOW when the
    caller warmed explicitly (require_cached=False), and refuse loudly
    otherwise. Nothing is ever priced by estimate."""
    from dataflow_training.distributed.grouped_lowering import (
        lower_with_group)

    program = lower_with_group(cfg, dp_group, recompute_levels=levels,
                               parallel=parallel,
                               zero1rs_world=zero1rs_world)
    profiles = load_or_profile(
        program, resolver, backend, require_cached=require_cached,
        min_sample_seconds=PRODUCTION_SAMPLE_SECONDS)
    return replace(apply_measured_costs(
                       program, profiles, resolver,
                       require_sample_seconds=PRODUCTION_SAMPLE_SECONDS),
                   bandwidth_from_slow=pcie.bidi_h2d,
                   bandwidth_to_slow=pcie.bidi_d2h)


class UnderSampledProfileError(ValueError):
    """A production price was built from profiles taken under too little load.

    Distinct from MissingProfileError (nothing measured this signature at
    all): the measurement exists, it is just not defensible as a price.
    """


class MissingProfileError(LookupError):
    """A task was priced against a profile table that never measured it.

    Distinct from the planner's own ValueError refusals, which mean "this
    program does not fit" and are a legitimate result. This means the profile
    table is incomplete — a fault in what was measured, not in the plan — and
    callers that treat refusals as data must not absorb it as one.
    """


def apply_measured_costs(program: Program, profiles: dict[tuple, TaskProfile],
                         resolver=None, *,
                         require_sample_seconds: float = 0.0) -> Program:
    """Price ``program`` from measured profiles.

    ``require_sample_seconds`` is the production guard: a caller that plans
    real work passes PRODUCTION_SAMPLE_SECONDS, and any profile taken under
    less load than that is refused by name. Sampling floor is opt-in (most
    callers want a cost table, not a price), so without this a pricing path
    that forgot to ask for it would quietly plan on burst-timed numbers that
    read several percent fast and look entirely plausible. Same doctrine as
    MissingProfileError: a fault in what was MEASURED fails loudly rather
    than being absorbed as a result.
    """
    sizes = program.object_sizes()
    new_tasks = []
    for task in program.tasks:
        sig = _signature(task, sizes, resolver)
        if sig not in profiles:
            raise MissingProfileError(
                f"task {task.id!r} has no measured cost: no entry matches "
                f"{sig}. The profile table was built from a different program "
                f"-- every variant that gets priced has to be profiled.")
        p = profiles[sig]
        if 0.0 <= p.sample_floor_s < require_sample_seconds:
            raise UnderSampledProfileError(
                f"task {task.id!r} is priced from a profile taken under a "
                f"{p.sample_floor_s:.2f} s sampling floor, under the "
                f"{require_sample_seconds:.2f} s production floor "
                f"(it ran {p.sampled_us / 1e6:.2f} s). "
                f"A signature timed in a short burst reads "
                f"faster than the same work under sustained load, so this "
                f"price is optimistic. Profile through the production path "
                f"(min_sample_seconds=PRODUCTION_SAMPLE_SECONDS) or drop "
                f"require_sample_seconds if this caller only needs a cost "
                f"table.")
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


PROFILE_CACHE_REV = "7"  # manual override; impl_fingerprint() below
# auto-invalidates on ANY kernel/block source change (the muon bf16
# migration changed optimizer cost 11x without touching an impl_id or
# this rev — a stale 414 ms profile priced a 37 ms task until the
# plan-vs-measured check caught it)


def impl_fingerprint() -> str:
    """Hash of every kernel/block implementation source file. Costs are
    measurements OF THIS CODE: any change re-measures (profiling is
    cheap by doctrine; silent staleness is not). Over-invalidation on
    comment edits is accepted for that guarantee."""
    import hashlib
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for sub in ("kernels", "blocks"):
        for f in sorted((base / sub).rglob("*.py")):
            h.update(str(f.relative_to(base)).encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
    # the measurement code itself shapes the numbers (the contender
    # starvation fix changed a head profile 20%): fingerprint this file
    h.update(hashlib.sha256(Path(__file__).resolve().read_bytes()).digest())
    return h.hexdigest()[:16]
#   "3": float inputs seeded with real-scale values (fill_realistic) — every
#        earlier profile timed zero-valued operands and ran ~1.25x optimistic
#   "4": that seeding extended to COMPOSITE operands (saved-activation contexts,
#        packed weights carry no element type); rev 3 left them zero, which is
#        most of the bytes a block reads
#   "5": attention-bearing tasks carry their sequence length, so batch x seq
#        combinations with equal token counts stop sharing one timing
#   "6": that geometry moved into the task's cost_key, read from the task
#        rather than through a resolver


def load_or_profile(
    program: Program,
    resolver,
    backend,
    *,
    cache_dir=None,
    kernel_set: dict[str, str] | None = None,
    refresh: bool = False,
    require_cached: bool = False,
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
    import os
    from pathlib import Path

    import torch

    sizes = program.object_sizes()
    if kernel_set is None and hasattr(resolver, "kernel_set"):
        kernel_set = resolver.kernel_set.describe()
    # The key describes the MEASUREMENT ENVIRONMENT, not this program. Costs
    # are stored per signature, so two programs that share work share its
    # measurements instead of each paying in full: the same model under a
    # second optimizer differs only in its optimizer tasks, yet keying the file
    # by the whole signature SET made it re-measure every block and head task
    # as well.
    env = {
        "kernel_set": kernel_set or {},
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "soak_seconds": kwargs.get("soak_seconds", 1.0),
        "min_sample_seconds": kwargs.get("min_sample_seconds",
                                 DEFAULT_SAMPLE_SECONDS),
        "contend_pcie": kwargs.get("contend_pcie", True),
        "repeats": kwargs.get("repeats", 9),
        "torch": torch.__version__,
        "code_rev": PROFILE_CACHE_REV,
        "impl": impl_fingerprint(),
    }
    key = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir) if cache_dir is not None else Path("artifacts/profile-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"profiles-{key}.json"

    store: dict[tuple, TaskProfile] = {}
    if path.exists() and not refresh:
        raw = json.loads(path.read_text())
        store = {eval(k): TaskProfile(**v) for k, v in raw["profiles"].items()}
    wanted = {_signature(t, sizes, resolver) for t in program.tasks}
    missing = wanted - set(store)
    if missing and require_cached:
        # The caller declared itself measurement-free (a predict stage on a
        # warm cache, possibly on a box with no GPU time to give). Missing
        # signatures then mean the profile stage did not cover this program
        # — profiling here silently would put GPU work where the caller
        # promised none, so refuse with the remedy instead.
        raise RuntimeError(
            f"profile cache {path.name} is missing {len(missing)} of "
            f"{len(wanted)} task signatures and require_cached is set — "
            f"run the profile stage for this geometry first")
    if missing:
        print(f"profile cache {path.name}: {len(wanted) - len(missing)}/{len(wanted)} "
              f"known, measuring {len(missing)}")
    profiles = profile_program(program, resolver, backend, have=store, **kwargs)
    if len(profiles) > len(store):
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "env": env,
            "profiles": {repr(k): vars(v) for k, v in profiles.items()},
        }, indent=2) + "\n")
        os.replace(tmp, path)      # atomic: a killed run cannot leave a torn store
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
