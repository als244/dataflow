# dataflow.runtime — generic execution engine

**Purpose.** Execute an *annotated* `dataflow.core.Program` over a
`DeviceBackend`: object tracking, memory accounting, buffer pooling, task
dispatch, directive execution (release / offload / prefetch), and tracing.
Workload-agnostic: no DNN knowledge, no torch, no simulator imports (vendor
bindings only inside `device/cuda.py`, when it lands in M2).

## Execution model

One control thread; every state change happens inside a completion-token
handler, so the engine never polls device state and never sleeps.

- **Dispatcher** (chain order, strict pacing): for each task — (0) wait for
  the previous task's done-token (all earlier directives are then applied;
  this is what makes host-observed slot states equal the plan position),
  (1) wait until every input's fast slot is live, (2) wait until the ledger
  admits the task's fast outputs (the simulator's task stall), (3) reserve,
  launch the executable on the compute stream, register the done-token.
- **Directives** fire in the task-done handler, in the simulator's order:
  outputs live → mutations (version bump; backing goes stale) → releases →
  offload enqueues → prefetch enqueues (deferring when the object's offload
  is still in flight) → poke transfer queues.
- **Transfer engines** (one per direction, own stream): FIFO; at most one in
  flight; destination bytes charged at *start*, never enqueue; a blocked head
  waits for byte-freeing retirements and never blocks the dispatcher.
- **Ledger**: host-authoritative bytes per location; charges at
  reservation/transfer-start, frees at release/offload-completion; admission
  = `can_reserve`; any charge that was not admitted raises (invariant).
- **Pool** (`pool.py` + `slab.py`): exact-size free lists over two physical
  regimes — slab-backed (one upfront allocation per bounded location;
  best-fit carve + coalescing + headroom + pre-reserved overflow arena) and
  direct (unbounded locations, prewarmable). `slab_overflows` counts escapes
  to the vendor allocator; steady state must keep it at zero.
- **Dead-everywhere releases**: a release whose object has no later chain
  reference and no `final_locations` entry frees the backing copy too
  (mirrored in the simulator) — consumed activations vanish from both
  memories immediately.

## Static placement (`placement.py`)

The annotated plan fully determines every fast allocation's birth and death,
so placement is solved OFFLINE instead of gambled online: a fake-backend dry
run records the instance stream (`record_placement=PlacementRecorder()`),
`compute_placement` packs instances into one contiguous extent (multi-order
+ seeded-restart lowest-offset packing) and raises `PlacementError` at
planning time if the extent exceeds physical VRAM. Execution passes
`placement=` and the pool hands out fixed offsets from a single base
allocation — runtime fragmentation becomes impossible, and the slab/arena/
headroom heuristics are bypassed entirely for the fast location.

Two honest costs, both first-class in results: the packed extent genuinely
exceeds the peak concurrent load (`Placement.overhead`, the *geometry tax*
of contiguous placement — eliminating it needs VMM chunk-backed remapping,
a planned follow-up); and real completion order can differ from dry-run
order, so assigned-mode `can_get` refuses an offset while a prior
overlapping instance is live — callers stall exactly like a capacity block
(under strict pacing the blocker is always an in-flight transfer or an
earlier task, so progress is guaranteed).

Placement is an **independent, optional optimization** — the engine takes
`placement=None` (dynamic slab+arena) or a `Placement` (assigned offsets)
with identical execution semantics; the training loop's knob is
`train(placement_mode="static"|"dynamic")`, static by default. Static mode
requires a **shape-stable** program (every instance the size the dry run
recorded — the pool raises on any mismatch rather than risk overlap).
Variable-length sequence training, where object sizes differ per grad-accum
round or step, runs in dynamic mode; recording a placement from a
max-shape dry run and letting shorter instances ride in the oversized slots
is the planned middle path.

## Multi-step sessions

`Session` owns the pool, streams, and (under placement) the base allocation
across `execute()` calls; incarnation counters reset per run
(`reset_placement_epoch`), so the same placed program replays every
optimizer step. `RunResult.close()` releases engine-local resources when no
session is used.

Deadlock (dispatcher blocked or queues stuck with nothing in flight) raises
`DeadlockError` with the waiting reason and queue contents. Directive-state
violations raise `ExecutionError`. `final_locations` are verified at the end
(latest-version bytes present at the required location) — strict by default.

## Parity contract (M1 gate — enforced by tests/runtime/)

On the fake (virtual-clock) backend, the engine reproduces
`dataflow_sim.engine.simulator` *exactly*: task intervals, transfer intervals
(including `from_slow:obj#N` naming), and peak fast bytes. Load-bearing
details: transfer duration `max((size + bw - 1) // bw, 1)` with per-trigger
override; tie order at equal times = from_slow done, to_slow done, task done
(`PRIORITY_*` in `device/base.py`); reservations charged at task start.

Strict pacing costs one host wake-up per task on real hardware; an
aggressive dispatch-ahead mode (device-side input waits + committed-ahead
accounting) is an M2 experiment — its parity divergence must be quantified
before it becomes default.

## API surface

- `Engine(backend, validate=True, strict_final_locations=True, session=None)`
  → `.execute(program, resolver=None, initial_buffers=None,
  pool_prewarm=None, record_placement=None, placement=None) -> RunResult`
- `RunResult{trace, makespan_us, peak_fast_bytes, final_location_violations,
  buffers_allocated, buffers_reused, slab_overflows, pool_demand, objects,
  close()}`
- `PlacementRecorder` / `compute_placement(recorder, physical_limit_bytes)`
  / `Placement{offsets, extent_bytes, load_bytes, overhead}` / `PlacementError`
- Executable contract: `Executable.launch(TaskContext)` — enqueue on
  `ctx.stream` only; no allocation, no sync. `SyntheticExecutable` +
  `synthetic_resolver` model tasks by planned runtime.
- `compare_to_sim_eventlog(trace, event_log) -> ParityDiff`
- `device.base.DeviceBackend` — ~12-call vendor boundary (CUDA∩HIP);
  implementations: `device.fake.FakeBackend` (virtual clocks, CI-without-GPU),
  `device/cuda.py` (M2).
