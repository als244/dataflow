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
- **Pool**: exact-size free lists; physical availability == logical
  availability when sizes repeat. Real-backend preallocation arrives in M2.

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

- `Engine(backend, validate=True, strict_final_locations=True)` →
  `.execute(program, resolver=None, initial_buffers=None) -> RunResult`
- `RunResult{trace, makespan_us, peak_fast_bytes, final_location_violations,
  buffers_allocated, buffers_reused}`
- Executable contract: `Executable.launch(TaskContext)` — enqueue on
  `ctx.stream` only; no allocation, no sync. `SyntheticExecutable` +
  `synthetic_resolver` model tasks by planned runtime.
- `compare_to_sim_eventlog(trace, event_log) -> ParityDiff`
- `device.base.DeviceBackend` — ~12-call vendor boundary (CUDA∩HIP);
  implementations: `device.fake.FakeBackend` (virtual clocks, CI-without-GPU),
  `device/cuda.py` (M2).
