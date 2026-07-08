# dataflow.runtime.device — vendor boundary

**Purpose.** The only place vendor GPU runtimes are touched. The engine sees
`DeviceBackend` (~14 calls, `base.py`); everything is restricted to the
CUDA∩HIP common subset so an AMD backend is a mechanical addition.

## Contract highlights

- **Completion tokens are the only progress signal**: `notify_after(stream,
  event, token, priority)` + `next_completion()`. The engine never polls
  device state itself and never sleeps. `next_completion` returns tokens in
  completion order (ties by `priority`: h2d done < d2h done < task done —
  the simulator's processing order) and `None` when nothing is pending
  (deadlock detection).
- **Timebase**: `event_time_us(event)` on a shared origin; valid only after
  the event completed. `mark_origin()` re-zeros after setup so traces
  measure execution, not allocation. `event_complete(event)` is the
  non-blocking pending-check (guard hygiene in the pool; legal on pending
  events, unlike `event_time_us`).
- **Annotation** (`annotate.py`): a 3-method vendor-portable protocol
  (`range_push` / `range_pop` / `mark`), `DATAFLOW_NVTX=1` selects the NVTX
  implementation (an AMD backend plugs roctx into the same calls);
  `RecordingAnnotator` is the test double. Display names may be rewritten
  per run via `Engine.execute(annotate_rename=)` — trace/plan ids never
  change.
- **Virtual-time hooks**: `advance_stream` / `align_stream_to_host` exist for
  the fake backend's clocks; real backends no-op align and reject advance
  (real executables enqueue real work).
- **`physical` flag**: True when allocations consume real memory — drives
  slab sub-allocation in the pool (fake: False).

## Implementations

- `fake.py` — virtual clocks + a heap of pending completions. Drives the M1
  parity gates and all CI-without-GPU. Setup costs zero virtual time.
- `cuda.py` — cuda-python (`cuda.bindings.runtime`). Nonblocking streams,
  timing events, `cudaMalloc`/`cudaHostAlloc`, `cudaMemcpyAsync`,
  `cudaEventElapsedTime` timebase. Two completion modes, measured on the
  RTX 5090 (engine gate, small config):
  - `poll` (default): control thread polls per-stream head events with
    `cudaEventQuery` (~µs wake latency; idle-gap p50 231µs incl. handler +
    launch; replay-fidelity gap +1.8%).
  - `hostfn`: `cudaLaunchHostFunc` → token queue (~160-270µs delivery;
    idle-gap p50 491µs; replay gap +7.1%). Kept for comparison.
  `measure_pcie()` measures pinned bandwidth in BOTH regimes — on this
  platform directions contend (uni h2d 35.5 / d2h 56 GB/s, but concurrent
  ~25.3 GB/s each). Plan transfers with the bidirectional numbers.
- `cuda_spin.py` — NVRTC-compiled spin kernel on `%globaltimer` (nanosecond
  wall clock). Wall-true regardless of SM clock ramping (a clock64-cycle
  spin calibrated at boost clock ran 2.5x+ long when launched after idle);
  verified ratio ~1.001-1.004. `make_spin_resolver(backend)` turns every
  task into a spin of its planned runtime for synthetic real-GPU runs.

## Engine gate results (RTX 5090, 2026-07-02)

Full 8B-shaped chain (103 tasks, 16 GiB budget, measured-bidi plan):

- **replay-fidelity gap +0.48%** — re-simulating with measured durations as
  overrides, the runtime's schedule matches the simulator's ideal to half a
  percent (+1.8% on the transfer-dominated small config);
- real makespan 11% *faster* than the bidi-conservative plan (unidirectional
  phases run at uni bandwidth) — attributed, not mysterious;
- peak fast bytes exactly equal to the sim's (15.7395 GiB);
- spin duration error p50 0.14%;
- dispatch overhead (GPU idle between consecutive tasks) p50 155µs ≈ 0.5%
  of mean task time; large gaps (p95+) are genuine planned transfer stalls;
- nsys API audit: zero `cudaMalloc`/`cudaFree`/`*Synchronize` on the
  steady-state path (all counts attributable to setup: bandwidth probe, spin
  verify, origin marks, slab + prewarm).

Known limitation (tracked): the fast-memory slab uses best-fit + coalescing
+ 25% headroom + counted overflow fallback (8B run: 1 overflow). The
static-assignment mode (offline placement proof from the dry run, with
event-ordered offset handoff) replaces this heuristic — see the spawned
follow-up task and PLAN_V4 §Runtime.
