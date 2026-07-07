# The task contract: what a task may do, what it may not, and why

The engine's performance model rests on three load-bearing properties.
Every rule in this contract exists to protect one of them; every rule has
been violated at least once by accident, and the cost was measured, so the
"why" columns below cite real incidents rather than theory.

1. **One control thread does everything.** The dispatcher walks the chain,
   launches compute, enqueues/completes transfers, and applies directives —
   all in one thread whose only blocking operation is waiting for the next
   completion token (`runtime/engine.py`). There is no background thread
   that keeps transfers flowing.
2. **Strict pacing.** Exactly one compute task is outstanding; task N+1 is
   dispatched only after N's done-token is host-observed. Ledger charges
   (output reserves at launch, releases/offloads/prefetches at task-done)
   therefore land at the same virtual times the simulator charges them —
   sim parity holds *by construction*, not by calibration.
3. **Profiled costs are the plan's truth.** The planner prices tasks with
   per-signature measured costs (contended, soaked, repeated). Anything a
   back-to-back profiling loop cannot observe — in particular, behavior
   that depends on how much work is queued at a given instant — is
   invisible to the cost model and will surface as plan-dependent
   real-vs-sim error.

## The contract

Three nested layers sign it: kernel implementations (`tasks/kernels/registry.py`
ABI), task executables (`runtime/executable.py` `launch()`), and resolve-time
code (capability probes, warmup).

### In the launch path (executable `launch()` and every kernel impl it calls)

**MUST**
- Enqueue all device work on the task's compute stream (`ctx.stream` /
  the ambient torch stream / `kctx.stream_handle`) and return immediately.
- Be bitwise deterministic given identical inputs (per implementation).
- Derive every shape from `TaskSpec` / dims / spec objects — host-static
  information fixed at lowering time.
- Confine side effects to declared `outputs`/`mutates` buffers (ctx views
  included). No globals, no retained references past the call.

**MUST NOT**
- Read device memory back to the host: no `.item()`, `.cpu()`, `.tolist()`,
  `.nonzero()` shape use, D2H `memcpy` into pageable memory, or any aten op
  that does one internally (`torch.bincount` does; `F.grouped_mm` does).
- Synchronize: no `stream/device.synchronize()`, `event.synchronize()`,
  blocking waits, or legacy-default-stream use.
- Branch on device data (data-dependent host control flow) — shapes and
  kernel-launch decisions must not depend on tensor *values*.
- Allocate through vendor APIs (`cudaMalloc`/`cudaFree` outside torch's
  caching allocator) — `cudaFree` device-syncs. Impls that may do this
  internally are flagged `allocates="vendor"` and demoted to A/B-only.
- Launch work on streams the runtime doesn't know about (see the
  relaxation section for the sanctioned future exception).

**MAY**
- Allocate scratch through torch's caching allocator (`allocates="torch"`,
  `workspace=internal(...)`) — steady-state cache hits are async; the
  profiler measures the peak and the planner reserves it. (This supersedes
  the older "no allocation" phrasing in `executable.py`.)
- Compose many kernels/aten calls — count is irrelevant; blocking is what's
  banned.
- Write through into context views; mutate declared `mutates` in place.

### At resolve time (sanctioned sync point)

`requires=` capability probes, one-shot correctness probes, and JIT warmup
may sync freely — resolution happens once, before any plan executes.
Profiling reps warm triton compiles, so first-call compilation never lands
inside a measured or executed step.

## Why — measured incidents, one per rule class

| violation | mechanism | measured cost |
|---|---|---|
| `torch.bincount` in `moe_sort` (hidden D2H of validation `min()`) | host blocks until the compute stream drains to the readback point; queue past it can't be enqueued → GPU starves after | 22 ms drain per recompute task; rc tasks 48% GPU-idle inside their own windows |
| `F.grouped_mm` (reads `offs` to host every call to build cutlass group descriptors) | same, ×4 per bwd task, ×1-2 per fwd/rc | the entire bs64 envelope-curve inversion; real-vs-sim −5..−18%, plan-dependent; fixed by the device-offset triton kernel → curve monotone, ±5% |
| any host block in `launch()` | the dispatcher **is** the transfer engine driver: while blocked, no tokens are processed, no queued transfer starts, no directive fires — the overlap schedule the planner counted on collapses machine-wide, not just on the compute lane | (same incidents; the single-thread coupling is why a "small" sync is never small) |
| why profiles can't save you | back-to-back reps keep the queue deep on *both* sides of a sync — drain cost ≈ 0 in the profile, large and queue-depth-dependent in the strict-paced run | profiled recompute 44.8 ms ≈ pure kernel sum; in-run 73-82 ms |

**Diagnostic signature** (how this class announces itself): replay-fidelity
tight (<1%) while real-vs-sim error is large and varies by plan/envelope/
task class; NVTX task span ≫ Σ kernel time (idle *inside* windows).
Contention looks different: kernels themselves slow down, fidelity loosens.

**Enforcement:** (1) spin-audit every new launch-path op —
`torch.cuda._sleep(~100 ms of cycles)` then call the op; a host-side
elapsed > 50 ms means it syncs (all four `F.grouped_mm` forms fail this;
a perf bench does NOT catch it). (2) nsys: any `Device->Pageable` copy
inside a task window is a finding. (3) Vendor impls that sync stay
registered for A/B only: demoted priority + `allocates="vendor"`.

## Consequence: which GEMM backends are legal where

- **cublasLt / cuBLAS on host-static shapes** (dense projections, router,
  shared expert — everything whose M/N/K is fixed at lowering): **legal**.
  Descriptors are built from plan-time constants; nothing reads the device.
  flextrain's `matmul_dispatcher` (ctypes cublasLt with heuristic caching)
  would be a clean registry impl for dense ops *after* a spin-audit of its
  steady state, with first-call heuristic queries pushed to resolve/profile
  warmup.
- **cublasLt / cuBLAS grouped over dynamic expert segments**: **illegal
  under this contract** — not because the library is slow, but because
  every grouped-style cuBLAS API (`cublasLtMatmul` descriptors,
  `cublasGemmGroupedBatchedEx`) takes per-problem sizes as **host
  integers**, and dropless segment sizes exist only on device (produced by
  the in-graph sort). Every path from those counts to a descriptor passes
  through a launch-path D2H readback — the exact violation above. This is
  also why torch's own `F.grouped_mm` (cutlass underneath, but descriptors
  built host-side by the wrapper) had to be replaced rather than tuned.
- **Device-descriptor backends** (our triton grouped GEMM; a future direct
  cutlass grouped integration whose problem visitor reads offsets from
  device memory): **legal and default** — the datacenter-grade path when
  triton trails cublasLt-class kernels on H100/B200.

## Relaxation design space (defined, not built)

The prohibition is on *blocking the control thread*, not on host knowledge
per se. Three sanctioned ways to let host-shape APIs participate, in
rising order of engine change:

- **C. Device-descriptor kernels (status quo).** No engine change; per-op
  kernel engineering. This is what shipped for grouped GEMM.
- **B. Capacity mode (static segments).** A per-expert row cap makes
  segment shapes plan-time constants → any host-shape library becomes
  legal with zero engine changes, and shapes go fully static (even
  friendlier to this IR than today's dynamic-within-fixed-total). Costs:
  padded FLOPs at cap ≈ 1.1-1.25× mean, and semantics change (token drops
  beyond cap — breaks dropless exactness/parity). `MoESpec`/
  `moe_local_rows` is the seam where this policy already lives; EP
  capacity policies may force it anyway.
- **A. Planned host-readbacks (the principled dynamic option).** A plan-
  level `readback_after` directive, symmetric with `offload_after`/
  `prefetch_after`: after the producing task's done-token, the engine
  copies a small device object (e.g. `route_offsets`, 260 B) into pinned
  staging on a **side stream** and marks a host-value table entry when its
  event completes — observed via the normal token loop, never by blocking.
  Downstream tasks declare a host-value precondition the dispatcher checks
  like input-liveness; their executables receive the host integers and may
  legally drive cublasLt per-segment calls, kernel-launch geometry, etc.
  The sim models it as a tiny transfer plus a host-availability edge; the
  profiler measures phase costs as usual. Two notes: (1) a *secondary
  stream* alone does not solve the problem — the issue is host knowledge,
  not stream occupancy; the side stream is just what keeps the readback
  from serializing behind the compute queue tail (aten's mistake). (2) In
  bwd, offsets are already-saved ctx — a single fwd-time readback per
  (layer, round) would cover all four bwd grouped calls with zero
  bwd-time latency. **EP will force this decision regardless**: all-to-all
  dispatch needs received-row counts host-side for most collective APIs,
  so building A is the likely EP prerequisite — treat it as the sanctioned
  extension point, and don't add ad-hoc syncs in the meantime.

Costs of A worth stating up front: a new token/directive type in engine +
sim + replay + profiler; a bounded new timing sensitivity (host-value
availability = event completion + token-poll latency, tens of µs); and
plans gain edges that only exist under dynamic routing — replay fidelity
machinery must carry the host-value table across runs.
