# Dataflow Runtime — Project Plan (v2)

Supersedes [INITIAL_PLAN.md](INITIAL_PLAN.md). Changes in v2: references treated as concept sources only (fresh code, minimal surface); DeviceBackend abstraction layer; cost/size/workspace system owned by this repo with measurement as the authoritative source; explicit memory-accounting design; multi-step training from the start; linear-chain dispatch over a DAG-ready IR.

## Context

Goal: realize the [dataflow simulator](https://dataflowsim.sunshein.net/) as a true CPU–GPU runtime: a generic engine that ingests annotated dataflow programs (tasks with input/mutated/output objects + release/offload/prefetch directives), owns GPU/pinned-host memory, streams/events and transfers, and dispatches task executables — plus a DNN-training lowering layer that turns model definitions into such programs, annotated by dataflow_sim's PressureFit policy + recompute planner, with end-to-end multi-step training throughput matching simulator predictions.

Hardware: RTX 5090 (32 GB, CUDA 13.1), i9-13900KF, 188 GB RAM, PCIe x16. Env: conda `dataflow` (py3.12) with `dataflow_sim` installed editable (done).

## Design principles

1. **References are references.** `refs/prior_attempt` and `refs/flextrain` contribute *concepts*, not code. Everything is written fresh in this repo, fitted together, with a minimal public surface per layer. Concepts worth absorbing: the prior attempt's object-slot state machine, `launch(ctx)` binding shape, and sim-parity harness; flextrain's declarative activation schema + stateless composable blocks; both repos' documented failure modes (split-brained layers, N code paths agreeing by convention, embed/head special-casing).
2. **Device abstraction.** The runtime never calls CUDA directly; it talks to a small `DeviceBackend` interface whose surface is restricted to the CUDA∩HIP common subset, so an AMD backend later is mechanical. Only `cuda` (+ virtual-clock `fake`) implemented now.
3. **Measurement over estimation.** Object sizes are exact by construction. FLOPs/bytes are declared next to op implementations and test-validated. Task runtimes and temp/workspace usage are *measured* per unique task by a profiling harness; measurements are authoritative and are written back into programs before final planning. The simulator consumes costs; it is never their source.
4. **Linear chains now, DAG-ready IR.** The dispatcher executes the task list in order (the sim's and policies' model). The IR derives dependencies from object producer/consumer relations rather than baking in "position = dependency", so a DAG dispatcher can arrive later (multi-GPU era) without a program-format change.
5. **The runtime is policy-agnostic.** It consumes an annotated program; PressureFit/recompute planning is an isolated, swappable step owned by the training layer via dataflow_sim.

## Ground truth from the three repos (read in full / via deep readers)

**dataflow_sim** (we control it):
- Schemas: portable costed `DataflowProgram` v1 (pydantic; `objects`, `compute_blocks` with `DataflowCost` subops, `tasks` w/ `compute_block_key`+`metadata`, `metrics`, `final_locations`) and executable `TaskChain` (`Task{inputs, outputs, runtime, releases_after, offload_after, prefetch_after, mutates_inputs}`).
- Engine semantics the runtime must reproduce (`engine/simulator.py`): serial compute chain; task start = inputs live-in-fast AND fast capacity reservable for outputs; one in-flight transfer per direction, FIFO; **destination bytes allocated at transfer start, not enqueue**; queue head blocks on capacity, retried on frees; prefetch-during-offload defers until the offload completes; release requires live state; mutated inputs must be offloaded, not released.
- PressureFit: `apply_pressurefit_policy(bare, fast_memory_capacity=)` — interval residency → pressure reduction → 4 inbound schedules → sim-verified fastest; assumes linear chains; schema-driven.
- Recompute planner: `plan_with_recompute(build_variant_fn, rewrites, policy_fn, ...)` with `RecomputeRewrite{object_id, f_task_id, r_task_id, options[RecomputeOption{level, saved_bytes, recompute_us}], f/r_compute_block_key, group_key}`.
- Webapp: `POST /api/simulate` accepts `{source:"schema", schema: DataflowProgram}`; renders timeline/memory/summary/diagnostics/budget-sweep. Gotcha: schema uploads carry no recompute rewrites → run recompute locally before export (webapp-side support is a later sim-repo enhancement).
- Its `workloads/` model builders remain webapp presets and a cross-check, not our lowering.

**prior_attempt** (concepts): scheduler achieved ~µs parity vs sim on the full 8B chain with synthetic tasks (fake backend) — the parity-harness idea is the key keeper. Its flaw: capacity waits host-blocked the dispatch loop (would stall compute on real HW). Its JAX lesson: the execution substrate must own buffers/streams/workspaces. Its native probes proved cuBLASLt bf16 GEMM (~0.8 eff) + custom bf16 kernels + compute/D2H overlap with pinned memory on this stack.

**flextrain** (concepts): stateless blocks with explicit `weights/slot/grads` args and a declarative per-block activation schema (fields, tiers, offload flags) compose into 10+ model families with no duplication; ~20 Triton kernels + cuBLASLt dispatcher + flash-attn cover the math; per-tensor dtype specs (compute/master/grad/opt_state). Its predecessor's disorganization (4 code paths agreeing on names, dense/MoE copy-paste, embed/head special cases) is the anti-pattern our schema-driven design avoids.

## Architecture

Monorepo at this directory, one installable package `dataflow`, strict layering enforced by an import-boundary test:

```
src/dataflow/
├── core/        # L0 — program IR: objects/tasks/directives + tensor metadata (shape/dtype/strides)
│                #      + compute_block_key/params + recompute metadata; JSON round-trip; validation;
│                #      converters ↔ sim TaskChain and → DataflowProgram v1 (webapp export).
│                #      Zero heavy deps (no torch/jax/cuda/dataflow_sim at import time).
├── runtime/     # L1 — generic engine: object table + byte ledger, memory pools, dispatcher,
│                #      transfer engines, directive execution, trace/telemetry.
│                #      device/ subpackage: DeviceBackend interface + fake + cuda implementations.
│                #      No DNN knowledge; no torch; cuda-python only inside device/cuda.py.
├── tasks/       # L2 — executable library: ops → blocks; stateless launch(ctx); declared
│                #      flops/bytes/workspace beside each op; torch/Triton backend first,
│                #      native (cuBLASLt/custom kernels) grown for hot ops later.
├── training/    # L3 — lowering: ModelDef + TrainingSpec → core program (+ RecomputeRewrites
│                #      + executable factory table keyed by compute_block_key); planning step calls
│                #      dataflow_sim (PressureFit + plan_with_recompute); profiling harness.
├── models/      # L4 — declarative model definitions (llama3 first).
tools/           # parity/profile/export/bench/run scripts
tests/
```

Dependency rule: `core ← runtime`, `core ← tasks`, `core+tasks ← training ← models`; only `training` imports `dataflow_sim`; only `tasks` imports torch/triton; only `runtime/device/cuda.py` imports cuda-python.

### DeviceBackend (L1 boundary to vendor runtimes)

~12 calls, all with direct HIP equivalents: `create_stream`, `record_event`, `stream_wait_event`, `query_event`, `sync_event` (shutdown/error paths only), `alloc_device`, `alloc_pinned`, `free_*` (setup/teardown only), `memcpy_async`, `launch_host_callback`, `event_elapsed_time`, `device_info`. Implementations: `fake` (virtual clocks — drives sim-parity gates and CI-without-GPU), `cuda` (cuda-python). Executables receive an opaque stream handle; torch/Triton executables wrap it (`torch.cuda.ExternalStream`) and are themselves ROCm-portable; native executables are per-vendor plugins.

### End-to-end flow

```
models.llama3 (ModelDef) + TrainingSpec
  → training.lower(): objects (exact sizes) + tasks (block keys, params, declared costs)
                      + RecomputeRewrites + executable factories
  → bare chain → plan_with_recompute(rebuild_variant, rewrites, apply_pressurefit_policy)
  → annotated chain → core program (tensor/binding metadata joined back)
  → [profile pass: measured runtimes + workspace → re-plan]        (mandatory before headline runs)
  → runtime.execute(program, executables, initial_buffers)          (cuda backend)
  → trace (events, memory, per-task timings) → compare vs sim EventLog; export webapp JSONs
```

## Runtime design (L1)

**Dispatcher** — one host thread walks the chain in order, running ahead of the GPU. Per task: resolve input/mutate slots, reserve outputs in the ledger, `stream_wait_event(compute, ready)` per input, `executable.launch(ctx)`, record done-event, register directive actions anchored on it. Completions (task/transfer done) arrive as tokens via host callbacks (fallback: polling reaper) into a queue the dispatcher drains.

**Memory accounting (explicit design)** — a host-authoritative byte ledger per location (fast/backing), mirroring sim states exactly: bytes count from output-reservation or transfer-start until release-retirement or offload-completion. The ledger never queries the GPU; it changes only at dispatcher decisions and completion-token retirements. Admission points checking `used + need ≤ capacity`:
1. *Task output reservation* (dispatcher, in task order). Insufficient space ⇒ dispatch pauses — precisely the sim's task stall — and resumes on byte-freeing tokens. Cost of the host being in this loop: µs-scale wake latency on top of an already-stalled schedule.
2. *h2d (prefetch) queue head* (transfer engine). A blocked head delays only transfers, never compute dispatch (fixes the prior attempt's flaw). Re-attempted on every byte-freeing retirement.
3. *d2h destination* against backing capacity, likewise.
Deadlock (waiters exist, nothing in flight frees) ⇒ hard error with a sim-style diagnostic. Physical invariant: the pool must satisfy anything the ledger admits — exact-size free-lists make physical == logical for repeated sizes (transformer chains repeat few sizes); novel sizes fall back to slab best-fit; an admitted-but-unsatisfiable request is a loud invariant violation (VMM remap is the upgrade path). Cross-stream buffer reuse is safe via event ordering: a reused buffer's next consumer stream waits on the previous occupant's release event.

**Transfer engines** — per-direction FIFO over h2d/d2h streams; launch `memcpy_async` with stream-waits on anchor (task-end) + source-ready events; completion tokens retire source-free/dest-live and re-attempt blocked heads; deferred prefetch (same-object offload in flight) exactly as the sim.

**Object table** — id → {size, role, tensor meta, version, fast/backing slots with states live/reserved/pending_inbound/inbound/pending_outbound/outbound/released, ready events}. Mutation bumps versions; mutated objects must be offloaded per plan (validated).

**Trace** — pre-created event pools; per-task/per-transfer timestamps; memory trace by band; export shape-compatible with sim `EventLog` for side-by-side comparison and webapp-style plotting.

**Multi-step execution** — lowering guarantees `final_locations` == next-step initial locations; the runtime replays the same annotated chain per optimizer step with persistent objects (params, optimizer state) and per-step input injection; host syncs only at step boundaries.

## Task executables (L2)

Contract: `Executable.launch(ctx)`, `ctx = {task, stream, inputs/outputs/mutates: {id → Buffer(ptr, size, TensorMeta)}, workspace}` — enqueue on `ctx.stream` only; no allocation; no sync; no globals; idempotent. Resolution by `(compute_block_key, params)` so planner-inserted recompute tasks bind automatically.

Torch/Triton backend discipline:
- Zero-copy tensor views over runtime buffers via DLPack; ops under `torch.cuda.ExternalStream(runtime_stream)`; all block outputs written into runtime-provided buffers (`out=` / slot-style, as flextrain blocks already do).
- **Temp/workspace policy (hybrid, measurement-authoritative)**: large deterministic workspaces are declared per-op (formulas beside the implementation) and lowered into explicit `temp` objects the policy can see. A small bounded torch-allocator scratch lane (pre-warmed, accounted as overhead inside the fast budget) absorbs residual op scratch. The profiling harness *measures* per-unique-task workspace high-water (allocator peak-delta with externally preallocated I/O); measured values override declarations in the program, and any task whose scratch exceeds its envelope fails loudly. `torch.compile` is not used in v1 executables (no public workspace introspection; cudagraph pools break external-buffer ownership) — eager + Triton + explicit kernels; revisit later.
- Assert-no-alloc test mode (allocator stats before/after each task) keeps steady state cudaMalloc-free.

## Lowering + costs (L3/L4)

- ModelDef: declarative composition (flextrain-style) — param specs, block sequence, per-block saved-context schema (fields/tiers → packed `A_*` layouts and `RecomputeOption`s), block keys + params; embed/head first-class.
- Lowering emits step/round/layer tasks (sim-compatible naming `W_i, A_s_r_i, dW_s_i, O_i`), grad-accum via round-0-fresh / later-rounds-mutate, optimizer tasks, `RecomputeRewrite`s, and `rebuild_variant(levels)` for the recompute planner. Multi-step-invariant by construction.
- Cost system (measurement over estimation):
  - sizes exact (shape × dtype), test-asserted vs real tensors;
  - per-op declared flops/bytes, test-validated with `torch.utils.flop_counter.FlopCounterMode`;
  - first plan from a small owned roofline util (declared costs × hardware spec table);
  - **mandatory profile pass**: per unique `(block_key, shape-sig)` — CUDA-event-timed runtime + measured workspace, written back into the program metadata (estimate and measurement both kept); final planning on measured costs;
  - webapp export carries declared flops/bytes subops so the webapp can re-resolve for any hardware.

## Milestones (each with a hard gate)

- **M0 — scaffold + IR + sim interop.** Package layout, `core` IR/validation/JSON, converters, import-boundary test. Gate: tiny + 8B-shaped llama3 programs round-trip core↔sim; PressureFit + recompute produce annotated chains; sim runs them; export accepted by local webapp (`POST /api/simulate`, source:"schema").
- **M1 — runtime parity, virtual.** Full engine on the fake backend (ledger, transfer engines, directives, non-blocking admission). Gate: makespan/peak-fast/event-order parity vs sim EventLog on the full 8B annotated chain, plus adversarial unit chains (deferred prefetch, blocked queue head, mutation offload, re-prefetch, deadlock detection).
- **M2 — runtime real, synthetic tasks.** `cuda` DeviceBackend; calibrated spin-kernel executables + real memcpys on a scaled-to-32GB program. Gate: on-GPU trace vs sim within a few % makespan; overlap visible in nsys; zero implicit syncs (nsys audit); measured host dispatch overhead ≪ mean task runtime.
- **M3 — real executables (torch/Triton).** Blocks: embed, rmsnorm, GQA attention, SwiGLU, head+loss, adamw, block fwd/bwd/recompute compositions; DLPack/ExternalStream plumbing; no-alloc discipline; profiling harness (runtime + workspace). Gate: per-block and end-to-end gradient parity vs plain-torch reference; measured-cost programs re-planned; trace sim-consistent at small scale; workspace envelopes hold.
- **M4 — end-to-end memory-constrained multi-step training.** Llama3-8B-class on the 5090 across a fast-budget sweep (bf16 params+grads exceed VRAM → genuine weight/grad offload + recompute + Adam-state-on-host). Gate: N-step training, decreasing loss, state persistent across steps, steady-state throughput within target band of measured-cost sim prediction (≥90% at generous budget; quantified gap analysis under pressure); program + measured trace uploadable to webapp.
- **M5 — stretch.** Native executables for hot ops; CUDA-graph capture of the steady-state step; Qwen3 dense/MoE; sim-repo enhancement (schema carries recompute rewrites); AMD backend when hardware exists; multi-GPU groundwork.

## Risks / watch-items

- Torch hidden allocations re-creating the JAX problem → measurement-authoritative workspace + no-alloc assertions + scratch lane; fallback native executables for offenders.
- Python dispatch overhead → measured at M2; fallbacks: batch pre-enqueue, Cython/C++ hot loop, CUDA graphs.
- Exact-size pooling fragmentation on diverse programs → size-class rounding, then VMM remap.
- Host-callback latency/GIL → polling-reaper fallback maintained from M1.
- PressureFit plans assume exact runtimes; real jitter shifts interleavings → correctness unaffected (event-driven); throughput sensitivity quantified in M4 (jittered-sim experiment).

## Decision log

- 2026-07-02 (v1): torch/Triton-first executables; Python-first runtime core; model definitions live in this repo (sim consumed as a library for policies/planning/verification/webapp); M4 targets Llama3-8B-class.
- 2026-07-02 (v2): fresh code only, references for concepts; DeviceBackend abstraction (CUDA∩HIP surface, cuda-only impl for now); costs/sizes owned by this repo — declared per-op, validated by FlopCounterMode, runtimes+workspace measured per unique task and authoritative; hybrid temp policy (declared temps + bounded measured scratch lane); host-authoritative byte ledger mirroring sim states with three admission points; multi-step training from the start; linear-chain dispatch with DAG-ready IR.

## Immediate next steps (M0)

1. `git init` + pyproject scaffold; deps: `dataflow_sim` (editable sibling), `cuda-python`, torch+triton (tasks layer), pytest.
2. `dataflow.core` IR + validation + JSON + sim converters (fresh implementation).
3. Golden-path script: tiny llama3-shaped program → PressureFit → recompute → sim run → webapp JSON; committed as regression fixture.
4. M1 fake-backend engine + parity harness.
