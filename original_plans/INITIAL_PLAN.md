# Dataflow Runtime — Initial Project Plan

## Context

Goal: realize the [dataflow simulator](https://dataflowsim.sunshein.net/) as a true CPU–GPU runtime: a generic engine that ingests annotated dataflow programs (tasks with input/mutated/output objects + release/offload/prefetch directives), owns GPU/pinned-host memory, CUDA streams/events and transfers, and dispatches task executables — plus a DNN-training lowering layer that turns model definitions into such programs, annotated by dataflow_sim's PressureFit policy + recompute planner, with end-to-end throughput matching simulator predictions.

The prior attempt (refs/prior_attempt = learning_jax) established: program generation/costing/annotation works; a generic pointer-scheduler replays annotated chains at ~µs parity with the simulator when tasks are synthetic; JAX gradient math is correct — but JAX cannot be the execution substrate (owns streams/buffers/workspaces/autotuning; per-task barriers were load-bearing). Restart: keep the scheduler architecture and program contracts, build a runtime-owned CUDA execution substrate, use JAX/torch only as reference/math.

Hardware: RTX 5090 (32 GB, CUDA 13.1), i9-13900KF, 188 GB RAM, PCIe x16. Env: conda `dataflow` (py3.12) with `dataflow_sim` installed editable (done).

## What exists (verified by reading all three repos)

**dataflow_sim** (we control it):
- Two schema layers: portable costed `DataflowProgram` v1 (pydantic: `objects`, `compute_blocks` (subops = `DataflowCost` fixed/roofline/sum), `tasks` w/ `compute_block_key`+`metadata`, `metrics`, `final_locations`) and executable `TaskChain` (`core/schema.py`: `Task{inputs, outputs, runtime, releases_after, offload_after, prefetch_after, mutates_inputs}`).
- `realize_dataflow_program(program, hw) → Workload{chain, metadata}` resolves roofline costs against `HardwareSpec` (presets incl. RTX_5090).
- Engine semantics (`engine/simulator.py`, read in full — the runtime must reproduce these): serial compute chain; task start = inputs live-in-fast AND fast capacity reservable for outputs; one in-flight transfer per direction with FIFO queues; **destination bytes allocated at transfer start, not enqueue**; queue head blocks on insufficient capacity, retried on frees; prefetch-during-offload defers until offload completes; release requires live state; mutated inputs must be offloaded (not released) to persist.
- PressureFit (`apply_pressurefit_policy(bare, fast_memory_capacity=)`): interval residency → greedy pressure reduction → 4 inbound schedules → sim-verified fastest. Schema-driven, assumes linear chains.
- Recompute planner (`planning/recompute.py`): `plan_with_recompute(build_variant_fn, rewrites, policy_fn, max_iters, max_wall_s)` — stall-blame evidence loop over per-object binary save/recompute levels; `RecomputeRewrite{object_id, f_task_id, r_task_id, options[RecomputeOption{level, saved_bytes, recompute_us}], f/r_compute_block_key, group_key}` published in training-workload metadata.
- Full model-training lowering already in-repo: `workloads/ops` (cost factories) → `modules` → `models/{llama3,qwen3,...}` (`Config` + `{Model}ForTraining(TrainingBuilder)`) → costed programs with objects `W_i / A_s_r_i / dW_s_i / O_i / y_*`, tasks `block_fwd/bwd/recompute/optimizer/head/loss`, `DTypePolicy` per-role dtypes.
- Webapp: FastAPI `POST /api/simulate` accepts `{source:"schema", schema: DataflowProgram}` + hardware + planner params; UI renders timeline/memory/summary/policy-diagnostics/budget-sweep. **Gotcha**: schema uploads carry no `recompute_rewrites`, so webapp recompute planning only works for built-in workloads (fixable later in sim repo; non-blocking — we upload post-recompute programs).

**Prior attempt (learning_jax)** — reusable pieces:
- `dataflow_runtime/schema.py`: runtime-side program IR mirroring sim TaskChain 1:1 + lossless converters both ways (`to/from_dataflow_sim_task_chain`), `sim_export.py` → webapp JSON. (Port; drop its JAX import.)
- `async_runtime.py`: single-host-thread in-order dispatcher over `RuntimeBackend` protocol; exact-size buffer free-lists; pending-action retirement on events; achieved ~µs parity vs sim with virtual-clock fake backend on full 8B chain. **Flaw to fix**: `_ensure_capacity` host-blocks in the dispatch loop on prefetch/output reservation → on real HW stalls subsequent compute dispatch; sim instead blocks only the transfer-queue head while compute proceeds.
- `pointer_types.py` binding contract: `PointerTaskBinding.launch(PointerTaskContext{task, stream, inputs/outputs/mutates: {id→BufferHandle}, backend})` — enqueue, never sync. Keep.
- `backends/cuda_runtime.py` (real `cuda.bindings` streams/events/memcpy) + `cuda_utils.py` (bf16 kernels: embedding fwd/bwd-accum, rmsnorm, rope, swiglu, residual, CE-loss bwd, naive attention; KernelHelper) + `cublas_utils.py` (cuBLASLt bf16 GEMM ~0.8 eff) + overlap probe proving compute/D2H concurrency with pinned memory.
- Lowering precedent: tasks+bindings emitted together, bindings keyed by task_id (closures) — new design keys executables by `compute_block_key`+params instead so planner-inserted tasks resolve automatically. Grad accum: round 0 outputs fresh `dW`, later rounds input+mutate it.
- Cost lessons: JAX/XLA cost analysis incomplete → reconcile(max) with structural + analytical formulas; roofline first, profile unique tasks (dedup by block key + shapes), store both estimate+profile in metadata.
- `buffer_planning.py`: exact-size interval→slot static reuse analysis (useful for a later static-assignment mode).

**flextrain** — the ops/modules/models precedent (largely adaptable to L2):
- Stateless composable blocks already in task shape: `Layer` protocol (`forward/backward/forward_recompute/compute_cost` + declarative `schema: ActivationSchema` + `param_spec`), blocks (RMSNorm/GQA-attn/SwiGLU/MoE/LinearAttn) take explicit `weights`/`slot`/`grads` dicts, no hidden state; 10+ model families via pure composition (~500 LOC each).
- Kernels: ~20 Triton kernels (norm/swiglu/gelu/rope/CE/moe-scatter-gather-topk/adamw/muon), cuBLASLt C++ dispatcher (fused matmul+activation+residual), flash-attn bindings; per-tensor dtype specs (compute/master/grad/opt_state).
- Its documented failure modes (from its own predecessor) = what our schema-driven design must avoid: N code paths agreeing on tensor names by convention; dense/MoE duplication; embed/head special-casing; interface drift.

## Architecture

Monorepo at `~/Documents/grad_school/research/dataflow`, one installable package `dataflow`, strict layering (enforced by an import-boundary test):

```
src/dataflow/
├── core/        # L0 — program IR: ObjectSpec/TaskSpec/directives + tensor metadata (shape/dtype)
│                #      + compute_block_key/params + RecomputeRewrite passthrough; JSON round-trip;
│                #      converters: ↔ sim TaskChain, → DataflowProgram v1 (webapp). No jax/torch/cuda deps.
├── runtime/     # L1 — generic engine: object table, device+pinned pools, streams/events,
│                #      transfer engines, dispatcher, directive execution, trace/telemetry.
│                #      Backends: fake (virtual clock) | cuda (cuda.bindings). No DNN knowledge, no torch.
├── tasks/       # L2 — executable library: ops → blocks (torch/Triton backend first, native later);
│                #      executables resolve by (compute_block_key, params); stateless launch(ctx).
├── training/    # L3 — DNN lowering: ModelDef + TrainingSpec → dataflow.core program (+ rewrites,
│                #      + binding factory table); planning step calls dataflow_sim (PressureFit +
│                #      plan_with_recompute); profiling harness writes measured runtimes back.
├── models/      # L4 — declarative model definitions (llama3 first; flextrain-style composition).
tools/           # export/parity/profile/bench/run scripts
tests/
```

Dependency rule: `core ← runtime`, `core ← tasks`, `core+tasks ← training ← models`; only `training` imports `dataflow_sim`; only `tasks` imports torch. The runtime never imports torch, jax, or dataflow_sim (validation via sim is done by callers through `core` converters).

### End-to-end flow

```
models.llama3 (ModelDef) + TrainingSpec
  → training.lower(): objects+tasks (shapes/dtypes/block keys) + RecomputeRewrites + cost subops
      (cost formulas reuse dataflow_sim.workloads.ops factories where they fit)
  → dataflow_sim: realize costs w/ HardwareSpec → bare TaskChain
  → plan_with_recompute(rebuild_variant, rewrites, apply_pressurefit_policy)   [isolated, swappable]
  → annotated chain  → core program (join back tensor/binding metadata)
  → runtime.execute(program, executables, initial_buffers)   [cuda backend]
  → trace (events, memory, per-task CUDA timings)  → compare vs sim EventLog; export webapp JSONs
```

Program structure is defined once, in this repo, execution-grade (shapes/dtypes first-class); dataflow_sim is consumed as a library for costs/policies/planning/verification/visualization. The sim's own `models/` stay as webapp presets; a consistency test cross-checks our llama3 program totals against sim's built-in llama3 builder for the same config.

### Runtime design (L1 — the hard part)

- **Dispatcher**: one host thread walks the chain in order, dispatching ahead of the GPU. Per task: resolve input slots (must exist per plan validity), reserve output buffers, `cudaStreamWaitEvent(compute, ready_event)` per input/mutate, call executable `launch(ctx)`, record done-event, register directive actions anchored on that event. Completions (task done, transfer done) delivered as tokens via `cudaLaunchHostFunc` → thread-safe queue drained by the dispatcher (fallback: event polling).
- **Never block dispatch on transfers** (fixes prior flaw): prefetch/offload directives enqueue jobs into per-direction host FIFO queues; a queue head that can't reserve destination capacity waits inside the transfer engine while compute dispatch continues. The only legitimate dispatch stall is a task's own inputs/output-reservation (sim-equivalent stall), realized as waiting on the completion queue, not spinning on CUDA.
- **Transfer engines**: per-direction FIFO (h2d, d2h streams); admission = capacity reservation at start (sim semantics); launch `cudaMemcpyAsync` with stream-waits on anchor task-done + source-ready events; completion token retires source-free/dest-live and re-attempts blocked queue heads. Deferred prefetch (offload of same object in flight) handled exactly like the sim.
- **Memory manager**: device slab(s) preallocated to fast budget; exact-size bucketed free-lists (matches sim's byte accounting and prior parity result; transformer programs have few distinct sizes) with buffer reuse safety by event ordering (a reused buffer's new consumer stream waits on the releasing event). Pinned-host pool likewise. No cudaMalloc/Free in steady state (they sync). Object table: id → {size, role, tensor meta, version, fast/backing slots, states, ready events} — port prior `ObjectRecord/ObjectSlot` state machine (live/reserved/pending_inbound/inbound/pending_outbound/outbound/released).
- **Trace**: pre-created CUDA event pools; per-task and per-transfer start/end timestamps; memory trace by band; export shape-compatible with sim `EventLog` for side-by-side comparison and webapp-style plotting.
- **Multi-step training**: program's `final_locations` must equal initial locations of the next step so the same annotated chain replays each optimizer step without replanning; runtime supports repeated `execute()` with persistent objects (params, optimizer state) held across runs.

### Task executable contract (L2)

`Executable.launch(ctx)` with `ctx = {task, stream, inputs/outputs/mutates: {id → Buffer(ptr, size, TensorMeta)}, workspace}`: enqueue on `ctx.stream` only; no allocation (large known temps are declared `temp` objects in the program; small op scratch comes from a bounded workspace lane); no sync; no globals; idempotent.

Torch backend discipline (the JAX lesson applied):
- Zero-copy views over runtime buffers via DLPack (`torch.from_dlpack` on our capsules); ops run under `torch.cuda.ExternalStream(runtime_stream)`; Triton launches on torch current stream; flextrain's cuBLASLt dispatcher takes explicit streams.
- All block outputs written into runtime-provided buffers (flextrain blocks already write into explicit slot tensors).
- Torch caching allocator confined to a bounded, pre-warmed scratch lane accounted in the fast budget; assert-no-alloc test mode (allocator stats before/after each task); large workspaces surfaced as declared temp objects so the policy sees them.
- Port flextrain block math (Triton kernels, dispatcher, flash-attn) with thin adapters mapping ctx buffers → the `weights/slot/grads` dicts; flextrain's `ActivationSchema` declarations become the layout source for packed per-block `A_*` context objects and recompute tiers → `RecomputeOption` metadata.
- Native backend (prior `cuda_utils.py`/`cublas_utils.py` kernels) kept for benchmarking and grown opportunistically for hot ops.

### Lowering + costs (L3)

- ModelDef: declarative, flextrain-composition-style — param specs, block sequence, per-block saved-context schema + recompute options, block keys + params; embed/head first-class (no special cases).
- Lowering emits step/round/layer-structured tasks (naming compatible with sim conventions `W_i, A_s_r_i, dW_s_i, O_i`), grad-accum mutation pattern, optimizer tasks, and `RecomputeRewrite`s; `rebuild_variant(levels)` re-lowers for the recompute planner.
- Costs: analytical FLOPs/bytes per block (reuse sim `workloads/ops` cost factories) → roofline via sim HardwareSpec for initial runtimes; profiling harness executes unique executables (dedup by block key + shape signature, warmup+repeats, CUDA-event timed) and writes measured `runtime_us` back into the program before final planning. Object sizes are exact by construction (shape × dtype).

## Milestones (each with a hard verification gate)

- **M0 — scaffold + IR + sim interop.** Package layout, `core` IR + JSON I/O + converters, import-boundary test. Gate: llama3-8B-shaped program (tiny + full) round-trips core↔sim; PressureFit + `plan_with_recompute` produce annotated chains; sim runs them; export JSON accepted by local webapp (`uvicorn dataflow_sim.app.server.main:app`, POST /api/simulate with source:"schema").
- **M1 — runtime parity, virtual.** `runtime` with fake backend (virtual clock) replicating full semantics incl. non-blocking transfer admission. Gate: makespan/peak-fast/event-order parity vs sim EventLog on the full 8B annotated chain (prior attempt's ~µs parity reproduced; port of `run_pointer_synthetic_plan` harness) + adversarial unit chains (deferred prefetch, blocked queue head, mutation offload, re-prefetch).
- **M2 — runtime real, synthetic tasks.** CUDA backend: streams/events/pools/pinned memory/host-func completions; calibrated spin-kernel executables + real memcpys on a scaled-to-32GB program. Gate: on-GPU trace vs sim within a few % on makespan; overlap visible in nsys; zero implicit syncs (nsys audit); host dispatch overhead measured ≪ mean task time.
- **M3 — real executables (torch backend).** Port flextrain blocks (embed, rmsnorm, GQA attention, SwiGLU, head+loss, adamw, block-level fwd/bwd/recompute compositions) as stateless executables; DLPack/ExternalStream plumbing; no-alloc discipline; profiling harness. Gate: per-block and end-to-end gradient parity vs plain-torch reference (tiny model, fp32/bf16 tolerances); profiled-cost programs re-planned; trace still sim-consistent at small scale.
- **M4 — end-to-end memory-constrained training.** Llama3-8B class on the 5090 across a fast-budget sweep (bf16 params+grads alone exceed 32 GB → genuinely exercises weight/grad offload, activation recompute, Adam-state-on-host). Gate: multi-step loss-decreasing training; throughput within target band of profiled-sim prediction (≥90% at generous budget; quantified gap analysis under pressure); program + measured trace uploadable to webapp for visual comparison.
- **M5 — stretch.** Native executables for hot ops; CUDA-graph capture of steady-state step; second model family (Qwen3 dense/MoE); sim-repo upgrades (schema carries recompute_rewrites so webapp can re-plan uploads); multi-GPU groundwork.

## Risks / open design watch-items

- **Torch hidden allocations** (workspaces, autotune) re-introducing the JAX problem — mitigated by scratch-lane budgeting + no-alloc assertions + declared temps; fallback is native executables for offending ops.
- **Host dispatch overhead (Python)** on ~ms-scale tasks should be ignorable off critical path; measured at M2 — fallback: batch pre-enqueue, Cython/C++ hot loop, or CUDA graphs (M5).
- **Exact-size pooling fragmentation** if programs diversify sizes — fallback: size-class rounding, then VMM (cuMemMap) remapping.
- **cudaLaunchHostFunc callback latency/GIL** — fallback: dedicated polling reaper thread (prior design) which M1/M2 keep working.
- **PressureFit assumes exact runtimes**; real jitter shifts transfer/compute interleaving — runtime is correct regardless (event-driven); throughput sensitivity quantified in M4 via jittered-sim experiments.

## Decisions (confirmed 2026-07-02)

1. **Task backend: torch/Triton first** — port flextrain blocks onto runtime-owned buffers (DLPack + ExternalStream), strict allocator containment; native kernels later for hot ops.
2. **Runtime core: Python-first** over cuda-python, with dispatch-overhead measurement as an explicit M2 gate; Cython/C++/CUDA-graph escape hatches if needed.
3. **Model definitions live in the new repo** (execution-grade); dataflow_sim consumed as a library (costs/policies/recompute planner/verification/webapp export); consistency test vs sim's built-in llama3 builder.
4. **M4 target: Llama3-8B class** with host-offloaded optimizer state — the genuinely memory-constrained headline workload.

## Immediate next steps (implementation order within M0)

1. `git init` + pyproject scaffold (`src/dataflow/{core,runtime,tasks,training,models}`); conda env `dataflow` already prepared; deps: `dataflow_sim` (editable, sibling), `cuda-python`, torch+triton (tasks layer only), pytest.
2. `dataflow.core` IR ported/cleaned from prior `dataflow_runtime/schema.py` (JAX import removed, tensor metadata made first-class) + converters + JSON I/O + validation.
3. Golden-path script: tiny llama3-shaped program → PressureFit → recompute planner → sim run → webapp-JSON export; committed as regression fixture.
4. Then M1 (fake-backend runtime + parity harness) per milestones above.
