# Dataflow Runtime — Project Plan (v4)

Supersedes [PLAN_V3.md](PLAN_V3.md) (and V2/INITIAL). New in v4: (a) explicit **verification section** — correctness measured against known references (golden plain-torch model + derived per-task references) through a parity ladder, async/buffer bugs hunted via a plan-invariance axiom, poison-on-free, and interleaving stress; performance measured against both the measured-cost simulator sweep and an unconstrained plain-torch baseline with named-cause gap attribution; (b) **general gradient-checking helpers** (`dataflow.training.testing`) that verify backward math for any op, module, or composed bwd task — backward functions are error-prone and get first-class tooling; (c) **documentation requirements** — per-layer contract docs plus an extending guide (custom ops/modules/models → lowering → dataflow tasks), delivered incrementally with each milestone. (v3 retained: composer-compiled task workspaces. v2 retained: fresh code only, DeviceBackend abstraction, measurement-authoritative costs, explicit memory accounting, multi-step training, linear-chain DAG-ready IR.)

## Context

Goal: realize the [dataflow simulator](https://dataflowsim.sunshein.net/) as a true CPU–GPU runtime: a generic engine that ingests annotated dataflow programs (tasks with input/mutated/output objects + release/offload/prefetch directives), owns GPU/pinned-host memory, streams/events and transfers, and dispatches task executables — plus a DNN-training lowering layer that turns model definitions into such programs, annotated by dataflow_sim's PressureFit policy + recompute planner, with end-to-end multi-step training throughput matching simulator predictions.

Hardware: RTX 5090 (32 GB, CUDA 13.1), i9-13900KF, 188 GB RAM, PCIe x16. Env: conda `dataflow` (py3.12) with `dataflow_sim` installed editable (done).

## Design principles

1. **References are references.** `refs/prior_attempt` and `refs/flextrain` contribute *concepts*, not code. Everything is written fresh in this repo, fitted together, with a minimal public surface per layer. Concepts worth absorbing: the prior attempt's object-slot state machine, `launch(ctx)` binding shape, and sim-parity harness; flextrain's declarative activation schema + stateless composable blocks; both repos' documented failure modes (split-brained layers, N code paths agreeing by convention, embed/head special-casing).
2. **Device abstraction.** The runtime never calls CUDA directly; it talks to a small `DeviceBackend` interface whose surface is restricted to the CUDA∩HIP common subset, so an AMD backend later is mechanical. Only `cuda` (+ virtual-clock `fake`) implemented now.
3. **Measurement over estimation.** Object sizes are exact by construction. FLOPs/bytes are declared next to op implementations and test-validated. Task workspaces are compiled from implementation structure and calibrated/validated by measurement. Task runtimes are measured per unique task by a profiling harness; measurements are authoritative and written back into programs before final planning. The simulator consumes costs; it is never their source.
4. **Every result is checked against a reference.** Math correctness against a golden plain-PyTorch model and derived per-task references (generic gradcheck helpers); scheduler behavior against the simulator; throughput against both the measured-cost simulator prediction and an unconstrained plain-torch baseline. See Verification.
5. **Documentation is part of the contract.** Every layer ships a contract doc (purpose, assumptions, relationships, API surface); an extending guide covers custom ops/modules/models and how lowering turns them into dataflow tasks. Docs are milestone deliverables, not afterthoughts.
6. **Linear chains now, DAG-ready IR.** The dispatcher executes the task list in order (the sim's and policies' model). The IR derives dependencies from object producer/consumer relations rather than baking in "position = dependency", so a DAG dispatcher can arrive later (multi-GPU era) without a program-format change.
7. **The runtime is policy-agnostic.** It consumes an annotated program; PressureFit/recompute planning is an isolated, swappable step owned by the training layer via dataflow_sim.

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
├── tasks/       # L2 — executable library: ops → blocks; stateless launch(ctx); each op carries
│                #      impl + pure-torch reference + declared flops/bytes + op_scratch; the composer
│                #      compiles task workspace layouts; torch/Triton backend first, native later.
├── training/    # L3 — lowering: ModelDef + TrainingSpec → core program (+ RecomputeRewrites
│                #      + executable factory table keyed by compute_block_key); planning step calls
│                #      dataflow_sim (PressureFit + plan_with_recompute); profiling harness;
│                #      testing/ subpackage: generic gradcheck + parity + reference harness.
├── models/      # L4 — declarative model definitions (llama3 first) + golden torch reference models.
tools/           # parity/profile/export/bench/run scripts
tests/
docs/            # per-layer contract docs + extending guide (see Documentation)
```

Dependency rule: `core ← runtime`, `core ← tasks`, `core+tasks ← training ← models`; only `training` imports `dataflow_sim`; only `tasks` imports torch/triton; only `runtime/device/cuda.py` imports cuda-python.

### DeviceBackend (L1 boundary to vendor runtimes)

~12 calls, all with direct HIP equivalents: `create_stream`, `record_event`, `stream_wait_event`, `query_event`, `sync_event` (shutdown/error paths only), `alloc_device`, `alloc_pinned`, `free_*` (setup/teardown only), `memcpy_async`, `launch_host_callback`, `event_elapsed_time`, `device_info`. Implementations: `fake` (virtual clocks — drives sim-parity gates and CI-without-GPU), `cuda` (cuda-python). Executables receive an opaque stream handle; torch/Triton executables wrap it (`torch.cuda.ExternalStream`) and are themselves ROCm-portable; native executables are per-vendor plugins.

### End-to-end flow

```
models.llama3 (ModelDef) + TrainingSpec
  → training.lower(): objects (exact sizes) + tasks (block keys, params, declared costs,
                      composer-computed workspace temps) + RecomputeRewrites + executable factories
  → bare chain → plan_with_recompute(rebuild_variant, rewrites, apply_pressurefit_policy)
  → annotated chain → core program (tensor/binding metadata joined back)
  → [profile pass: measured runtimes + workspace validation → re-plan]   (mandatory before headline runs)
  → runtime.execute(program, executables, initial_buffers)               (cuda backend)
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

**Each op carries four things beside its kernel implementation**: a pure-torch eager *reference* function (used for parity + derived gradchecks, see Verification), declared *flops/bytes*, declared intermediates' shapes, and private *op_scratch(shapes)*. This co-location is what keeps costs, references, and math from drifting apart.

**Workspace is compiled, not declared.** Tasks are higher-order than ops (module / combination of modules), so peak scratch accrues in the *inter-op intermediates* (q/k/v between prelude and attention, x1/x3 between projections and swiglu, normed activations, …) and depends on implementation efficiency — per-op declarations alone cannot see it. Resolution:
- a small static planner (the **composer**) runs liveness over a task implementation's op sequence → `peak = max over steps of (live intermediates + op_scratch)` → and emits an **offset layout** for one task-workspace arena (greedy interval packing; the outer memory problem one level down, but trivial: serial, all-resident, no transfers);
- the executable *executes against that same layout* — intermediates are views at the planned offsets into the workspace buffer the runtime provides — so declared == actual **by construction**; a fused implementation has a different op list and gets a smaller computed workspace automatically;
- workspace is a property of the chosen *implementation*, exposed as `ExecutableFactory.workspace_bytes(params, shapes)`; lowering calls it to emit each task's `temp` object so the policy sees per-task workspace pressure;
- opaque library scratch: cuBLASLt takes an explicit workspace pointer/size → a slice of our arena (fully owned); torch-internal allocations (flash-attn lse/rng state, etc.) are *measured per shape-class* and folded into that op's `op_scratch` calibration;
- the profiling harness still measures true per-unique-task high-water; divergence beyond the calibrated residual fails loudly. Chain of trust: computed layout (exact for intermediates) + measured calibration (opaque scratch) + measured validation (backstop).
- **Granularity dial**: a task whose computed workspace is large is a quantified signal to split it — promoting intermediates into first-class program objects the outer planner can release/offload between the halves. Lowering chooses the decomposition; the composer prices each choice. (Connects to the sim README's finer-grained-decomposition TODO.)

Torch/Triton backend discipline:
- Zero-copy tensor views over runtime buffers via DLPack; ops under `torch.cuda.ExternalStream(runtime_stream)`; all block outputs written into runtime-provided buffers (`out=` / slot-style).
- A small bounded torch-allocator scratch lane (pre-warmed, accounted inside the fast budget) absorbs only the calibrated opaque residual. `torch.compile` is not used in v1 executables (no public workspace introspection; cudagraph pools break external-buffer ownership) — eager + Triton + explicit kernels; revisit later.
- Assert-no-alloc test mode (allocator stats before/after each task) keeps steady state cudaMalloc-free.
- Native backend (cuBLASLt + custom kernels, concepts proven by prior probes) grown for hot ops later; per-vendor plugins.

## Lowering + costs (L3/L4)

- ModelDef: declarative composition (flextrain-style) — param specs, block sequence, per-block saved-context schema (fields/tiers → packed `A_*` layouts and `RecomputeOption`s), block keys + params; embed/head first-class.
- Lowering emits step/round/layer tasks (sim-compatible naming `W_i, A_s_r_i, dW_s_i, O_i`), grad-accum via round-0-fresh / later-rounds-mutate, optimizer tasks, `RecomputeRewrite`s, and `rebuild_variant(levels)` for the recompute planner. Multi-step-invariant by construction.
- Cost system (measurement over estimation):
  - sizes exact (shape × dtype), test-asserted vs real tensors;
  - per-op declared flops/bytes, test-validated with `torch.utils.flop_counter.FlopCounterMode`;
  - task workspace from the composer (`workspace_bytes(params, shapes)`) emitted as temp objects;
  - first plan from a small owned roofline util (declared costs × hardware spec table);
  - **mandatory profile pass**: per unique `(block_key, shape-sig)` — CUDA-event-timed runtime + workspace validation, written back into program metadata (estimate and measurement both kept); final planning on measured costs;
  - webapp export carries declared flops/bytes subops so the webapp can re-resolve for any hardware.

## Verification: correctness & performance references (drives M3/M4 gates)

**Two kinds of math reference:**
- *Derived per-task references (automatic).* Because every task implementation is an explicit op sequence and every op carries a pure-torch reference, the composer can assemble an autograd-able reference forward for **any** composed task automatically. Autograd through it yields reference gradients for that task — so backward checking generalizes to arbitrary ops, modules, and compositions without hand-written per-block references.
- *Independent golden model (per family).* Each model definition ships a plain single-stream eager-PyTorch implementation (same init, same token stream): `loss = ref_model(x); loss.backward(); opt.step()`. Being hand-written and independent, it also catches errors in the composition graph itself. (Optional secondary cross-check vs JAX.)

**Gradient-checking helpers (`dataflow.training.testing`)** — general, public, and used for every builtin and every future custom op/module (backward math is the most error-prone code in the project; it gets first-class tooling):
- `check_op_backward(op, shapes, dtypes, *, seed, tol)` — analytic backward vs autograd on the op's reference; fp64/fp32 finite-difference mode (`torch.autograd.gradcheck`) at tiny shapes; bf16 tolerance mode at real shapes.
- `check_task_backward(fwd_exec, bwd_exec, shapes, *, seed, tol)` — runs the pair through the standard executable contract: forward (saving `A_*` context), backward from a random upstream grad; compares every produced gradient (inputs and params) against autograd through the derived reference. Also validates: saved-context pack/unpack round-trip, recompute-path equivalence (`recompute + bwd ≡ save + bwd`), and gradient-accumulation semantics (`accum=True` adds onto the prior value; mutation versioning respected).
- `check_model_step(modeldef, training_spec, *, steps, tol)` — full program vs golden model: loss, every `dW`, updated params + optimizer state after 1 and N steps.
Helpers operate purely through the executable/ModelDef contracts, so a user-defined op or module gets the same coverage by construction; the extending guide makes running them a required step.

**Correctness ladder** — all levels must pass before any headline run:
1. *Op backward*: `check_op_backward` for every op (builtin or custom).
2. *Block/task backward*: `check_task_backward` for every fwd/bwd/recompute task family.
3. *Single-step end-to-end*: `check_model_step(steps=1)` through the real runtime (cuda backend) vs golden model.
4. *Multi-step end-to-end*: `check_model_step(steps=N)` — loss trajectory + final params/opt-state (fp32 mode at small scale for tight bounds; bf16 at real scale with pre-registered tolerances).

**Async/buffer-bug detection** — targets scheduling/lifetime bugs that math tests cannot see:
- **Plan-invariance axiom**: annotations are semantically neutral. The same program under *different* plans (save-all, recompute-heavy, different budgets, different inbound schedules) and different backends must produce identical gradients — bitwise for fixed seeds where kernels are deterministic, else within dtype-noise bounds. Any divergence is a lifetime/movement/reuse bug by definition. The single highest-leverage system test; runs at every scale.
- *Determinism*: same-seed runs repeated → bitwise-identical outputs (catches races).
- *Interleaving stress*: injected delays/jitter on compute/h2d/d2h streams must not change results (event-ordering correctness under arbitrary interleavings; catches missing waits).
- *Poison-on-free debug mode*: freed buffers filled with NaN patterns at retirement + guard regions around buffers — use-after-release and overruns surface immediately instead of as silent corruption.
- *Always-on assertions* (debug runs): object-version checks, slot state-machine transitions, ledger-vs-pool consistency.

**Performance references (M4)** — two axes, both required:
- *Vs simulator*: reproduce the webapp's throughput-vs-fast-budget sweep with **real runs** overlaid on measured-cost sim curves. Target: ≥90% of sim-predicted tokens/s at generous budgets; at tight budgets, quantified gap attribution (host dispatch latency, achieved PCIe bandwidth vs assumed, kernel-time jitter, transfer-admission lag) — every percentage point of gap gets a named cause.
- *Vs plain torch*: at unconstrained memory, the runtime must ≈ the plain eager-torch baseline (grounds absolute throughput — no self-referential victory). The headline claim is then **throughput retained as the budget shrinks**: tokens/s across the sweep vs sim, with stream-utilization/overlap evidence from traces + nsys.

## Documentation (milestone deliverables, not afterthoughts)

- **Per-layer contract docs** (`docs/` + a README at each subpackage root): purpose, assumed contracts and invariants, relationship to adjacent layers, and the public API surface — for `core` (IR + converters), `runtime` (engine semantics, ledger rules, trace format), `runtime/device` (DeviceBackend interface + how to add a vendor), `tasks` (executable contract, op anatomy: impl/reference/costs/scratch, the composer), `training` (lowering, planning integration, profiling, testing helpers), `models` (ModelDef format).
- **Extending guide** (`docs/extending.md`): the walkthrough we ourselves follow when adding builtin families — how to write a custom op (impl + reference + costs + scratch, run `check_op_backward`), compose a module/block (composer, `check_task_backward`), define a model (ModelDef + golden reference, `check_model_step`), and what lowering does with each piece to produce dataflow tasks/objects/rewrites. Reuse of existing ops/blocks across families is the explicit design goal.
- Docs land with the milestone that introduces their subject (M0: core + architecture overview; M1/M2: runtime + device; M3: tasks + training + extending guide; M4: end-to-end usage walkthrough) and are updated in the same PR as any contract change.

## Milestones (each with a hard gate)

- **M0 — scaffold + IR + sim interop.** Package layout, `core` IR/validation/JSON, converters, import-boundary test. Gate: tiny + 8B-shaped llama3 programs round-trip core↔sim; PressureFit + recompute produce annotated chains; sim runs them; export accepted by local webapp (`POST /api/simulate`, source:"schema"). Docs: architecture overview + core contract.
- **M1 — runtime parity, virtual.** Full engine on the fake backend (ledger, transfer engines, directives, non-blocking admission). Gate: makespan/peak-fast/event-order parity vs sim EventLog on the full 8B annotated chain, plus adversarial unit chains (deferred prefetch, blocked queue head, mutation offload, re-prefetch, deadlock detection). Docs: runtime contract.
- **M2 — runtime real, synthetic tasks.** `cuda` DeviceBackend; calibrated spin-kernel executables + real memcpys on a scaled-to-32GB program. Gate: on-GPU trace vs sim within a few % makespan; overlap visible in nsys; zero implicit syncs (nsys audit); measured host dispatch overhead ≪ mean task runtime. Docs: DeviceBackend contract.
- **M3 — real executables (torch/Triton) + correctness ladder + gradcheck tooling.** Blocks: embed, rmsnorm, GQA attention, SwiGLU, head+loss, adamw, block fwd/bwd/recompute compositions built through the composer; DLPack/ExternalStream plumbing; no-alloc discipline; profiling harness (runtime + workspace calibration/validation); golden torch reference model; **`dataflow.training.testing` helpers implemented and applied to every builtin op/block**. Gate: correctness ladder levels 1–3 pass via the helpers; plan-invariance across ≥3 distinct plans at small scale; poison-on-free + interleaving-stress runs clean; measured-cost programs re-planned; trace sim-consistent at small scale. Docs: tasks + training contracts + extending guide.
- **M4 — end-to-end memory-constrained multi-step training + performance verification.** Llama3-8B-class on the 5090 across a fast-budget sweep (bf16 params+grads exceed VRAM → genuine weight/grad offload + recompute + Adam-state-on-host). Gate: correctness ladder level 4 (N-step parity vs golden model at feasible scale) + plan-invariance at full scale; loss decreases over N steps with persistent state; **performance**: real-run sweep overlaid on measured-cost sim curves with ≥90% of prediction at generous budgets and named-cause gap attribution at tight budgets; ≈ plain-torch baseline at unconstrained memory; program + measured trace uploadable to webapp. Docs: end-to-end usage walkthrough.
- **M5 — stretch.** Native executables for hot ops; CUDA-graph capture of the steady-state step; Qwen3 dense/MoE (exercising the extending guide + op reuse); sim-repo enhancement (schema carries recompute rewrites); AMD backend when hardware exists; multi-GPU groundwork.

## Risks / watch-items

- Torch hidden allocations re-creating the JAX problem → composer-owned workspaces + calibrated opaque scratch + no-alloc assertions + bounded lane; fallback native executables for offenders.
- Python dispatch overhead → measured at M2; fallbacks: batch pre-enqueue, Cython/C++ hot loop, CUDA graphs.
- Exact-size pooling fragmentation on diverse programs → size-class rounding, then VMM remap.
- Host-callback latency/GIL → polling-reaper fallback maintained from M1.
- PressureFit plans assume exact runtimes; real jitter shifts interleavings → correctness unaffected (event-driven, proven by interleaving-stress tests); throughput sensitivity quantified in M4 (jittered-sim experiment).
- Nondeterministic kernels (atomics in some backward ops) weaken bitwise plan-invariance → prefer deterministic implementations where cheap; otherwise pre-registered statistical bounds.

## Decision log

- 2026-07-02 (v1): torch/Triton-first executables; Python-first runtime core; model definitions live in this repo (sim consumed as a library for policies/planning/verification/webapp); M4 targets Llama3-8B-class.
- 2026-07-02 (v2): fresh code only, references for concepts; DeviceBackend abstraction (CUDA∩HIP surface, cuda-only impl for now); costs/sizes owned by this repo — declared per-op, validated by FlopCounterMode, runtimes+workspace measured per unique task and authoritative; hybrid temp policy; host-authoritative byte ledger mirroring sim states with three admission points; multi-step training from the start; linear-chain dispatch with DAG-ready IR.
- 2026-07-02 (v3): task workspace compiled by the composer (liveness + offset layout over each task implementation's op sequence; declared == actual by construction); opaque library scratch calibrated per shape-class; per-unique-task measurement as validation backstop; `ExecutableFactory.workspace_bytes(params, shapes)` emitted as per-task temp objects; task-granularity dial.
- 2026-07-02 (v4): verification harness — golden plain-torch model per family + composer-derived per-task references; general gradcheck helpers in `dataflow.training.testing` (`check_op_backward`, `check_task_backward`, `check_model_step`) applied to all builtin and custom ops/modules; correctness ladder (op → block → single-step → multi-step); plan-invariance axiom as flagship async/buffer-bug detector; poison-on-free + guard regions + interleaving stress + determinism checks; performance vs measured-cost sim sweep (≥90% at generous budgets, named-cause gap attribution) and vs unconstrained plain-torch baseline; per-layer contract docs + extending guide as milestone deliverables.

## Immediate next steps (M0)

1. `git init` + pyproject scaffold; deps: `dataflow_sim` (editable sibling), `cuda-python`, torch+triton (tasks layer), pytest.
2. `dataflow.core` IR + validation + JSON + sim converters (fresh implementation).
3. Golden-path script: tiny llama3-shaped program → PressureFit → recompute → sim run → webapp JSON; committed as regression fixture.
4. M1 fake-backend engine + parity harness.
