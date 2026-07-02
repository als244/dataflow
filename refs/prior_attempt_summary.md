# Project Summary

This document is a restart note for the dataflow runtime project. It is meant
to capture the goal, the architecture we were aiming for, what actually worked,
where the approach got messy, and what a cleaner next pass should preserve or
change.

## Goal

The project goal is to build a runtime system and surrounding scaffolding that
can actually execute programs in the manner suggested by the dataflow simulator:

https://dataflowsim.sunshein.net/

The simulator indicates that large DNN training workloads can fit into a much
smaller fast-memory budget if execution is scheduled carefully: activations and
gradients can be saved, recomputed, released, offloaded, prefetched, and reused
according to an annotated dataflow plan. The real project goal is to take that
from simulation to execution.

The intended stack has three mostly independent layers.

## Layer 1: Generic Dataflow Runtime

The runtime should be isolated from model details. It should accept an annotated
dataflow program and execute it. It should not know about Llama, Qwen,
transformer blocks, attention, MLPs, or gradient accumulation.

Its contract is object/task based:

- objects have ids, sizes, locations, versions, and optional tensor metadata,
- tasks have inputs, outputs, mutated objects, and runtime/task bindings,
- movement directives include `release_after`, `offload_after`, and
  `prefetch_after`,
- runtime state tracks fast/backing copies, pending transfers, readiness, dirty
  state, and object versions,
- the runtime owns dependency handling, memory reservations, buffer reuse,
  streams, events, and transfer ordering.

In the current repo this is mostly represented by:

- `dataflow_runtime.schema`
- `dataflow_runtime.planning`
- `dataflow_runtime.async_runtime`
- `dataflow_runtime.pointer_types`
- `dataflow_runtime.backends.fake`
- `dataflow_runtime.backends.cuda_runtime`

The important architecture point is that this runtime should consume an already
annotated plan. It should not be responsible for DNN-specific lowering or
choosing which transformer layers are recomputed.

## Layer 2: Planning And DNN Training Lowering

Above the runtime, we need a compiler/lowering layer for DNN training. This
layer should take:

- a model definition,
- a training spec,
- shape/dtype information,
- cost and memory estimation/profiling options,
- checkpoint/recompute choices,
- hardware assumptions.

It should produce two things.

First, it should produce a generic dataflow program with object sizes and task
costs. That program should be uploadable to the simulator webapp and should also
be usable locally with `dataflow_sim`.

Second, it should produce task execution bindings for each unique compute block:
embedding forward/backward, transformer block forward/recompute/backward,
LM-head forward/backward, optimizer tasks, and so on. The runtime can then
dispatch those bindings when the annotated plan says a task is ready.

In the current repo, pieces of this are represented by:

- `dnn_training.lowering`
- `dnn_training.models.llama3`
- `dnn_training.backends.llama_tasks`
- `dataflow_runtime.analysis`
- `dataflow_runtime.profiling`
- `dataflow_runtime.sim_export`
- `tools/sim_custom_upload_probe.py`
- `tools/llama3_budget_sweep.py`

The lowering layer should remain separate from the runtime. The runtime should
not import Llama model code.

## Layer 3: Model Definitions

The final layer is a set of model-family definitions that are simple to extend.
The first target was Llama3. Future targets could include Qwen3 and MoE
families.

The model definition should describe architecture, parameter groups, activation
boundaries, and compute blocks. It should not bury generic recompute or runtime
logic inside model-family code.

For a restart, a clean model file should say something like:

- here are the parameter objects,
- here are the forward blocks,
- here are the backward blocks,
- here are block-local cache/recompute boundaries,
- here are the task binding factories for this backend.

Generic training lowering should turn that into a concrete dataflow program.

## What Worked

### Dataflow Program Export

Creating dataflow programs with real object sizes, task costs, and simulator
metadata mostly worked.

The exported files:

- `llama3_8b_s4096_b1_16gib_roofline.dataflow.json`
- `llama3_8b_s4096_b1_16gib_profiled.dataflow.json`

represent a full Llama3-8B-shaped case with `seq_len=4096`, `batch=1`, and a
16 GiB fast-memory budget. They include object sizes, task costs, metadata,
hardware assumptions, and simulator-readable task structure.

We also added cost plumbing:

- JAX `memory_analysis()` for input/output/temp/alias bytes,
- JAX `cost_analysis()` where available,
- roofline runtime estimates,
- optional profiling of unique JAX task shapes,
- cost reconciliation for missing matmul/attention accounting.

This part is close to the desired compiler/scaffolding shape.

### PressureFit And Recompute Integration

We connected generic plans to `dataflow_sim` for:

- validation,
- PressureFit annotation,
- recompute planning,
- simulator preview/simulation paths,
- exported webapp-compatible JSON.

The DNN lowering can produce concrete variants such as save-all, recompute-all,
and mixed policies. Longer term, the policy/planner should choose the
checkpoint/recompute decisions rather than having the user manually select
`save_all` or `recompute_all`.

### JAX Was Useful For Correctness

JAX was very useful for quickly creating isolated forward/backward compute
blocks.

We used local `jax.vjp`-based functions so that a task like
`transformer_block.backward` could be expressed without manually deriving every
primitive gradient. This let us build and validate the dataflow task graph
quickly.

Gradient correctness evidence is now fairly strong for the Python/JAX-bound
path:

- tiny matrix across save-all, recompute-all, mixed recompute, gradient
  accumulation, `jax`, and `jax-ref`,
- BF16/cuDNN probes with real Llama3 hidden/FFN/head dimensions,
- six Llama3-8B-shaped layers at `seq_len=1024` for recompute-all,
- six Llama3-8B-shaped layers at `seq_len=1024` for save-all,
- six Llama3-8B-shaped layers with two gradient accumulation rounds at
  `seq_len=512`.

Those compare against:

```text
jax.jit(jax.value_and_grad(reference_loss))(params, token_ids, target_ids)
```

See `docs/llama3_gradient_parity_matrix.md`.

### JAX-Ref Full-Shape Execution

The full Llama3-8B-shaped plan can execute through the JAX-ref bridge:

- 32 layers,
- BF16,
- cuDNN attention,
- `seq_len=4096`,
- `batch=1`,
- 16 GiB logical fast-memory budget,
- no full-reference gradient comparison.

The stable full-shape JAX-ref setting is:

```text
--runtime jax-ref
--value-init zeros
--no-compare-reference
--max-ref-pool-gb 0
--reuse-initial-values
--persist-initial-objects
--persist-initial-locations backing
```

Measured ready time is around `2.71s`. The profiled simulator expectation is
around `1.30s`, so this path is about `2.08x` slower than the simulator.

This was useful because it proved the full annotated task graph can be replayed
with real JAX math, but it is not the final high-throughput runtime.

### Generic Pointer Scheduler Matches The Simulator

The most successful runtime result is the synthetic pointer-runtime parity
check.

`AsyncPointerDataflowRuntime` can now execute the annotated full 8B simulator
chain using synthetic no-op task bindings and match simulator timing and memory
nearly exactly:

```text
profiled sim:
  pointer_synthetic=1301576.650us
  simulator=1301577.411us
  peak_fast=15.97 GiB

roofline sim:
  pointer_synthetic=1055159.255us
  simulator=1055160.017us
  peak_fast=15.97 GiB
```

This is important: the generic scheduler, object tracking, release/offload/
prefetch semantics, transfer timing, and buffer reuse model can reproduce the
simulator when compute is synthetic.

See:

- `tools/run_pointer_synthetic_plan.py`
- `artifacts/llama3_8b_s4096_b1_16gib_pointer_synthetic_vs_sim_profiled.json`
- `artifacts/llama3_8b_s4096_b1_16gib_pointer_synthetic_vs_sim_roofline.json`

This narrows the remaining problem: scheduler semantics are no longer the
primary gap. The gap is real task bindings that operate on runtime-owned
buffers.

## What Did Not Work

### JAX Is Not A Good Backend For Runtime-Owned Memory

JAX was excellent for defining math and checking gradients, but it was a poor
fit for the runtime we actually want.

The desired runtime wants to own:

- device buffers,
- pinned host buffers,
- compute/H2D/D2H streams,
- CUDA events,
- transfer queues,
- exact buffer reuse,
- release timing,
- prefetch timing,
- memory reservations.

Public JAX APIs do not let arbitrary `jax.jit` tasks run on a runtime-owned CUDA
stream. JAX/XLA owns dispatch, streams, internal workspaces, temporary buffers,
autotuning, and output allocation.

That creates several concrete problems.

### JAX Arrays Are The Canonical Buffers

In the JAX-bound path, the canonical runtime object is ultimately a JAX array or
JAX ref. That means the runtime cannot truly say:

```text
object A and object B are assigned to the same physical fast-memory buffer
because their lifetimes do not overlap.
```

We can approximate this with JAX refs, but the physical allocation is still
JAX-owned. Retained refs continue to exert pressure on the JAX allocator and on
XLA workspaces.

Planned JAX-ref reuse proved this. With 2 GiB headroom, the full shape completed
and reused `69/135` output refs. It reduced `reserve_outputs` from about
`0.76s` to about `0.15s`, but it increased block backward task time and raised
allocator peak. End-to-end time did not improve.

So planned JAX-ref reuse is logically valid, but it is not equivalent to
runtime-owned pointer reuse.

### JAX Stream Control Is Too Limited

The simulator assumes overlap between compute, D2H, and H2D streams. The
runtime should enqueue D2H after producer compute, prefetch after backing is
available and fast capacity is reserved, and allow compute to proceed when its
inputs are ready.

With arbitrary JAX tasks, we cannot schedule the kernel on our own compute
stream. We can wait on JAX futures and then fire triggers, but this is still not
the simulator's owned-stream model.

We tried an async JAX-ref dispatch mode that skipped per-task
`block_until_ready` and retained released refs as pending until futures were
ready. Small correctness probes passed. Full 8B-shaped runs failed:

- unbounded async dispatch OOMed at the final readiness barrier,
- window 4 OOMed during XLA loss/reduction autotuning,
- window 1 OOMed during XLA/Triton GEMM autotuning,
- trying `TF_GPU_ALLOCATOR=cuda_malloc_async` did not fix it in this setup.

This suggests that the conservative JAX-ref per-task barrier is not merely
needless overhead. It is also preventing too many JAX futures, workspaces, and
temporaries from piling up.

### JAX Autotuning And Workspaces Distort Memory Planning

The simulator reasons about logical object lifetimes and transfer timings. JAX
adds hidden physical memory pressure:

- XLA workspaces,
- autotuning allocations,
- compiler-chosen Triton/cuBLASLt strategies,
- temporary arrays,
- allocator fragmentation,
- output placement.

Several experiments failed with OOMs trying to allocate roughly `1000 MiB`
temporaries during GEMM or log-softmax/reduction autotuning. These allocations
are not first-class objects in our dataflow plan.

We added task-local temp objects from JAX memory analysis, but hidden workspace
pressure still matters.

### JAX Cost Analysis Was Useful But Incomplete

JAX cost analysis did not always count the expensive operations we cared about,
especially around custom calls, Triton GEMMs, cuBLASLt dispatch, and attention.
We added reconciliation logic and profiling support, but this is a lesson for a
restart: cost estimation should be treated as a model with validation, not a
truth source.

The best current approach is:

- use JAX analysis for object/temp sizes where helpful,
- use explicit analytical FLOP/byte reconciliation for matmul and attention,
- use roofline estimates as a first pass,
- optionally profile unique compute blocks,
- store both estimate and profile metadata in exported dataflow programs.

### The Codebase Became Too Split-Brained

Some of the early work mixed these concerns:

- JAX demos,
- DNN lowering,
- runtime execution,
- native pointer probes,
- simulator export,
- Llama-specific logic,
- compatibility shims.

The direction is now clearer, but a restart should keep the package boundaries
much sharper:

- `dataflow_runtime`: generic runtime only,
- `dnn_training`: generic DNN lowering only,
- `models`: model-family definitions only,
- `backends`: task execution bindings for JAX reference, native CUDA, etc.,
- `tools`: export/profile/probe scripts.

Avoid putting Llama task math inside the generic runtime.

## Current Status

The honest status is:

```text
Full 8B high-throughput real-math runtime: not done.
```

But several prerequisites are in place:

- full 8B dataflow programs can be generated and uploaded,
- PressureFit/recompute integration works,
- JAX-bound task math has strong gradient correctness evidence,
- full 8B JAX-ref execution works but is slower than simulator,
- generic pointer scheduler matches simulator for full 8B synthetic tasks,
- native CUDA pointer probes exist for several subcomponents.

The remaining work is to replace synthetic no-op pointer tasks with real
runtime-owned task bindings and prove correctness/performance end to end.

## Recommended Restart Plan

### 1. Keep The Runtime Generic

The runtime should only consume:

- annotated plan,
- object metadata,
- task bindings,
- initial buffers.

It should produce:

- final object handles,
- event trace,
- memory trace,
- timing metrics,
- validation errors if directives are invalid.

It should not know about model architecture.

### 2. Treat JAX As Reference, Not Runtime

Use JAX for:

- reference gradients,
- shape checking,
- quick task math prototypes,
- memory/cost hints,
- profiling experiments.

Do not rely on JAX for:

- runtime-owned stream execution,
- exact buffer reuse,
- reliable low-memory overlap,
- final high-throughput execution.

### 3. Build Pointer-Native Task Bindings Incrementally

The next real milestone is not another JAX scheduling trick. It is native
pointer task coverage.

A reasonable order is:

1. token embedding forward/backward,
2. LM head forward/backward,
3. MLP/FFN forward/backward,
4. RMSNorm forward/backward,
5. attention prelude,
6. flash/cudnn attention binding,
7. full transformer block forward/backward,
8. gradient accumulation and optimizer tasks,
9. full 8B run with real pointer tasks.

Each native task should have:

- pointer-runtime binding,
- isolated correctness test against JAX,
- memory footprint check,
- CUDA event timing,
- integration into a small annotated plan.

### 4. Keep The Simulator Parity Harness

`tools/run_pointer_synthetic_plan.py` is valuable. It verifies the runtime
scheduler against the same annotated chain that the simulator emits. Keep it as
a regression gate.

For real native tasks, add a second harness:

```text
annotated plan + real pointer bindings -> runtime trace
```

Then compare:

- makespan,
- stream utilization,
- peak fast bytes,
- transfer counts,
- reuse counts,
- event ordering,
- final gradients.

### 5. Make Model Definitions Boring

The model-family file should be declarative and minimal. Generic lowering
should handle recompute, checkpointing, object creation, and task-chain
construction.

A good model definition should expose:

- config,
- parameter specs,
- activation boundary specs,
- ordered blocks,
- compute block keys,
- binding factories.

The lowering layer should turn this into a dataflow program and bindings.

### 6. Separate Three Kinds Of Success

Do not conflate these:

1. **Compiler success**: can generate a good dataflow program with object sizes
   and runtimes.
2. **Scheduler success**: can execute the annotated plan with correct movement,
   memory, overlap, and buffer reuse.
3. **Training success**: can run real model math with correct gradients and high
   throughput.

Right now:

- compiler success is mostly achieved for Llama3,
- scheduler success is achieved synthetically,
- training success is achieved only in the JAX bridge, not in the final
  high-throughput pointer runtime.

## Key Artifacts

- `llama3_8b_s4096_b1_16gib_roofline.dataflow.json`
- `llama3_8b_s4096_b1_16gib_profiled.dataflow.json`
- `llama3_8b_s4096_b1_16gib_roofline.dataflow.simulation.json`
- `llama3_8b_s4096_b1_16gib_profiled.dataflow.simulation.json`
- `artifacts/llama3_8b_s4096_b1_16gib_jax_ref_persist_backing.metrics.json`
- `artifacts/llama3_8b_s4096_b1_16gib_pointer_synthetic_vs_sim_profiled.json`
- `artifacts/llama3_8b_s4096_b1_16gib_pointer_synthetic_vs_sim_roofline.json`
- `docs/full_8b_execution_probe.md`
- `docs/llama3_gradient_parity_matrix.md`
- `docs/native_cuda_overlap_probe.md`
- `docs/native_pointer_math_probe.md`

## Bottom Line

The project clarified the architecture:

```text
model definition
  -> DNN training lowering
  -> dataflow program with sizes/costs
  -> dataflow_sim annotation/recompute
  -> generic runtime execution
  -> task bindings for a backend
```

It also clarified the main lesson:

```text
JAX is excellent for reference math and prototyping isolated task functions,
but it is not enough to implement the final simulator-style runtime because it
does not give us ownership of streams, buffers, hidden workspaces, and exact
memory reuse.
```

The next clean attempt should keep JAX as the reference backend and make the
pointer runtime the real execution target.
