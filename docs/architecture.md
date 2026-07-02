# Architecture

Full project plan: [PLAN_V4.md](../PLAN_V4.md). This document is the standing
map of the codebase; per-layer contracts live in each subpackage's README.

## What this project is

A CPU–GPU dataflow runtime realizing the model demonstrated by
[dataflow_sim](https://dataflowsim.sunshein.net/): programs are linear chains
of tasks over named objects; each task declares input / mutated / output
objects; after a task completes, annotated directives fire — **release**
(free fast memory), **offload** (async copy fast→backing, then free),
**prefetch** (async copy backing→fast, admitted only when destination
capacity is reservable). A policy (PressureFit) plus a recompute planner
choose those annotations so that a memory-constrained execution approaches
unconstrained-memory throughput by overlapping transfers with compute.

The first workload is memory-constrained single-GPU DNN training, but the
runtime layers are workload-agnostic.

## Layers

```
src/dataflow/
├── core/        # L0 program IR + validation + JSON + sim converters
├── runtime/     # L1 generic engine (object table, byte ledger, pools,
│                #    dispatcher, transfer engines, trace)
│   └── device/  #    DeviceBackend interface + fake (virtual clock) + cuda
├── tasks/       # L2 executable library (ops -> blocks, composer-planned
│                #    workspaces; torch/Triton first, native later)
├── training/    # L3 lowering, planning via dataflow_sim, profiling,
│                #    gradcheck/testing helpers
└── models/      # L4 declarative model definitions + golden torch references
```

Import rules (enforced by `tests/test_import_boundaries.py`):

- `core` imports nothing heavy (stdlib only at import time; its sim
  converters import `dataflow_sim` lazily inside functions).
- `runtime` never imports torch/jax/`dataflow_sim`; cuda bindings only inside
  `runtime/device/cuda.py`.
- `tasks` is the only layer importing torch/triton; it never imports the sim.
- `training` is the only layer importing `dataflow_sim` (planning,
  verification, webapp export are *consumers* of the sim, never the reverse).

## End-to-end flow

```
ModelDef + TrainingSpec
  → training.lower()             objects (exact sizes) + tasks (block keys,
                                 declared costs, workspace temps) + rewrites
  → dataflow.training.planning   PressureFit + plan_with_recompute (dataflow_sim)
  → annotated core Program       (directives joined back, capacities authoritative)
  → profile pass                 measured runtimes + workspace → re-plan
  → dataflow.runtime.execute     cuda DeviceBackend, real streams/events
  → trace                        compare vs sim EventLog; export webapp JSONs
```

## Simulator semantics the runtime reproduces

From `dataflow_sim.engine.simulator` (the contract for M1/M2 parity gates):

- serial compute chain; a task starts when all inputs are live in fast memory
  AND fast capacity is reservable for its outputs;
- one in-flight transfer per direction with FIFO queues; **destination bytes
  are allocated at transfer start, not enqueue**; a queue head that cannot
  reserve destination capacity blocks (only the queue — never compute
  dispatch) and is retried when bytes free;
- a prefetch requested while the same object's offload is in flight defers
  until the offload completes;
- release requires the fast copy to be live; mutated inputs must be offloaded
  (not released) or their update is lost — planners guarantee this, the
  runtime validates it.

## Status

- **M0 (this milestone): done.** IR, validation, JSON, converters, shaped
  llama3 generator (tiny + 8B), PressureFit + recompute planning integration,
  simulator round-trips, webapp acceptance (in-process FastAPI tests), golden
  path artifacts under `examples/`.
- M1: runtime engine on the fake (virtual-clock) backend; parity gate vs sim.
- M2: cuda backend + synthetic calibrated tasks; real-overlap gate.
- M3: torch/Triton executables + correctness ladder + gradcheck tooling.
- M4: end-to-end memory-constrained multi-step training vs sim prediction.
