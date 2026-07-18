# Architecture

The standing map of the codebase; per-layer contracts live in each
package's README, and the workload<->engine seam is specified in
[program_contract.md](program_contract.md).

## What this project is

A CPU–GPU dataflow runtime realizing the model demonstrated by the
webapp simulator: programs are linear chains of tasks over named
objects; each task declares input / mutated / output objects; after a
task completes, annotated directives fire — **release** (free fast
memory), **offload** (async copy fast→backing, then free),
**prefetch** (async copy backing→fast, admitted only when destination
capacity is reservable). A policy (PressureFit) plus a recompute
planner choose those annotations so that a memory-constrained
execution approaches unconstrained-memory throughput by overlapping
transfers with compute.

The first workload is memory-constrained DNN training (single-GPU and
small fleets), but the engine layers are workload-agnostic.

## The three universes

```
src/dataflow/                ENGINE — executes programs, no model vocabulary
├── core/                    L0 program IR + validation + JSON + sim converters
├── runtime/                 L1 generic engine (object table, ledger, pools,
│   └── device/                 dispatcher, transfers, trace; DeviceBackend
│                               interface + fake (virtual clock) + cuda)
└── service/                 the persistent daemon (dataflowd): store slab,
                             wire protocol, runs, snapshots, peers, and the
                             RESOLVER REGISTRY (the one workload seam)

src/dataflow_training/       WORKLOAD — builds programs, registers resolvers
├── model_families/          one package per family (model/blocks/bridge/
│                            presets) + the family registry + twin bridges
├── blocks/                  shared executables, packed layouts, optimizers,
│                            moe/dsa/mla modules, adamw comm variants
├── kernels/                 registry op implementations (eager/triton/…)
├── lowering/                shaped programs, exact sizes, freeze plans,
│                            PressureFit planning (the dataflow_sim consumer)
├── data/                    segments (varlen), fineweb stream, packing
├── run/                     drivers (reference + engine service), presets,
│                            recipes, profiling
├── distributed/             topology, fleet conductor, sharding
└── testing/                 gradcheck harnesses

reference_models/            TRUTH — isolated pure-torch twins (repo root,
                             torch-only, no dataflow imports, no cross-
                             imports; the per-family equivalence bar)

tools/                       CLIs over both packages (dataflowd, train_solo,
tests/                       train_fleet, bench_frontier, verify_family, …);
                             tests mirror the split (tests/dataflow/,
                             tests/dataflow_training/, tests/reference_models/)
```

Tools and tests sit OUTSIDE all three packages and may cross the seam;
the packages themselves may not.

## Import rules, as enforced

`tests/test_import_boundaries.py` enforces the layering two ways:
runtime checks (fresh interpreter per check, so prior imports cannot
mask a transitive leak) and a static AST scan of every import
statement.

Runtime checks:

- `dataflow.core` imports nothing heavy (no torch/jax/cuda/
  dataflow_sim at import time).
- `dataflow.runtime` never imports torch, jax, or `dataflow_sim`
  (torch enters only through the cuda device backend and
  `runtime/interop.py`, which callers import explicitly).
- `dataflow_training.blocks` never pulls in `dataflow_sim`.

Static rules:

- **R1 — the engine is blind.** No module under `src/dataflow` imports
  `dataflow_training` or `reference_models`. The engine never sees the
  workload or the truth tree.
- **R2 — the workload uses public engine surfaces only.** Modules
  under `src/dataflow_training` import `dataflow` only through:
  `dataflow.core.*`, `dataflow.runtime.*` (the ABIs),
  `dataflow.service` itself (Server/EngineConfig/EngineClient for
  rigs), `dataflow.service.client`, `dataflow.service.registry`
  (`register_program_resolver`), and `dataflow.service.wire`
  (`ServiceError`).
- **R3 — tools stay near package roots**, with documented looseness:
  tools may import `dataflow` itself and anything under the
  `dataflow.core`/`dataflow.runtime`/`dataflow.service` subtrees, and
  `dataflow_training` at most two levels deep
  (`dataflow_training.x.y`) — deeper only inside
  `dataflow_training.model_families` (family packages) and
  `dataflow_training.blocks`. This is the tightest rule the current
  tools pass; the gap to the "package-root public exports only" ideal
  is accepted, not aspired away.
- **R4 — the simulator is optional everywhere but planning.** No
  module under `src/` imports `dataflow_sim` at module top level
  except under `dataflow_training.lowering` (tools and tests are
  exempt by scope). Lazy in-function imports are allowed — that is how
  `dataflow.core.convert` keeps the simulator optional.

Accepted static-scan limitation: `from dataflow.service import X` is
judged by the module (`dataflow.service`), so pulling a non-exported
submodule through an allowed package would pass the scan; the allowed
packages re-export their public surface, so the rule tracks the real
contract.

## The dataflow_sim dependency map

`dataflow_sim` (the sibling simulator repo) is consumed, never the
reverse:

- `dataflow_training/lowering/planning.py` — THE planning boundary:
  `plan_program` (PressureFit + `plan_with_recompute`),
  `simulate_program`; all sim imports in-function.
- `dataflow_training/lowering/replay.py` — `replay_gap_pct`
  re-simulates with measured durations (in-function import).
- `dataflow.core.convert` — schema converters (`to_sim_chain`,
  `to_webapp_program`), lazily imported so `dataflow.core` stays
  dependency-free.
- tools (`export_program`, `export_measured_run`, `trace_real_run`)
  and the parity/convert tests import it directly (out of R4's scope).

## End-to-end flow

```
ShapedConfig (family + shapes)
  → fam.lower(cfg)                  model_families/<fam>/model.py via
                                    lowering/shaped_program + emit:
                                    exact-size objects + tasks + rewrites
  → profile pass                    run/profiling: measured runtimes +
                                    workspace + PCIe (disk-cached)
  → lowering/planning               PressureFit + plan_with_recompute
                                    (dataflow_sim; preplace="task0")
  → annotated core Program          directives joined back; static placement
                                    packed + proven against physical VRAM
  → dataflowd                       register once (resolver_spec kind
                                    "model_family"), init-as-program seeds
                                    W/O in the store, run() per step —
                                    state persists between runs
  → trace / report                  real + wall tok/s vs sim; webapp exports
```

Drivers: `tools/train_solo.py` (single box, reference-vs-engine
parity + scaling), `tools/train_fleet.py` (data-parallel fleet),
`tools/bench_frontier.py` (throughput sweeps).

## Simulator semantics the runtime reproduces

From `dataflow_sim.engine.simulator` (the contract the engine-vs-sim
parity gates pin):

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

## Verification gates

Every layer sits behind a standing gate:

- engine-vs-sim parity on the fake backend, exact
  (`tests/dataflow/runtime/test_parity_vs_sim.py`);
- real-GPU synthetic execution vs the simulator's prediction
  (`tools/engine_gate.py`);
- per-op / per-block / per-model correctness ladders
  (`tests/dataflow_training/modules/`, `tests/dataflow_training/models/`;
  `tools/verify_family.py` audits the canon per family) against the
  isolated twins in `reference_models/`
  ([correctness_compare.md](correctness_compare.md));
- engine stress — poison-on-free, interleaving, measured-cost replan
  (`tests/dataflow/runtime/test_engine_stress.py`);
- the layering rules themselves (`tests/test_import_boundaries.py`)
  and the external-family plugin gate (`tests/test_external_family.py`);
- measured benchmark sweeps with per-cell provenance
  (`docs/benchmarking.md`).
