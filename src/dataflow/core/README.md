# dataflow.core — program IR

**Purpose.** The single schema every other layer builds on: objects, tasks,
movement directives, recompute metadata. JSON round-trip, structural
validation, and converters to/from `dataflow_sim`'s schemas.

**Imports.** Standard library only at import time. `convert.py` imports
`dataflow_sim` lazily inside functions; callers that never convert never need
the simulator installed.

## Contract

- `Program` is a **linear chain**: task list order is execution order.
  Dependencies remain derivable from producer/consumer relations (the IR is
  DAG-ready; the current dispatcher and all policies assume chain order).
- Objects are named once: an id is created either in `initial_objects` or by
  exactly one task's `outputs`, never both, never twice. Exception: an
  initial object may appear once per *location* (a planner may pre-place a
  fast copy of a backing-source object — the simulator pools by
  `(id, location)`).
- `TaskSpec.inputs` are read-only unless listed in `mutates`
  (`mutates ⊆ inputs`). A mutated object's backing copy is stale until
  offloaded; planners must offload (never bare-release) mutated objects that
  persist.
- `TaskSpec.compute_block_key` + `block_params` identify the executable
  implementing the task. Executables resolve by key — never by task id — so
  planner-inserted tasks (recompute) bind automatically.
- `TaskSpec.comm_groups` maps a comm purpose to a peer-group NAME
  (`{"dp": ...}`, `{"tp": ...}`): which tasks communicate is program
  structure; the live handles bind per run, and a run without the group
  executes the task standalone. `block_params` stays geometry-only.
- Sizes are **exact**: when an object carries dense `TensorMeta`
  (shape+dtype, no strides), validation asserts `size_bytes` equals the dense
  size. Units: bytes, microseconds; bandwidths are bytes/µs.
- Directive fields empty ⇒ *bare* program (what lowering emits, what policies
  consume). Directive fields filled ⇒ *annotated* program (what the runtime
  executes). Same schema both ways; `Program.is_annotated()` distinguishes.
- `final_locations` constrain terminal placement. For multi-step training,
  lowering must make `final_locations` equal each persistent object's initial
  location so one annotated chain replays per optimizer step.

## API surface

- `program.py` — `Program`, `ObjectSpec`, `OutputSpec`, `TaskSpec`,
  `TransferDirective`, `RecomputeOption`, `RecomputeRewrite`
- `types.py` — `TensorMeta`, `dtype_nbytes`, `DTYPE_BITS`
- `validate.py` — `validate_program` (structural; movement feasibility is the
  simulator's/runtime's job), `ValidationError`
- `jsonio.py` — `program_to_dict/from_dict`, `save_program/load_program`
  (`schema_version: "dataflow-rt/v1"`)
- `convert.py` —
  - `to_sim_chain(program)` → `dataflow_sim.core.schema.TaskChain`
  - `apply_chain_annotations(program, chain)` → annotated `Program` (joins
    directives + pre-placed initial fast copies; the chain is authoritative
    for capacities/bandwidths)
  - `to_webapp_program(program)` → DataflowProgram v1 dict (webapp upload;
    exports declared `cost_subops` metadata as roofline subops so the webapp
    can re-resolve costs for any hardware)
  - `from_sim_chain(chain, name=)` → `Program`

## Gotchas

- The sim's `validate_chain` is location-aware and only meaningful for
  **annotated** chains — a bare chain with backing-source objects legally
  fails it (no prefetches exist yet).
- DataflowProgram v1 carries no movement directives and no recompute
  rewrites; run planning locally before webapp export (the webapp re-plans
  with its own policy controls).
