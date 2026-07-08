# The Dataflow Program schema

The `Program` is the contract between everything in this project: the
lowering emits one, the planners consume and re-emit one, the engine
executes one, and the webapp simulator renders one. It is a public API
(`dataflow.core`), versioned as `SCHEMA_VERSION = "dataflow-rt/v1"`.

**Construct in Python** (the dataclasses below + `validate_program`);
treat JSON as the versioned serialization
(`program_to_dict` / `program_from_dict`, `save_program` /
`load_program` — the same JSON uploads to the webapp simulator).
Field additions bump the schema version; don't hand-write JSON.

## Program

| field | type | meaning |
|---|---|---|
| `name` | str | free-form; artifact filenames derive from it |
| `initial_objects` | tuple[ObjectSpec] | objects that exist BEFORE task 0, with starting locations (weights, optimizer state, inputs, checkpoints, metadata payloads) |
| `tasks` | tuple[TaskSpec] | the linear chain, in EXECUTION ORDER — order is the schedule |
| `final_locations` | dict[str, str] | where each persistent object must END; the replay contract: equal to its initial location ⇒ one annotated program replays every step |
| `fast_memory_capacity` | int | device budget in bytes (the ledger the planners fill) |
| `backing_memory_capacity` | int | pinned-host budget in bytes |
| `bandwidth_from_slow` / `bandwidth_to_slow` | float | bytes/s for the two transfer directions (planning + sim costs) |
| `recompute_rewrites` | tuple[RecomputeRewrite] | optional declarations the recompute planner may exercise (training chains; empty for custom programs) |
| `metadata` | dict | free-form provenance (lowering tag, measured-costs stamp, ...). Saved plans from `bench_train`/`gap_analysis` stamp their capacities here — `budget_gib` + `budget_semantics` (device envelope vs ledger), `planned_budget_gib` (the ledger the plan was fit to), `backing_plan_cap_gib` + `backing_cap_source` (flag vs auto-host), `placement` — so every plan is tied to the capacities it was planned for. |
| `schema_version` | str | `"dataflow-rt/v1"` |

## TaskSpec

| field | type | meaning |
|---|---|---|
| `id` | str | unique; free-form at the schema level (naming shapes like `block_fwd_{step}_{round}_{layer}` are TRAINING-WRAPPER conventions, not schema) |
| `inputs` | tuple[str] | object ids that must be resident in fast memory to dispatch |
| `outputs` | tuple[OutputSpec] | objects this task CREATES (with sizes/roles) |
| `mutates` | tuple[str] | objects read-modify-written in place; ordering among mutators of the same object is a dependency |
| `runtime_us` | float | task cost — roofline at lowering, replaced by measured profiles; sim + PressureFit quality depend on it |
| `group` | str | coarse label (`"optimizer"`, `"recompute"`, ...) used by tooling/display, not by the engine |
| `compute_block_key` | str | the resolver key: `resolver(task) -> executable` dispatches on it |
| `block_params` | dict | executable parameters (e.g. `layer`, optimizer `step` index) |
| `releases_after` | tuple[str] | directive: free these objects' fast allocations when the task completes |
| `offload_after` | tuple[TransferDirective] | directive: enqueue fast→slow copy, release fast on completion |
| `prefetch_after` | tuple[TransferDirective] | directive: enqueue slow→fast copy |
| `label`, `metadata` | str, dict | display + provenance |

The three `*_after` directive fields are what planning ADDS — a bare
program has them empty; PressureFit fills them. `TransferDirective` is
`(object_id, runtime_us)`.

## ObjectSpec / OutputSpec

| field | type | meaning |
|---|---|---|
| `id` | str | unique object name |
| `size_bytes` | int | ≥ 1; sizes are packed-layout truth, never ad-hoc math |
| `location` | one of `("fast", "backing")` | where the object starts (ObjectSpec) / is created (OutputSpec) |
| `role` | one of `("parameter", "gradient", "optimizer_state", "activation", "input", "output", "temp", "other")` | semantic tag for tooling/placement heuristics |
| `tensor` | TensorMeta \| None | optional `(dtype, shape, strides)` for display; byte size is authoritative |

## RecomputeRewrite / RecomputeOption

A rewrite declares that object `object_id` (an activation context),
produced by `f_task_id`, MAY instead be re-derived before `r_task_id`
consumes it, at one of several `options`:
`RecomputeOption(level, saved_bytes, recompute_us, label)` — level 0 is
"save everything" and each higher level trades saved bytes for
recompute time. `f_/r_compute_block_key` bind the re-lowered variants;
`group_key` ties rewrites that must move together. Only the recompute
planner reads these; the engine never does.

## Validation

`dataflow.core.validate.validate_program(prog)` raises a
`ValidationError` listing every structural problem: duplicate ids,
zero/negative sizes, unknown locations/roles, tasks referencing
undeclared objects, outputs colliding with existing objects,
`final_locations` naming objects that never exist, and
mutation-ordering violations. Run it after any hand construction —
the builtin lowerings and planners validate what they emit.

## What is NOT schema

Naming conventions (`W_i`, `A_{s}_{r}_{i}`, task-id shapes), the
metadata-object grammar (`M_*`/`dM_*`), and grad-accumulation
create-vs-mutate patterns are conventions of the TRAINING WRAPPERS and
builtin executables (see the README and
[extending_programs.md](extending_programs.md)). The engine and this
schema are name-agnostic; adopt the conventions when you want to reuse
the builtin executables and tools, ignore them when you don't.
