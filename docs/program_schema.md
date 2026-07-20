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
| `bandwidth_from_slow` / `bandwidth_to_slow` | int | bytes per MICROSECOND for the two transfer directions (planning + sim costs; transfer duration = size_bytes // bandwidth) |
| `recompute_rewrites` | tuple[RecomputeRewrite] | optional declarations the recompute planner may exercise (training chains; empty for custom programs) |
| `metadata` | dict | free-form provenance. The lowerings stamp `family`, `primary_unit`/`primary_count` (tokens per step), and the `config` dict; `apply_measured_costs` stamps per-TASK `measured` metadata beside the profiled `runtime_us`. |
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
| `block_params` | dict | executable parameters — geometry/math the block needs (e.g. `layer`, shard regions, `tp_slices`) |
| `comm_groups` | dict | comm purpose → peer-group NAME (`{"dp": name}` gradient exchange, `{"tp": name}` tensor-parallel collectives); names resolve to live handles per run, an absent group means the task runs standalone, empty means pure-local (omitted from JSON when empty) |
| `releases_after` | tuple[str] | directive: free these objects' fast allocations when the task completes |
| `offload_after` | tuple[TransferDirective] | directive: enqueue fast→slow copy, release fast on completion |
| `prefetch_after` | tuple[TransferDirective] | directive: enqueue slow→fast copy |
| `label`, `metadata` | str, dict | display + provenance |

The three `*_after` directive fields are what planning ADDS — a bare
program has them empty; PressureFit fills them. `TransferDirective` is
`(object_id, runtime_us)`.

### Task cost contract: `runtime_us` + `metadata["cost_subops"]`

Every emitted task carries its cost twice, with a strict division of
roles:

- **`runtime_us`** (scalar): what the planner and simulator SCHEDULE
  with. Seeded from the roofline at lowering; `apply_measured_costs`
  REPLACES it with profiled truth (the subops are preserved untouched).
- **`metadata["cost_subops"]`** (list of dicts): the analytic
  decomposition the scalar was seeded from, and the source of FLOP
  accounting (`dataflow_training/lowering/flops.py`). Uniform shape:

      {"kind": "roofline", "name": "<label>",
       "flops": int, "memory_bytes": int,
       "efficiency": "matmul" | "attention" | "memory"}

  `efficiency` selects the ShapedHardware converter that priced the
  subop (`matmul_us` / `attn_us` / `mem_us`); `runtime_us` at seed time
  is their sum. Special names the FLOP walker keys on: subops tagged
  `efficiency: "attention"` form the attention buckets (causal kinds
  get the 8/10 effective-bwd split — flash backward EXECUTES the
  0.5*10 recompute-inclusive count while the algorithmic cost is
  0.5*8 — and varlen quadratic scaling);
  `"muon_ns"` marks optimizer Newton-Schulz matmul work (the optimizer
  seeds are OPT-POLICY-CONSULTED — adamw is pure `"memory"` traffic
  with `flops: 0`). Zero-cost plumbing (round prologue, init) carries
  an EMPTY list; a task with NO `cost_subops` metadata hard-fails FLOP
  accounting unless exempted.

Invariants and non-invariants: the stamping covers all four task
groups (forward/backward/recompute/optimizer) in every family — the
completeness tripwire in `tests/dataflow_training/pretrain/test_flops.py`
enforces it. Nothing structurally forces `runtime_us` to equal the
subop-derived sum after lowering: measured programs DIVERGE them by
design (`runtime_us` = profiled, subops = analytic FLOP truth), which
is exactly why FLOP reporting stays valid on measured plans.

## ObjectSpec / OutputSpec

| field | type | meaning |
|---|---|---|
| `id` | str | unique object name |
| `size_bytes` | int | ≥ 1; sizes are packed-layout truth, never ad-hoc math |
| `location` | one of `("fast", "backing")` | where the object starts (ObjectSpec) / is created (OutputSpec) |
| `role` | one of `("parameter", "gradient", "optimizer_state", "activation", "input", "output", "temp", "other")` | semantic tag for tooling/placement heuristics |
| `tensor` | TensorMeta \| None | optional `(dtype, shape, strides)` for display and cross-checking (a dense meta must agree with `size_bytes`; byte size is authoritative) |

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
mutated objects not declared as inputs. Run it after any hand construction —
the builtin lowerings and planners validate what they emit.

## What is NOT schema

Naming conventions (`W_i`, `A_{s}_{r}_{i}`, task-id shapes), the
aux-object grammar (`Aux_*`/`AuxTemp_*`/`dAuxTemp_*`), and grad-accumulation
create-vs-mutate patterns are conventions of the TRAINING WRAPPERS and
builtin executables (see the README and
[extending_programs.md](extending_programs.md)). The engine and this
schema are name-agnostic; adopt the conventions when you want to reuse
the builtin executables and tools, ignore them when you don't.
