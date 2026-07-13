# Bring your own Program: custom task graphs on the core engine

`extending.md` / `extending_external.md` cover new MODEL FAMILIES inside
the standard training flow (`lower -> profile -> plan -> train`). This
document is for everything else: programs whose structure the family
grammar does not describe — RL post-training, partial pipelines,
inference-adjacent graphs, research schedules. For these you build the
`Program` yourself and interface with the pipeline stages directly,
ending at `Engine.execute`.

## The pipeline, stage by stage — what each consumes and produces

The standard flow calls these under the hood; a custom program calls
them explicitly. Every arrow is plain data (`dataflow.core.Program`).

| stage | call | consumes | produces |
|---|---|---|---|
| build | your code (or a family `lower`) | — | bare `Program`: tasks `(inputs, outputs, mutates)`, `ObjectSpec` sizes + initial locations, bandwidths, capacities |
| cost | `apply_measured_costs(prog, profiles)` from `load_or_profile`, or hand-set `runtime_us` | bare Program | same Program, per-task measured costs (sim + PressureFit quality depend on these) |
| recompute planning | `plan_program(..., recompute=True, build_variant=...)` | Program + `recompute_rewrites` + a variant re-builder | Program re-lowered at the chosen recompute levels. **Standard-training only** — see below |
| PressureFit | `plan_program(prog, fast_memory_capacity=cap)` | costed Program + device budget | ANNOTATED Program: per-task `offload_after` / `prefetch_after` / `releases_after` transfer directives + sim makespan (`PlannedProgram`) |
| placement | engine-side (static packing or `vmm`) | annotated Program | physical offsets (extent reported; see docs/benchmarking.md for envelope semantics) |
| execute | `Engine(backend).execute(prog, resolver=..., initial_buffers=...)` | annotated Program + resolver + pinned host buffers | `RunResult` (timings, pool stats, readbacks via object views) |

Notes:

- **The engine is name-agnostic.** Task/object ids are free-form at this
  level; the `<prefix>_{step}_{round}_{layer}` shape is a convention of
  the TRAINING WRAPPERS (`train()` loss readback, NVTX renaming, window
  analyzers), not the engine. Keep the shape if you want those tools;
  ignore it if you drive `Engine.execute` yourself.
- **Recompute planning is standard-training only.** The planner's greedy
  selection assumes the family chain grammar and a `build_variant` that
  re-lowers from a config. In a custom program YOU are responsible for
  recompute: emit recompute tasks explicitly as ordinary tasks (the
  builtin `BlockRecompute` executables work — that is the point of the
  worked example below), placed statically where you want them.
- **PressureFit works on any Program.** It reads task order, object
  sizes/lifetimes, costs, and the budget — no model knowledge.

## Building a Program in Python

Construct through `dataflow.core` (`Program`, `TaskSpec`, `ObjectSpec`
— field-by-field reference: [program_schema.md](program_schema.md));
`validate_program()` runs structural checks, `save_program`/
`load_program` round-trip it, and the same JSON uploads to the webapp
simulator. The engine surface it feeds: [engine_api.md](engine_api.md).

```python
from dataflow.core import ObjectSpec, Program, TaskSpec

objects = [
    ObjectSpec(id="W_0", size_bytes=..., location="backing", role="parameter"),
    ObjectSpec(id="x_ckpt_0", size_bytes=..., location="backing", role="activation"),
    ...
]
tasks = [
    TaskSpec(
        id="block_recompute_0_0_0",
        inputs=("x_ckpt_0", "W_0", "M_0_0_0"),
        outputs=("A_0_0_0",),
        mutates=(),
        runtime_us=1800.0,              # hand-set, or profile (below)
        compute_block_key="dsamoe_recompute",   # binds a builtin executable
        block_params={"layer": 0},
    ),
    ...
]
prog = Program(name="my-rl-step", tasks=tasks, objects=objects,
               fast_memory_capacity=..., bandwidth_to_slow=..., ...)
```

Semantics to get right (the validator catches most violations):

- `inputs` must be device-resident when the task runs (PressureFit
  guarantees it from the declarations); `outputs` are created by the
  task; `mutates` are read-modify-write (order among mutators is a
  dependency).
- Sizes NEVER come from ad-hoc math — if you reuse builtin blocks, ask
  the family layouts (`PackedLayout.total_bytes`), exactly as builtin
  lowering does.
- Optimizer tasks (`optimizer_block`/... compute keys) bind the shared
  `OptimizerStep` executable: per-FIELD optimizer choice and state
  slots come from the dims' `opt_policy` (extending.md §6) — O-object
  sizes in your hand-built program must match that policy's slots
  (size them with `opt_state_layout(..., opt_policy=...)`, never by
  hand).
- `compute_block_key` + `block_params` are the resolver-binding seam
  (buffer order is positional per key — documented next to each block
  class). Free-form executables can key however they like; the resolver
  is just `task -> executable`. Tasks that participate in collectives
  additionally declare `comm_groups` (purpose -> peer-group name; see
  distributed_training.md §2) — solo programs leave it empty.

Costs: hand-set `runtime_us` is fine to get running (PressureFit
schedules against whatever you declare; the sim prediction is only as
honest as the costs). To measure instead, the profiler works on any
program whose executables resolve: `load_or_profile(prog, resolver,
backend)` + `apply_measured_costs` — signatures with data-dependent
inputs need the `profile_fill` hook (extending.md §2).

## Reusing builtin tasks: the resolver

You do NOT need new executables for standard blocks. Build the family's
resolver and let your custom program's tasks carry the same
`compute_block_key`/`block_params` the family emits — recompute and
backward tasks then bind to the exact same battle-tested executables
(stage grammar, MetaState, in-place kernels, determinism contract):

```python
from dataflow.training.families import family

fam = family("glm52")
resolver = fam.build_resolver(fam.dims_of(cfg))     # cfg: a family config
                                                     # matching your dims
```

For genuinely new tasks (a custom loss, a fusion the grammar lacks),
write an executable class with `launch(ctx)` obeying
[the task contract](task-contract.md) — no host syncs, no D2H
readbacks, deterministic kernels — and compose a resolver that
dispatches yours and falls back to the family's for the rest.

## Worked example: RL post-training without a forward pass

Motivating scenario: an inference engine generated the rollout and
saved, per layer, an **activation checkpoint** (the block INPUT
`x_ckpt_i`), plus the **routing pack** and (for sparse-attention
models) the **index selection** — i.e. exactly the never-recompute
`M_{s}_{r}_{i}` objects of the metadata grammar. Training then:

1. takes a reward signal + the last block's output,
2. computes an RL loss at the head (the one genuinely custom op),
3. walks the layers in REVERSE: recompute layer i's float context from
   `x_ckpt_i` (+ its `M` — selection/routing are reused, never
   re-derived, which also guarantees the backward sees the inference
   engine's exact choices), then backward through layer i,
4. interleaves optimizer tasks after each dW's last mutation.

Why this is a small program, not a new subsystem: step 3 is the builtin
`BlockRecompute` + `BlockBwd` pair, bound through the family resolver;
the M objects arrive as INITIAL objects (`location="backing"`, values
from the inference engine) instead of being produced by forward tasks —
the recompute path already treats them as given (`meta_ready`); and
PressureFit handles the streaming of weights + checkpoints under
whatever device budget you set, which is the whole appeal of running RL
training on a small-memory box.

Program skeleton (one step; loop outside, or emit S steps with
`final_locations` restored for replay):

```
objects: W_i, O_i (from trainer checkpoint)  x_ckpt_i, M_i (from inference)
         reward, y_last                       A_i, dW_i, dx chain, loss
tasks:   rl_head_loss(y_last, reward, W_head) -> dy, loss     [custom op]
         for i = L-1 .. 0:
             block_recompute_i(x_ckpt_i, W_i, M_i) -> A_i     [builtin]
             block_bwd_i(dy_i, A_i, M_i, W_i, x_ckpt_i)
                 -> dx_i, creates/mutates dW_i                [builtin]
             optimizer_i(dW_i) mutates W_i, O_i               [builtin]
```

What to verify, in order: `validate(prog)` → FakeBackend dry-run
(`Engine(FakeBackend()).execute`) → CUDA with `poison_on_free=True` →
byte-compare a step against a plain-torch replica of the same math (the
golden habit — for this program the "golden" is autograd over the
recomputed layers with the same fixed selections).

## Is the Program schema a public API?

Yes, with discipline: every artifact in this repo's own results flows
through it (annotated replay, webapp upload, per-cell `program.json`),
it round-trips via `save_program`/`load_program`, and changes bump
`dataflow.core.SCHEMA_VERSION`. Recommendation for external authors:
**construct in Python** (the constructors + `validate()` are the stable
surface); treat the JSON as the versioned serialization, not something
to hand-write.

## What you give up vs. the standard flow

- `train()` (loss readback, ga-round accounting) — drive
  `Engine.execute` in your own loop, read results through object views.
- The replay contract — unless you keep the boundary invariant
  (`final_locations` == initial locations for persistent objects), each
  step needs its own annotated program.
- The recompute PLANNER — you place recompute tasks yourself (above).
- Bench tools keyed on presets (`bench_train`/`bench_frontier`) — time
  your own loop; PressureFit's sim makespan still gives you the
  prediction side, and the webapp renders your annotated program.
