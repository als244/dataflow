# The program contract: workload <-> engine

The repo is split into an **engine** (`src/dataflow` — executes
programs, knows no model vocabulary) and a **workload**
(`src/dataflow_training` — builds programs, knows everything about
models). This document is the contract between them: every rule below
is enforced by code cited next to it. The layering rules that keep the
two sides apart are in [architecture.md](architecture.md); the service
that hosts the seam is in [engine_service.md](engine_service.md).

```
workload (import time)                    engine (run time)
─────────────────────                     ────────────────
register_program_resolver(kind, build)    lookup_resolver(spec)        spec["kind"] -> build
                                          build(spec)      -> Resolver
                                          Resolver(task)   -> Executable   (per task)
                                          Executable.launch(TaskContext)   (enqueue-only)
```

## The rules

### 1. Registration carries an opaque `resolver_spec`; the engine reads one key

A program registers with a `resolver_spec` dict. The engine
(`src/dataflow/service/registry.py`) reads exactly one key — `"kind"`
— and hands the WHOLE spec to the build registered for that kind:

- `register_program_resolver(kind, build)` — workloads call this at
  import time. The daemon default-loads
  `dataflow_training.register.register_all` (which registers the one
  builtin kind, `"model_family"`); `tools/train/dataflowd.py
  --no-default-workloads` boots a bare engine, and `--plugin` /
  the `load_plugin` verb import modules that register more kinds.
- `lookup_resolver(spec)` — resolves `spec["kind"]` to its build and
  calls `build(spec)`, cached by the spec's canonical JSON (builds
  must be pure functions of their spec). Unknown or missing kinds fail
  loudly, naming what IS registered.

Everything else in the spec is the workload's own vocabulary. The
builtin kind's wire form (`dataflow_training/register.py`):

```python
{"kind": "model_family", "family": "<name>", "cfg": {...}, "hyper": {...}?}
```

The engine never learns what `family` or `cfg` mean.

### 2. `build(spec) -> Resolver`; `Resolver(task) -> Executable` for every task

The build returns a resolver: a callable `task -> executable` that
must cover EVERY task any of the kind's programs emit — including
tasks the planner inserts (recompute) and the shared `family_init`
task (rule 8). Dispatch is on `task.compute_block_key` (+
`block_params`), **never on task id**: planner-inserted tasks carry
fresh ids but the same compute keys, so they bind automatically
(`src/dataflow/core/program.py`, `TaskSpec` docstring). Resolvers
should fail loudly on unknown keys so lowering/resolver drift dies at
resolution, not mid-launch (see `ToyResolver` below).

### 3. `Executable.launch(TaskContext)` is enqueue-only

`src/dataflow/runtime/executable.py`: the runtime hands an executable
everything it may touch and nothing else — buffers for the declared
inputs/outputs/mutates (+ optional workspace), the compute stream, the
backend, and the opaque per-run values. `launch(ctx)` may only
**enqueue device work on `ctx.stream`** — no synchronization, no
globals; scratch allocation only through torch's caching allocator,
declared via the kernel registry's `allocates=`/`workspace=` fields. The full rulebook (with the measured
incidents behind each rule) is [task-contract.md](task-contract.md).

### 4. Buffers bind positionally

`ctx.inputs` / `ctx.outputs` / `ctx.mutates` are id-keyed buffer maps,
and the convention for WHICH id means what is the task's declaration
order: executables index `ctx.inputs[ctx.task.inputs[0]]`,
`ctx.outputs[ctx.task.outputs[0].id]`, … The positional convention per
compute key is documented next to each executable class; lowering and
blocks must agree on it (e.g. `ToyBlockBwd` below: inputs are
`(dy, A, x, W [, dW accum])` in that order).

### 5. `block_params` is geometry; `comm_groups` is peer-group names by purpose

`TaskSpec.block_params` stays geometry/math the block needs (layer
index, optimizer `step`, init `seed`, `tp_slices`, …). Group
addressing lives in `TaskSpec.comm_groups`: a map from comm PURPOSE to
the NAME of the peer group serving it — `{"dp": name}` for gradient
exchange, `{"tp": name}` for tensor-parallel collectives. The name is
the lookup key into the per-run `ctx.groups` table (the daemon's live
peer-group table, snapshotted per run); a run without that group
executes the task standalone ([distributed_training.md](distributed_training.md)).

### 6. Sizes are exact, validated, and strictly bound

Object sizes are byte-exact at lowering time (packed layouts — never
hand-computed). `dataflow.core.validate_program` enforces structural
sanity including `size_bytes >= 1` and that a declared dense
`TensorMeta` matches `size_bytes`; the service additionally binds each
initial object to a resident by STRICT size match at register/run time
(`BINDING_MISMATCH` in `src/dataflow/service/handlers_runs.py`).

### 7. Objects are opaque bytes

To the engine an object is `(id, size_bytes, location, role[, tensor
meta])` — a span of bytes it places, moves, and hands to executables.
Packed layouts, dtypes, and field views are workload vocabulary
(`dataflow_training/blocks/layouts.py`); the engine never interprets
object contents.

### 8. INIT IS A PROGRAM

There is no server-side "materialize the model" verb. Initialization
is an ordinary program with one task:

- `build_init_program(fam, cfg, seed=, object_sizes=, tp_view=)`
  (`src/dataflow_training/model_families/families.py`) builds a
  program whose single `family_init_0` task (compute key
  `family_init`) declares the training program's initial objects as
  backing-resident OUTPUTS.
- The registered kind's resolver dispatches `family_init` to
  `FamilyInitExecutable` (`src/dataflow_training/register.py`), which
  fills the output buffers through the family's own seeded
  `initial_values` — byte-identical to in-process init by
  construction, because it IS that code path writing into the task's
  buffers.
- `init_model(client, family, cfg_dict, ...)`
  (`src/dataflow_training/run/driver.py`) is the client-side sugar:
  build, register, run, unregister — the daemon's final-object capture
  persists every `W_/O_/Aux_/data` object into the store — server-side
  init with zero engine vocabulary.

### 9. `run_args` are opaque per-run values; tasks interpret them

`client.run(prog_id, args={...})` passes `args` through to every
task's `ctx.run_args` UNINTERPRETED (`handlers_runs.py`: "run_args
pass through OPAQUE") and immutable. What the keys mean is a contract
between the workload's driver and its tasks — e.g. the optimizer's
global `step`, the loss denominator `valid_rows`, and the packing
precedent:

**Segments** (`src/dataflow_training/data/segments.py`,
`resolve_segments`): the per-round varlen descriptor is resolved from
run_args by the FIRST consuming task and cached in `ctx.run_values`.
It accepts the internal form `run_args["segments"] = {round:
Segments}`, the wire form `run_args["seq_lens"] = {round: [0, b1, …,
t]}` cumulative boundaries, or NOTHING — the uniform partition implied
by dims. Device fields (`cu`/`positions`) materialize once, pinned +
non-blocking, value-deduped. The engine ships none of this; it is
workload code riding an opaque channel.

### 10. Determinism and audit are workload obligations

The engine replays an annotated chain deterministically (one control
thread, strict pacing), but MATH determinism — deterministic kernels,
no hidden host syncs, bitwise-reproducible steps — and correctness
auditing are the workload's job: the task contract
([task-contract.md](task-contract.md)), the kernel audit battery and
family ladders (`tests/dataflow_training/`), and the per-family
equivalence bar against the isolated twins
([correctness_compare.md](correctness_compare.md),
`reference_models/`). The engine validates structure, not math.

## Minimal worked example: the external toy family

`tests/fixtures/external_family/toy_family.py` is the complete
contract in one CPU-safe file: a residual-MLP family defined entirely
OUTSIDE `src/`, registering one block kind and resolving end to end
(`tests/test_external_family.py` is the gate that proves it). Walking
it top to bottom:

**A config the shaped builder accepts** — note the stamp gap (§KNOWN
GAPS in [extending_external.md](extending_external.md)):

```python
@dataclass(frozen=True)
class ToyConfig:
    """``n_heads``/``n_kv_heads`` are vestigial for this attention-free
    block — the generic builder stamps them into program metadata
    unconditionally, so every family must carry them."""
    n_layers: int = 1
    d_model: int = 64
    ...
```

**Byte-exact layouts and lowering** — sizes come from `PackedLayout`,
structure from the shared chain grammar:

```python
def lower_toy(cfg, recompute_levels=None):
    dims = toy_dims(cfg)
    shaped = build_shaped_program(cfg, kinds={"block": toy_kind_spec(dims)},
                                  family="toyfam",
                                  recompute_levels=recompute_levels)
    return apply_exact_sizes(shaped, "toyfam-exact",
                             object_size=object_size_factory(dims, toy_layouts(dims)))
```

**An executable: positional binding + enqueue-only launch** — buffers
are located by position in the task's declaration, viewed zero-copy
via `dataflow.runtime.interop.torch_view`:

```python
@dataclass(frozen=True)
class ToyBlockFwd:
    def launch(self, ctx) -> None:
        x = torch_view(ctx.inputs[ctx.task.inputs[0]], (d.tokens, d.d_model), torch.bfloat16)
        w = toy_weight_layout(d).views(ctx.inputs[ctx.task.inputs[1]])
        y = torch_view(ctx.outputs[ctx.task.outputs[0].id], ...)
        ...
        if len(ctx.task.outputs) > 1:      # A saved (recompute level 0)
            a = toy_activation_layout(d).views(ctx.outputs[ctx.task.outputs[1].id])
```

The backward (`ToyBlockBwd`) reads `ctx.task.mutates` to decide
create-vs-accumulate for `dW` — the grad-accum convention is carried
by the program, not hardcoded.

**A resolver: compute keys -> executables, loud on drift**:

```python
class ToyResolver:
    def __call__(self, task):
        key = task.compute_block_key
        if key not in self.table:
            raise KeyError(f"no toyfam executable for compute_block_key {key!r} ...")
        return self.table[key]

def build_toy_resolver(dims, hyper=None):
    table = {"embed_fwd": EmbedFwd(dims, kernels),
             "toy_block_fwd": ToyBlockFwd(dims, kernels),
             "toy_block_recompute": ToyBlockRecompute(dims, kernels),
             "toy_block_bwd": ToyBlockBwd(dims, kernels),
             "head_loss": HeadLoss(dims, kernels),
             "embed_bwd": EmbedBwd(dims, kernels),
             "optimizer_block": OptimizerStep(dims, kernels, hyper,
                                              resolve_layout=toy_optimizer_layout),
             ...}
    return ToyResolver(table)
```

Family-neutral tasks (embed, head/loss, optimizer) reuse the shared
templates from `dataflow_training.blocks.base_blocks`; only the
family's own block math is new code.

**Registration at import time**:

```python
register_family("toyfam", toy_family)      # model_families.families
```

Importing the module (via `load_plugins(explicit=["toy_family"])`, a
`dataflow.families` packaging entry point, or the daemon's `--plugin`)
is registration.

**End to end through the service seam** (from
`tests/test_external_family.py`):

```python
register_all()                                     # registers kind "model_family"
spec = {"kind": "model_family", "family": "toyfam",
        "cfg": dataclasses.asdict(cfg)}
resolver = lookup_resolver(spec)                   # engine-side: reads only "kind"
for task in fam.lower(cfg).task_by_id().values():
    ex = resolver(task)                            # every task resolves
    assert hasattr(ex, "launch")
init_task = F.build_init_program(fam, cfg, seed=0).task_by_id()["family_init_0"]
assert hasattr(resolver(init_task), "launch")      # init-as-program composes too
```

That is the whole contract: one registered kind, one build, one
resolver covering every task, executables that only enqueue.
