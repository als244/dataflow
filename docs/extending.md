# Extending: custom ops, blocks, and model families

The walkthrough we ourselves follow when adding builtin families. To add
a family from OUTSIDE the package (your own repo, engine unmodified):
[extending_external.md](extending_external.md) — same content, plugin
registration instead of registry edits. For NON-STANDARD structures
that the family grammar doesn't describe (RL post-training, partial
pipelines, arbitrary DAGs): [extending_programs.md](extending_programs.md)
— build the Program yourself, PressureFit it, drive the engine
directly. The workload<->engine seam every family ultimately rides:
[program_contract.md](program_contract.md).

Verifying correctness at every level is one command:
`python tools/verify_family.py --family <name>` — runs the family's
canonical test module (per-op pins, per-task fwd/recompute/bwd ladders
vs the reference, per-model step vs the isolated twin) and audits it
for the 11-gate canon. Perf: docs/benchmarking.md. The chain of
custody is always:

```
op → block executable (staged) → reference twin → lowering → planned program → engine service
```

with a gradcheck gate at each level. Nothing reaches throughput work until
its math is pinned to an autograd reference.

## Where a family's files live (the layout contract)

**A new family is three files + one twin + one registry line.** One
package under `src/dataflow_training/model_families/<family>/`:

| file | contents |
|---|---|
| `model_families/<family>/model.py` | Shaped config, Dims + `derive_dims`, `LayerKindSpec`(s), `FamilyLayouts`, `lower_<family>`, `initial_values_<family>` — pure declarations over the generic machinery |
| `model_families/<family>/blocks.py` | block executables (STAGES forwards, derived recomputes, backwards) + `build_<family>_resolver` |
| `model_families/<family>/bridge.py` | the weight bridge: engine packed init bytes -> the twin's `state_dict` (`build_reference_model` / `load_reference_init` / `to_*_state_dict`) |
| `model_families/<family>/presets.py` | preset builders (study/smoke shapes; `tiny()` classmethod lives on the config itself) |
| `reference_models/<family>.py` | the ISOLATED pure-torch twin — torch-only, self-contained, no dataflow imports (`reference_models/README.md`) |
| `model_families/families.py` | ONE registry entry (`_FAMILIES["<family>"]`) tying it together (§7) |
| `tests/dataflow_training/models/test_<family>.py` | the family's 11-gate ladder (the canonical module `tools/verify_family.py` runs) |

Shared machinery a family composes but never edits: the family-neutral
executables and task templates (`dataflow_training/blocks/base_blocks.py`),
shared building-block modules (`blocks/modules/` — the moe package,
`dsa_forms.py`, `mla_forms.py`), kernels (`dataflow_training/kernels/`),
packed layouts (`blocks/layouts.py`), the program builders
(`lowering/shaped_program.py`, `lowering/emit.py`), and the shared
block-math suites (`tests/dataflow_training/modules/`). External
(plugin) families mirror the same shape in their own package — see
extending_external.md.

## Freezing parameters (FreezePlan)

Freezing is part of the OPTIMIZER POLICY — the `freeze()` composer in
`dataflow_training/blocks/optim.py` is the front door:

```python
from dataflow_training.blocks.optim import freeze
cfg = replace(cfg, opt_policy=freeze(layers=range(0, 16)))   # bottom 16
cfg = replace(cfg, opt_policy=freeze(fields=("wq",)))        # fleet-wide field
cfg = replace(cfg, opt_policy=freeze(pairs=(("wo", 3),)))    # (field, layer)
cfg = replace(cfg, opt_policy=freeze(base="muon", embed=True))
```

Everything derives from the policy: frozen fields get no dW storage,
no optimizer state, and no update (partial layers carry SHRUNKEN dW/O
packed over the trainable fields); fully frozen layers with trainable
layers below keep a dgrad-only backward; fully frozen layers with
NOTHING below training lose their backward, their dy, and their saved
context entirely. The structural analysis is
`lowering/freeze_plan.py` (`derive_freeze_plan` -> `FreezePlan`),
consumed by `build_shaped_program(freeze=...)` via the surgery in
`lowering/freeze_program.py` — the dense warm-up is the
`objective="indexer_kl"` configuration of the same machinery. Gates:
`tests/dataflow_training/training/test_freeze_plan.py`. Full guide:
[frozen_training.md](frozen_training.md).

### The `acc` contract (frozen-safe weight gradients)

Every block backward writes weight gradients through the `acc(name,
value)` closure (`blocks/base_blocks.py`), and this is a FREEZE
contract, not just a convenience:

- `acc` SKIPS the write for any field absent from the (policy-filtered)
  dW layout, and is a no-op when the layer has no dW at all — frozen
  fields can never crash a backward or corrupt storage.
- wgrads with their OWN standalone cost (the `X.T @ dY` GEMMs) must be
  guarded at the call site: `if acc.wanted("wq"): acc("wq", h1.T @ dq)`
  — frozen fields then skip the COMPUTATION, not just the write.
- BYPRODUCT gradients (norm weights, biases, fla-kernel side outputs)
  call `acc` bare: they fall out of fused dgrad kernels at negligible
  cost, so there is nothing to skip — the write-skip is the whole
  story.

New block code MUST follow this split; the freeze gates
(`tests/dataflow_training/training/test_freeze_plan.py`) exercise both
paths.

## 1. Write an op (`blocks/ops.py` + `kernels/`)

An op is two functions plus cost knowledge:

- **launch form** — eager torch (or a registry kernel, below) writing into
  provided tensors where possible. It runs inside the executable's
  stream scope; never synchronize, never allocate runtime-owned memory
  (op-internal torch scratch is fine — it gets measured by profiling).
- **reference form** — pure, autograd-able torch. This is what gradcheck
  differentiates; keep the same numerics discipline (bf16 storage,
  fp32 reductions) so tolerances stay tight. (The isolated twin in
  `reference_models/` REIMPLEMENTS the math independently — that
  redundancy is deliberate; see §3.)

**Fused kernels go through the registry**
(`dataflow_training/kernels/`): one file per op family registering an
eager fallback (always available) plus fused implementations under a
fixed ABI — `fn(kctx, *args)`, declared workspace, `deterministic` /
`requires(caps)` flags. Use a registry op when the math is
elementwise/reduction-shaped and worth fusing (norms, activations, rope,
losses, optimizers); call cuBLAS / flash-attention / aten directly for
GEMMs, attention, and gather/scatter. The registry gives you: per-impl A/B
(`DATAFLOW_KERNELS=eager`), kernel-set stamping into profiles (measured
costs are measurements of a SPECIFIC kernel set), and a place for future
toolchains without touching callers. The generated fleet inventory of
every registered op: [kernel_registry.md](kernel_registry.md).

Gate — ladder level 1 (see `tests/dataflow_training/models/test_llama3.py`):
launch output vs reference forward; hand-written backward vs autograd on
the reference; `rel_l2` with bf16-honest tolerances (2–3e-2). Every
registry impl passes the same test.

## 2. Compose a block (`model_families/<family>/blocks.py`)

**Author the forward as stages.** A block's forward is an ordered `STAGES`
tuple of `(name, fn(kctx, kernels, dims, state), emitted_context_fields)`
— see `BlockFwd.STAGES` in `model_families/llama3/blocks.py`. Stages
share a `state` dict (inputs, weight views, intermediates) and write any
emitted fields into the saved context when one is attached. Everything
else derives from that one description:

- the full forward runs every stage;
- **the recompute variant is derived, never written**: it runs stages
  through the last context-emitting one and stops — work that exists only
  to produce the block output (e.g. a final down-projection) is excluded by
  construction;
- two structural tests come for free (copy
  `tests/dataflow_training/tasks/test_staged_blocks.py`): every declared
  context field is emitted by some stage, and the derived recompute
  boundary excludes at least one stage (the waste tripwire).

Ordering rule: emit context fields as early as their values are final — the
recompute boundary is only as tight as your last emission. The
generated inventory of every family's compute keys and executables:
[task_kinds.md](task_kinds.md); the per-family DEEP references
(object field tables, stage lists, measured kernel sequences —
regenerated per family by `tools/gen_model_docs.py`, plugins
included): [models/](models/README.md).

The rest of the block contract:

- Buffers resolve **positionally** from the task's input/output order;
  document the convention next to the class.
- Everything backward needs beyond (inputs, weights) goes into the saved
  context `PackedLayout` — and recompute must reproduce it bit-comparably
  from the same inputs.
- Accumulating variants must ADD when the task mutates an existing gradient
  object and WRITE when it creates one — the engine says which via
  `ctx.task.mutates`.
- Sizes come from layouts (`blocks/layouts.py`); never hand-compute bytes
  anywhere else. Lowering will ask the layout for `total_bytes`.

**Aux objects — never-recompute artifacts and persistent counters.**
Some stage outputs are cheap to store but expensive or ILLEGAL to
recompute (top-k routing selections, DSA index selections: recomputing
risks a different tie-break, and bwd must see the forward's exact
choice). The grammar makes them first-class, in two flavors declared
on the family's `LayerKindSpec`:

- **`AuxTemp_{s}_{r}_{i}`** (`aux_temp_bytes`): per-round forward
  artifacts packed in one layout (routing packs, index selections —
  layouts like `moe_aux_temp_layout`, `dsv32_aux_temp_layout`,
  `glm52_aux_temp_layout`). Emitted by fwd, consumed VERBATIM by
  recompute and bwd; never a recompute candidate — recompute
  repopulates ONLY the float context (A objects), never re-selects.
- **`Aux_{i}`** (`aux_bytes`): a PERSISTENT per-layer resident (like
  W/O) — e.g. per-step + all-of-training expert-assignment counts:
  zeroed at round 0 by the round prologue, accumulated by every
  round's fwd, read by the last round's bwd (per-STEP MoE load
  balancing, the noaux bias rule).

Blocks opt in via a `*AuxTempState` mixin (`_aux_temp_state(ctx)` —
`MoEAuxTempState`, `Glm52AuxTempState`); launches stay the base-class
ones, locating A/dW/AuxTemp buffers by id PREFIX. The fleet invariant:
NO family overrides the base `launch` (custom launches are how
ctx-ABI drift starts). `*ProfileFill` mixins must seed VALID
routing/index content into the AuxTemp buffers (garbage int fields =
illegal memory access in profiled bwd; concentrated routing =
unreproducible costs). Trade-off to know: recompute frees less memory
per layer when selection bytes are pinned in AuxTemp (they offload but
never drop), so tight-envelope plans buy more recompute than a
ctx-only family would.

**Cross-layer shared aux** (the IndexShare pattern): when several
layers CONSUME one layer's AuxTemp, declare `AuxShare(producer,
consumers, grad_bytes)` and pass `aux_shared=` to
`build_shaped_program` — consumers gain the producer's AuxTemp as an
input on fwd/rc/bwd, and a `dAuxTemp_{s}_{r}_{prod}` accumulator is
chained dW-style in reverse bwd order (last consumer creates, middles
mutate, producer consumes). See `model_families/glm52/` for the full
worked example (leader/follower blocks, centroid gradient through
dAuxTemp).

Gate — ladder level 2: `check_block_backward(dims)`
(`dataflow_training/testing/gradcheck.py`) verifies dx + every packed
dW field vs autograd, recompute-equivalence (recompute+bwd ≡
save+bwd), and 2x-accumulation semantics. Families with aux objects
or heterogeneous kinds run per-kind ladders in their own test module
instead (the registry entry omits the generic gradcheck bundle — §7).

## 3. Write the reference twin (`reference_models/<family>.py` + `bridge.py`)

The correctness authority is an ISOLATED plain-pytorch model at the
repo root: standard `nn.Module` + autograd, **no dataflow imports, no
cross-imports between twins** — shared primitives are deliberately
reimplemented per file, because a second from-scratch implementation
catches bugs a shared codebase would hide. Contract, numeric
conventions, and the per-family table: `reference_models/README.md`.

The family's `bridge.py` connects the two worlds: it loads the
engine's packed init bytes into the twin's `state_dict`
byte-identically (`build_reference_model` / `load_reference_init` /
`to_*_state_dict`; shared plumbing in
`model_families/bridge_common.py` — projections transpose, tables load
directly). The bridge is what makes "same bytes in, compare the
curves" a real experiment: [correctness_compare.md](correctness_compare.md)
is the full methodology (instrument ladder, gotcha catalog, band
calibration).

### Dtypes are policy, not convention

Nothing in a family may hardcode a trainable dtype. Weight layouts pull
each field's dtype from `dims.dtypes` (a `DTypePolicy` riding the Shaped
config: per-field `param`/`grad`/`opt` roles, fnmatch overrides, first
match wins). dW layouts come from `grad_layout(wl, policy)`, optimizer
state from `opt_state_layout(...)` (per-field slot sets decided by the
OPTIMIZER policy — `[m_f | v_f]` under the adamw default, fewer or none
for sgdm/sgd/muon; never a flat view; see §6), and `OptimizerStep`
updates per field through typed views, dispatching each field's step
rule per that same policy. Embed/head tables are policy-addressed
`"embed.w"` / `"head.w"`; a heterogeneous family's optimizer resolves
its layout per task (`resolve_layout`, size-verified). Policies can be
DEPTH-DEPENDENT (`layer_overrides`: first matching layer-set wins, its
sub-policy owns that layer); per-layer dtypes mean per-layer packed
sizes, so layouts resolve per layer everywhere — block executables
derive their layer from the task's `W_{i}` object. Mixed-policy E2E
gates: `tests/dataflow_training/training/test_dtype_policy_e2e.py`.

## 4. Lower it (`model_families/<family>/model.py`)

Lowering emits the bare task chain plus the pieces planning needs:

- **Structure**: you don't write it.
  `lowering/shaped_program.build_shaped_program` owns the
  family-generic chain grammar — per step, grad-accum rounds of
  `fwd → head/loss → (recompute?, bwd)` then optimizer tasks; task/object
  **naming** (step index first: `block_fwd_{step}_{round}_{layer}`,
  `A_{step}_{round}_{layer}`, globals `W_{layer}`/`O_{layer}` carry NO
  step index); the grad-accum mutation pattern (round 0 creates dW, later
  rounds mutate it); recompute tasks + `RecomputeRewrite`s per saved
  context; optimizer `step` in `block_params` and `group="optimizer"`.
  A family passes its config + explicit `kinds=` (one `LayerKindSpec`
  per layer kind; uniform dense families seed theirs with
  `roofline_block_kind_spec`). Sizes + initial values are generic too
  (`lowering/emit.py`): declare a `FamilyLayouts` (which packed layout
  backs each weight object, per layer; init specials) and call
  `object_size_factory` / `initial_values_from_layouts`. ONE module per
  family holds all of it — config, kind specs, dims mapping, layouts
  declaration (`model_families/llama3/model.py` and friends, pure
  declarations).
- **Optimizer placement**: emit each optimizer task immediately after the
  LAST mutation of its gradient (`optimizer_placement="interleaved"`, the
  default) — the legacy all-optimizers-at-the-end order costs a 1.5–2 s
  GPU-idle PCIe drain per step.
- **Replay contract**: `final_locations` must equal each persistent
  object's initial location so ONE annotated chain replays every optimizer
  step (the boundary invariant).
- **Sizes** come from the packed layouts; `initial_values()` fills pinned
  buffers with real weights/data — mind that its generation order is part
  of reference comparability, and that the SERVICE calls it with `into=`
  (and `tp_view=` where supported) for init-as-program
  ([program_contract.md](program_contract.md) rule 8).
- **Resolver** keyed by `compute_block_key` (never task id — the planner
  inserts recompute tasks and they must bind automatically).

## 5. Measure, plan, run, verify

```python
pcie     = cached_pcie(backend)                      # disk-cached: plans stay reproducible
profiles = load_or_profile(program, resolver, backend)  # runtimes + workspace, disk-cached
planned  = plan_program(apply_measured_costs(program, profiles),
                        fast_memory_capacity=cap, recompute=True,
                        build_variant=lambda lv: apply_measured_costs(
                            lower_<family>(cfg, recompute_levels=lv), profiles))
```

(`cached_pcie` / `load_or_profile` / `apply_measured_costs`:
`dataflow_training/run/profiling.py`; `plan_program`:
`dataflow_training/lowering/planning.py`.) Use the CACHED helpers:
re-measuring PCIe per run makes plans non-reproducible (bandwidth noise
flips recompute choices), and the profile cache keys on task signatures
+ kernel set + device so a kernel swap re-measures instead of silently
reusing stale numbers. `plan_program` defaults to `preplace="task0"`
(honest head: prefetches are planned and charged, not silently uploaded
before the clock starts).

Execution is the ENGINE SERVICE ([engine_service.md](engine_service.md)):
register the planned program with the family's resolver spec, seed
W/O via init-as-program, then `run()` per step —
`dataflow_training/run/driver.py` (`daemon_client`, `init_model`,
`run_engine`) is the reference driver, and `tools/train_solo.py`
wraps it.

Gates, in order:
1. ladder level 3: `check_model_step` at a few budgets — **plan-invariance**
   (different plans, identical math) is the highest-leverage async check;
2. `tests/dataflow/runtime/test_engine_stress.py` style poison-on-free +
   interleaving-stress runs;
3. throughput: add presets, then run a sweep (`tools/bench_frontier.py
   --presets <yours> --shapes oracle --run --out-dir
   results/bench/<name>`) — shape selection, envelope legality, tables,
   and per-cell provenance are the sweep's job, not yours. Full
   protocol: `docs/benchmarking.md`.
4. if the family added or changed KERNELS, bump `PROFILE_CACHE_REV`
   (`dataflow_training/run/profiling.py`) — stale cached task costs
   silently skew both sim and the planner's recompute choices.

### FLOP accounting requirements

`lowering/flops.py` reports per-step EFFECTIVE (algorithmic fwd + bwd
+ optimizer matmuls — the sim's makespan includes optimizer-task time,
so its work counts in the numerator too) and HARDWARE (+ recompute
replays + flash-internal recompute) TFLOPs by walking the lowered
program's `metadata["cost_subops"]` — the SAME roofline numbers the
simulator prices, so accounting is correct exactly when your cost
seeds are (see "Task cost contract" in program_schema.md). What a
family must guarantee:

1. **Every emitted task carries `cost_subops`.** Families lowering
   through `build_shaped_program` with populated `LayerKindSpec` subop
   lists and `LooseCosts` get this for free (the shared
   `roofline_block_kind_spec` covers dense-causal blocks). A custom
   emitter must stamp `{"name", "flops", "memory_bytes", "efficiency"}`
   dicts per subop, or add its zero-flop plumbing keys to
   `flops.EXEMPT` with a reason. An unstamped, non-exempt task
   HARD-FAILS the accounting (the completeness tripwire) — a new
   family cannot silently report wrong numbers.
2. **Tag attention as `efficiency: "attention"`.** That tag is how the
   walker separates the attention bucket (effective/hardware split,
   varlen scaling) from plain matmuls. Everything else sums into the
   matmul bucket; `flops: 0` entries (gathers, elementwise) are
   ignored.
3. **Causal-dense attention follows the pinned factors.** Stamp fwd at
   the triangular count `0.5 · 4 · Σ sᵢ² · H · hd` (uniform form:
   `2·t·seq·d`) and bwd at `0.5 · 10` (flash's in-kernel recompute is
   executed work). If your kind's `key_prefix` is in
   `flops.CAUSAL_DENSE_PREFIXES` the walker derives the effective bwd
   (`0.5 · 8`) via the 8/10 correction and scales the bucket by the
   round's actual `Σ ℓⱼ² / (t·seq)` under varlen feeds — ADD your
   prefix there when your seeds use these factors. Kinds with other
   attention structure (selected-prefix DSA, linear attention) stamp
   their own true counts and are reported as-is (effective ==
   hardware for that bucket; no varlen scaling).
4. **Seed optimizer tasks through `optimizer_cost_seed`.** Pass your
   kind's weight-layout fields (`[(f.name, f.shape) for f in
   wl.fields]`) — the helper consults the config's OptPolicy: adamw
   reproduces the historical 7×-traffic seed byte-identically, muon
   fields charge m-only traffic PLUS a `"muon_ns"` matmul subop (2-D
   directly, 3-D expert stacks per slice). The FLOP walker sources the
   optimizer bucket from those subops (layouts×policy walk only as
   fallback), and optimizer work counts in BOTH reported quantities.
   Every in-tree builder does this — copy any of them.
5. **Recompute is free.** Planner-inserted recompute tasks carry their
   own subops and land in HARDWARE flops automatically; the walk sees
   the frozen-form program, so FreezePlan-pruned work is excluded
   without any family effort.

Gate: `tests/dataflow_training/pretrain/test_flops.py` — the
parametrized `test_every_family_walks` picks up your family from the
registry automatically; add a hand-formula check if your kind's math
is novel.

## 6. Optimizers: per-field choice, per-optimizer state

Nothing in a family hardcodes AdamW. The optimizer executable
(`OptimizerStep` in `blocks/base_blocks.py`, shared fleet-wide;
`AdamWStep` is its back-compat alias) and the O-object sizing
(`opt_state_layout`) both dispatch through
`dataflow_training/blocks/optim.py`:

- an **optimizer** = (state slots, step rule): `adamw` ("m","v" —
  the default, byte-identical to the historical layout), `sgdm`/`muon`
  ("m",), `sgd` (stateless). `register_optimizer()` adds new ones —
  from plugins too.
- assignment is per FIELD (the finest grain: one packed-layout entry).
  Three forms, ergonomic to precise:

  ```python
  cfg = replace(Cfg.tiny(), opt_policy="muon")     # THE RECIPE (below)
  cfg = replace(Cfg.tiny(), opt_policy=OptPolicy(  # explicit patterns
      default="adamw", overrides=(("w?", "muon"), ("embed.*", "sgd"))))
  cfg = replace(Cfg.tiny(), opt_policy=MuonRecipePolicy(
      overrides=(("w_router", "muon"),)))          # recipe + exceptions
  ```

  Both policy types also take **`layer_overrides`** — the SAME depth
  convention as the dtype policy (§3): `((layers_tuple, sub_policy),
  ...)`, first layer-set containing the layer wins, its sub-policy
  owns every field decision for that layer. A layer whose assignment
  is fully STATELESS (all-sgd) drops its O object entirely — lowering
  scrubs the zero-byte object and its optimizer-task reference.

  **`opt_policy="muon"` means the hybrid recipe** (`MuonRecipePolicy`):
  muon for structurally-matrix weights — rank-2 projections and rank-3
  stacked expert weights (Newton-Schulz per expert slice, batched) —
  and adamw for embeddings, the LM head, norms/gains, routers, indexer
  fields, and every 1D parameter. Muon is nesterov-momentum + quintic
  NS, and `AdamWHyper.muon_lr` sets its learning rate separately from
  the adamw fields' `lr`. Muon's step math is a REGISTRY kernel
  (`muon_step`, `kernels/muon.py`: bf16 momentum arithmetic, fused NS,
  Moonshot `0.2*sqrt(max(r,c))` scaling), same as `adamw_step`.

- `update_specials` (noaux router bias, frozen fields) remain the
  HIGHEST-priority per-field override on top of the policy.
- All step math is fp32 with storage-dtype round-trips, except muon's
  momentum (momentum-dtype arithmetic).
- **Hyperparameters**: one baseline `AdamWHyper` per resolver
  (`build_resolver(dims, hyper)` — note the service passes `hyper`
  POSITIONALLY, see the known gaps in extending_external.md), refined
  per (layer, field) by the SAME policy object — `hyper_overrides=
  ((pattern, {field: value}), ...)`, first match wins.
- **LR schedules**: `AdamWHyper.schedule = LRSchedule(kind, ...)` —
  a pure function of the optimizer step index (deterministic,
  engine-safe). Kinds: `"constant"` (the default), `"wsd"`, `"cosine"`.
- Granularity invariant: ONE optimizer task per layer (plus embed/
  head) COMPOSES every field's step inside it, whatever mix of rules
  the policy assigns, and all of the layer's state slots pack into its
  single `O_{i}` object — task count and object grammar never depend
  on the policy (only sizes do).
- Gates: `tests/dataflow_training/tasks/test_optim.py` — per-step math
  vs inline formulas, NS properties, slot layouts, and a mixed-policy
  model step through the REAL engine vs a hand replica.

## 7. The ModelFamily contract, registration, and validation

A family IS its `ModelFamily` record
(`model_families/families.py`) — the config type plus typed callables,
each a `typing.Protocol` with the exact signature documented in its
docstring:

| field | contract (see the Protocol docstring for the full text) |
|---|---|
| `config_type` | frozen dataclass with preset classmethods (`tiny()` at minimum); carries the standard knobs (`dtypes`, `opt_policy`, `optimizer_placement`) which `derive_dims` forwards into the Dims |
| `derive_dims: DeriveDimsFn` | cfg -> Dims; incompatible knob combinations raise HERE, at build time |
| `lower: LowerFn` | cfg -> Program; keeps the task/object naming shape `<prefix>_{step}_{round}_{layer}` / `A_ dW_ W_ O_ Aux_ AuxTemp_`; accepts `recompute_levels=` for planner re-lowering |
| `initial_values: InitialValuesFn` | (program, cfg, backend, seed) -> pinned host tensors; generation ORDER is part of reference comparability; must also accept `into=` (and `tp_view=` where supported) for the service's init-as-program |
| `build_resolver: BuildResolverFn` | dims -> callable `task -> executable` (with `.launch(ctx)`); must resolve planner-inserted recompute tasks (key by compute key, never task id); accepts an optional positional `hyper` |
| `block_fwd/block_bwd/block_recompute`, `weight_layout`, `activation_layout` | OPTIONAL gradcheck bundle for the generic `check_block_backward` harness — heterogeneous/MoE families leave them `None` and run per-kind ladders in their own test module |
| `twin_module`, `bridge_module` | the parity-twin hooks: import path into `reference_models/` + the bridge module (`build_reference_model` / `load_reference_init` / `to_*_state_dict`) |

The record also carries the wire glue: `fam.cfg_dict(cfg)` and
`fam.resolver_spec(cfg, hyper)` produce the service resolver spec
(`{"kind": "model_family", ...}`), and `fam.bind(cfg)` returns a
`Model` — the (family, config) handle drivers pass around.

Builtin families register in the `_FAMILIES` table; external families
call `register_family()` from a plugin module discovered via a
`dataflow.families` entry point or the tools' `--plugin` flag
(extending_external.md — same contract, different registration).

**`validate_family("name")`** structurally checks the whole surface in
seconds, no GPU math: presets exist, lowering runs and keeps the naming
shape, the resolver covers every emitted task with a `.launch`.
`tools/verify_family.py` runs it as level 0 before the test module; run
it directly while wiring a new family — it catches plumbing mistakes
(missing resolver keys, misnamed tasks) long before a ladder would.

## 8. New model family checklist

What adding a family actually touches, in order:

1. `blocks/ops.py` + `kernels/` — any op the family adds (QK-norm,
   different activation, MoE dispatch). Ladder 1 per op.
2. `model_families/<family>/blocks.py` — `STAGES` forward, derived
   recompute, bwd, layouts (family-specific packed layouts live in
   `blocks/layouts.py`). Ladder 2 + structural stage tests.
3. `reference_models/<family>.py` — the isolated twin — plus
   `model_families/<family>/bridge.py` loading the engine's init bytes
   into it byte-identically.
4. `model_families/<family>/model.py` — ONE declaration module: the
   Shaped*Config, the `LayerKindSpec`(s) into `build_shaped_program`,
   the config->dims mapping, and the `FamilyLayouts` into the generic
   lowering (§4). The recompute `build_variant` is just the builder
   re-invoked with levels. The config carries the standard knobs and
   `derive_dims` FORWARDS them into the Dims: `dtypes` (§3) and
   `opt_policy` (§6) — a family that forgets the
   `opt_policy=cfg.opt_policy` forward silently pins its users to
   adamw (the Dims default).
5. `model_families/families.py` — register the `ModelFamily` entry
   (`resolve_family` dispatches on config type; builtin configs must
   NOT subclass another family's config).
6. `model_families/<family>/presets.py` — named presets: `tiny` (ladder
   scale) as a config classmethod, a smoke preset at the locked
   real-vocab smoke geometry (`run/presets.py` re-exports it for the
   parity gates), and real-scale presets with dims verified against the
   published HF config (param-count match is the acceptance test).
7. Ladder 3 + gates (§5); a frontier sweep for quoted numbers.
8. Regenerate the GENERATED docs — the family appears in all of them
   automatically (plugins included): `python tools/list_models.py >
   docs/builtin_models.md`, `python tools/list_tasks.py >
   docs/task_kinds.md`, `python tools/gen_model_docs.py --family
   <name>` (and `tools/list_kernels.py` if you added registry ops);
   pages at non-standard run shapes: `tools/gen_model_page.py`.
   Beyond the standard contract the generators need: no-arg preset
   CLASSMETHODS (`tiny()` mandatory — it is also the kernel-TRACING
   scale); an `_aux_temp_layout` dispatch entry in gen_model_docs if
   the family has AuxTemp objects; real DOCSTRINGS on new executables
   and ops (the inventories print their first lines).

Variants the fleet already exercises — reuse, don't reinvent:

- **Heterogeneous layer kinds** (qwen35: DeltaNet + gated attention):
  build one `LayerKindSpec` per kind (sizes from the packed layouts,
  roofline cost seeds, distinct `key_prefix` for the compute-block
  keys) and pass `kinds=` + `layer_kinds=` to `build_shaped_program`.
  Task IDs stay uniform (`block_fwd_{s}_{r}_{i}`); only the compute
  keys differ per kind. Lowering sizes per-layer via
  `apply_exact_sizes(..., object_size=)`. The `ModelFamily` entry
  omits the gradcheck bundle and the per-kind ladder-2 tests live in
  the family's own test module.
- **Tied embeddings**: a config flag (`tied_embeddings=True`). The chain
  builder emits no `W_head`/`O_head`/`optimizer_head`; head tasks read
  `W_embed` (packed `[table | final_norm_w]` via `head_weight_layout`);
  round-0 `head_bwd` CREATES the shared `dW_embed` and `embed_bwd`
  accumulates into it.
- **Third-party fused kernels** (fla, flash-attn, ...): pin the exact
  fwd/bwd contracts in the family's test module BEFORE the blocks call
  them (see `tests/dataflow_training/models/test_qwen35.py` part 1),
  then wrap them as registry ops so the kernel-set stamp covers them.
  Every tensor handed to a Triton kernel must be `.contiguous()` — a
  strided column slice out of a packed context is read with the wrong
  stride and corrupts results SILENTLY.
- **MoE variants (the pluggable module, `blocks/modules/moe/`)**: a
  family opts in through FIVE points and writes no MoE math of its own:
  1. layout builders append `moe_weight_specs(dims, moe)` /
     `moe_context_specs(dims, moe)` (stacked expert fields — never
     per-expert fields);
  2. block STAGES splice `MOE_STAGES` / `MOE_SHARED_STAGES` after the
     family's ffn-norm stage (state keys `st["h2"]`/`st["h_mid"]` are
     the family-invariant seam; combine emits nothing so derived
     recompute truncates it);
  3. the block backward overrides ONLY the MLP-tail hook ->
     `moe_mlp_tail_bwd(...)` and mixes in `MoEAuxTempState` +
     `MoEProfileFill` — REQUIRED: packed aux buffers carry int32
     routing fields the profiler would otherwise feed garbage;
  4. the twin composes its own MoE reference and autogradients CE + the
     per-layer aux terms while REPORTING CE only (the pinned scalar
     convention: engine loss objects are always pure CE); block-level
     ladders pin the discrete selection (near-tie top-k flips between
     two correct forwards are model sensitivity, not gradient error);
  5. the family Dims carries `moe: MoESpec` (routing mode, aux coef,
     shared expert, dispatch/combine dtype seams — `n_experts` is
     ROUTING-ONLY; everything that sizes or prices expert state reads
     `n_local_experts`).
  Roofline seeds for MoE kinds: FLOPs from ACTIVE params, weight bytes
  from the FULL expert stack. Sub-noise sign-lottery params compare via
  `check_model_step(field_atol=...)`.
- **DSA / sparse-attention variants (dsv32 / glm52)**: index selection
  rides AuxTemp (never recomputed — see §2); the indexer's KL loss is
  gradient-injected like MoE aux (the twin autogradients it, the engine
  reports CE only); dense warm-up and frozen-indexer ablations are
  config knobs validated in `derive_dims`; absorbed/MQA execution and
  FlashMLA live behind registry capability flags
  (`requires="flash_mla"`, `kernels/dsa_flashmla.py`).

The family test module's canonical ladder (copy the newest family's —
`tests/dataflow_training/models/test_glm52.py` — not the oldest):

1. op-level pins (each new op: launch vs reference fwd, hand-bwd vs
   autograd; constructed tie rows for any top-k).
2. twin self-train (CE ≈ ln(vocab) at init, decreasing).
3. per-kind block ladder-2, byte-comparing int aux fields where the
   family has them.
4. stage completeness + derived-recompute truncation (structural).
5. lowering validation + TRIPWIRE HASHES in
   `tests/dataflow_training/training/test_lowering_stability.py`
   (re-pin only with a structural-change justification in the commit
   message).
6. `check_model_step` (+ ga2 variant; `field_atol` envelopes ONLY for
   sub-noise sign-lottery params — zero-init biases whose first-step
   sign is decided by sub-tolerance grad noise).
7. plan-invariance (different budgets, forced recompute — identical
   math, byte-compared int aux fields).
8. poison-on-free + interleave stress.
9. measured-costs-replan (profiling E2E through every signature incl.
   the family's ProfileFill).
10. multistep loss-decreases + fixed-seed determinism twice
    (byte-compare; view bf16 pairs as fp32 bit patterns —
    `torch.equal` treats equal-byte NaNs as unequal).
11. FLOP accounting walk (`test_flops.py` picks the family up from the
    registry): every task stamped or exempt, attention tagged, causal
    prefixes registered — see "FLOP accounting requirements" in §5.

Known name-couplings to check when the family's TASK/OBJECT NAMES
differ (all fail loudly, none silently):

- `tools/window_plans.py` `_TASK_RE`/`_OBJ_RE` — the seam analyzer
  asserts full name coverage and raises on unknown ids (the tool's
  CLI still keys on the retired bench_train config registry — its
  regexes are the coupling to respect);
- the drivers (`dataflow_training/run/driver.py`, `tools/train_solo.py`)
  read the `loss_{s}_{r}` / `tokens_{s}_{r}` / `targets_{s}_{r}`
  conventions (round data puts, loss fetches).

Keeping the `family-prefix_{step}_{round}_{layer}` shape (new prefixes
are fine) means only the regex alternations grow; changing the shape
means generalizing those spots first.

## Known gaps

The contract has a few sharp edges, kept honestly — the full table
with workarounds lives in
[extending_external.md](extending_external.md#known-gaps): the closed
`CFG_DICT_BY_TYPE`/`RESOLVER_FAMILY_BY_TYPE` wire-spec tables, the
`InitialValuesFn` `into=`/`tp_view=` service contract, the positional
`hyper` in `build_resolver`, the private `_Base` executable base, and
the shaped-program's unconditional `n_heads`/`n_kv_heads`/`d_ff`
metadata stamp.

See also: [The task contract](task-contract.md) — what task
executables/kernels may and may not do in the launch path (no host
syncs, no D2H readbacks, determinism), why each rule exists (measured
incidents), the spin-audit enforcement recipe, and the sanctioned
relaxation paths for host-shape vendor APIs like cublasLt.
