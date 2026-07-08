# Extending: custom ops, blocks, and model families

The walkthrough we ourselves follow when adding builtin families. To add
a family from OUTSIDE the package (your own repo, engine unmodified):
[extending_external.md](extending_external.md) — same content, plugin
registration instead of registry edits. For NON-STANDARD structures
that the family grammar doesn't describe (RL post-training, partial
pipelines, arbitrary DAGs): [extending_programs.md](extending_programs.md)
— build the Program yourself, PressureFit it, drive Engine.execute
directly.

Verifying correctness at every level is one command:
`python tools/verify_family.py --family <name>` — runs the family's
canonical test module (per-op pins, per-task fwd/recompute/bwd ladders
vs golden autograd, per-model step vs golden params + optimizer state)
and audits it for the 11-gate canon. Perf: docs/benchmarking.md. The chain
of custody is always:

```
op → block executable (staged) → golden reference → lowering → planned program → runtime
```

with a gradcheck gate at each level. Nothing reaches throughput work until
its math is pinned to an autograd reference.

## 1. Write an op (`tasks/ops.py`)

An op is two functions plus cost knowledge:

- **launch form** — eager torch (or a registry kernel, below) writing into
  provided tensors where possible. It runs inside the executable's
  `torch.cuda.stream(external_stream(ctx.stream))` scope; never synchronize,
  never allocate runtime-owned memory (op-internal torch scratch is fine —
  it gets measured by profiling).
- **reference form** — pure, autograd-able torch. This is what gradcheck
  differentiates and what golden models are composed from; keep the same
  numerics discipline (bf16 storage, fp32 reductions) so tolerances stay
  tight.

**Fused kernels go through the registry** (`tasks/kernels/`): one file per
op family registering an eager fallback (always available) plus fused
implementations under a fixed ABI — `fn(kctx, *args)`, declared workspace,
`deterministic` / `requires(caps)` flags. Use a registry op when the math is
elementwise/reduction-shaped and worth fusing (norms, activations, rope,
losses, optimizers); call cuBLAS / flash-attention / aten directly for
GEMMs, attention, and gather/scatter. The registry gives you: per-impl A/B
(`DATAFLOW_KERNELS=eager`), kernel-set stamping into profiles (measured
costs are measurements of a SPECIFIC kernel set), and a place for future
toolchains (CuTe, TileLang, raw cubins) without touching callers. Contract
details: `tasks/README.md`; the generated fleet inventory of every
registered op (impls, priorities, signatures):
[kernel_registry.md](kernel_registry.md).

Gate — ladder level 1 (see `tests/tasks/test_llama3_math.py`): launch
output vs reference forward; hand-written backward vs autograd on the
reference; `rel_l2` with bf16-honest tolerances (2–3e-2). Every registry
impl passes the same test.

## 2. Compose a block (`tasks/<family>_blocks.py`)

**Author the forward as stages.** A block's forward is an ordered `STAGES`
tuple of `(name, fn(kctx, kernels, dims, state), emitted_context_fields)` —
see `BlockFwd.STAGES` in `llama3_blocks.py`. Stages share a `state` dict
(inputs, weight views, intermediates) and write any emitted fields into
`state["a"]` when a saved context is attached. Everything else derives from
that one description:

- the full forward runs every stage;
- **the recompute variant is derived, never written**: it runs stages
  through the last context-emitting one and stops — work that exists only
  to produce the block output (e.g. a final down-projection) is excluded by
  construction;
- two structural tests come for free (copy
  `tests/tasks/test_staged_blocks.py`): every declared context field is
  emitted by some stage, and the derived recompute boundary excludes at
  least one stage (the waste tripwire).

Ordering rule: emit context fields as early as their values are final — the
recompute boundary is only as tight as your last emission. The
generated inventory of every family's compute keys and executables:
[task_kinds.md](task_kinds.md).

The rest of the block contract:

- Buffers resolve **positionally** from the task's input/output order;
  document the convention next to the class.
- Everything backward needs beyond (inputs, weights) goes into the saved
  context `PackedLayout` — and recompute must reproduce it bit-comparably
  from the same inputs.
- Accumulating variants must ADD when the task mutates an existing gradient
  object and WRITE when it creates one — the engine says which via
  `ctx.task.mutates`.
- Sizes come from layouts (`tasks/layouts.py`); never hand-compute bytes
  anywhere else. Lowering will ask the layout for `total_bytes`.

**Metadata objects (`M_{s}_{r}_{i}`) — never-recompute artifacts.** Some
stage outputs are cheap to store but expensive or ILLEGAL to recompute
(top-k routing selections, DSA index selections: recomputing risks a
different tie-break, and bwd must see the forward's exact choice). The
grammar makes them first-class:

- The family's kind spec sets `meta_bytes` and `FamilyLayouts.block_meta_at`
  returns the M layout per layer (e.g. `moe_meta_layout`,
  `dsv32_meta_layout` — routing pack `route_w/ids/order/offsets`, and/or
  `dsa_idx` which is ALWAYS field 0 when present, the offset-0 ABI).
- Stages that populate M are marked with a 4th tuple element `"meta"`;
  they write into `st["meta"]` instead of the A-dict. On the derived
  recompute path the runner SKIPS meta-marked stages (`meta_ready`) —
  recompute repopulates ONLY the float ctx (A objects), never re-selects.
- Blocks opt in via a `*MetaState` mixin (`_meta_state(ctx)`); launches
  stay the base-class ones — locating A/dW/M by id PREFIX, not position.
  The fleet invariant: NO family overrides `BlockFwd.launch` (custom
  launches are how ctx-ABI drift starts).
- The backward receives M through `meta=` and merges it into the a-dict.
- `ProfileFill` mixins must seed VALID routing/index content INTO the M
  buffers (garbage int fields = illegal memory access in profiled bwd;
  concentrated routing = unreproducible costs).
- Trade-off to know: recompute frees less memory per layer when selection
  bytes are pinned in M (they offload but never drop), so tight-envelope
  plans buy more recompute than a ctx-only family would.

**Cross-layer shared metadata** (the IndexShare pattern): when several
layers CONSUME one layer's M, declare `MetaShare(producer, consumers,
grad_bytes)` and pass `meta_shared=` to `build_shaped_program` — consumers
gain the producer's M as an input on fwd/rc/bwd, and a `dM_{s}_{r}_{prod}`
accumulator is chained dW-style in reverse bwd order (last consumer
creates, middles mutate, producer consumes). See `training/glm52.py` +
`tasks/glm52_blocks.py` for the full worked example (leader/follower
blocks, centroid gradient through dM).

Gate — ladder level 2: `check_block_backward(dims)` verifies dx + every
packed dW field vs autograd, recompute-equivalence (recompute+bwd ≡
save+bwd), and 2x-accumulation semantics. Families with M objects extend
the harness with `meta_views` (fwd `extras={"meta": ...}`, rc
`extras+meta_ready`, bwd `meta=`) and byte-compare the int fields.

## 3. Write the golden reference (`models/<family>_reference.py`)

A hand-written plain-autograd model of the same family: it composes the
ops' *reference* forms, replicates the optimizer update exactly (including
the storage-dtype round-trips of grads and moments), and shares the
packed-weight layouts so state is comparable field-by-field
(`from_packed_bytes` constructors, per-field typed leaves,
`final_leaves(object_id)` for the gate comparisons). This is the
independent witness — it catches composition-graph mistakes that per-block
checks cannot. See `models/llama3_reference.py`.

### Dtypes are policy, not convention

Nothing in a family may hardcode a trainable dtype. Weight layouts pull
each field's dtype from `dims.dtypes` (a `DTypePolicy` riding the Shaped
config: per-field `param`/`grad`/`opt` roles, fnmatch overrides, first
match wins — `docs/notes/dtype-policy-design.md`). dW layouts come from
`grad_layout(wl, policy)`, optimizer state from `opt_state_layout(...)`
(per-field slot sets decided by the OPTIMIZER policy — `[m_f | v_f]`
under the adamw default, fewer or none for sgdm/sgd/muon; never a flat
view; see §6), and `OptimizerStep` updates per field through typed
views, dispatching each field's step rule per that same policy. Embed/head tables are
policy-addressed `"embed.w"` / `"head.w"`; a heterogeneous family's
optimizer resolves its layout per task (`layout_for`, size-verified).
Policies can be DEPTH-DEPENDENT (`layer_overrides`: first matching
layer-set wins, its sub-policy owns that layer); per-layer dtypes mean
per-layer packed sizes, so layouts resolve per layer everywhere — block
executables derive their layer from the task's `W_{i}` object
(`_Base.layer_of`). Mixed-policy E2E gates:
`tests/training/test_dtype_policy_e2e.py`.

## 4. Lower it (`training/`)

Lowering emits the bare task chain plus the pieces planning needs:

- **Structure**: you don't write it. `shaped_program.build_shaped_program`
  owns the family-generic chain grammar — per step, grad-accum rounds of
  `fwd → head/loss → (recompute?, bwd)` then optimizer tasks; task/object
  **naming** (step index first: `block_fwd_{step}_{round}_{layer}`,
  `A_{step}_{round}_{layer}`, globals `W_{layer}`/`O_{layer}` carry NO
  step index); the grad-accum mutation pattern (round 0 creates dW, later
  rounds mutate it); recompute tasks + `RecomputeRewrite`s per saved
  context; optimizer `step` in `block_params` and `group="optimizer"`.
  A family passes its config + explicit `kinds=` (one `LayerKindSpec`
  per layer kind; uniform dense families seed theirs with
  `roofline_block_kind_spec`). Sizes + initial values are generic too
  (`training/lowering.py`): declare a `FamilyLayouts` (which packed
  layout backs each weight object, per layer; init specials) and call
  `size_of_factory` / `initial_values_from_layouts`. ONE module per
  family holds all of it — config, kind specs, dims mapping, layouts
  declaration (`training/llama3.py` / `qwen3.py` / `qwen35.py`,
  ~130-230 lines each, pure declarations).
- **Optimizer placement**: emit each optimizer task immediately after the
  LAST mutation of its gradient (`optimizer_placement="interleaved"`, the
  default) — the legacy all-optimizers-at-the-end order costs a 1.5–2 s
  GPU-idle PCIe drain per step (docs/notes/step-boundary.md).
- **Replay contract**: `final_locations` must equal each persistent
  object's initial location so ONE annotated chain replays every optimizer
  step (the boundary invariant; same note, §1).
- **Sizes** come from the tasks layer's layouts (`lower_llama3` pattern);
  `initial_values()` fills pinned buffers with real weights/data — mind
  that its generation order is part of golden comparability.
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
report   = train(planned.program, cfg, backend, steps=N)
```

Use the CACHED helpers: re-measuring PCIe per run makes plans
non-reproducible (bandwidth noise flips recompute choices), and the profile
cache keys on task signatures + kernel set + device so a kernel swap
re-measures instead of silently reusing stale numbers. `plan_program`
defaults to `preplace="task0"` (honest head: prefetches are planned and
charged, not silently uploaded before the clock starts).

Gates, in order:
1. ladder level 3: `check_model_step` at a few budgets — **plan-invariance**
   (different plans, identical math) is the highest-leverage async check;
2. `tests/tasks/test_m3_gate.py` style poison-on-free + interleaving-stress
   runs;
3. throughput: add named configs to `tools/bench_train.py` `CONFIGS`, then
   run a sweep (`tools/bench_frontier.py --presets <yours> --shapes
   oracle --run --no-legacy --out-dir results/bench/<name>`) — shape
   selection, envelope legality (auto-headroom), tables, and per-cell
   provenance are the sweep's job, not yours. Full protocol:
   `docs/benchmarking.md`. `tools/gap_analysis.py` decomposes any
   real-vs-sim gap; `tools/window_plans.py` checks the step seam if the
   family's shape differs materially from llama.
4. if the family added or changed KERNELS, bump `PROFILE_CACHE_REV`
   (`training/profiling.py`) — stale cached task costs silently skew both
   sim and the planner's recompute choices.

## 6. Optimizers: per-field choice, per-optimizer state

Nothing in a family hardcodes AdamW. The optimizer executable
(`OptimizerStep`, shared fleet-wide; `AdamWStep` is its back-compat
alias) and the O-object sizing (`opt_state_layout`) both dispatch
through `tasks/optim.py`:

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
  (an `OptPolicy`, a recipe, or a string incl. `"muon"`) owns every
  field decision for that layer. So (layer index, param name)
  addressing is:

  ```python
  opt_policy = OptPolicy(default="adamw", layer_overrides=(
      (tuple(range(4)), "sgd"),                     # layers 0-3: all sgd
      ((7,), OptPolicy(overrides=(("w?", "muon"),))),  # layer 7: attn muon
  ))
  ```

  A layer whose assignment is fully STATELESS (all-sgd) drops its O
  object entirely — lowering scrubs the zero-byte object and its
  optimizer-task reference, so plans, transfers, and pinning shrink
  accordingly.

  **`opt_policy="muon"` means the hybrid recipe** (`MuonRecipePolicy`,
  flextrain's classification): muon for structurally-matrix weights —
  rank-2 projections and rank-3 stacked expert weights (Newton-Schulz
  per expert slice, batched) — and adamw for embeddings, the LM head,
  norms/gains, routers, indexer fields, and every 1D parameter. That
  split is the only configuration muon is meant to run in, so the
  string gives it to you; raw muon-on-everything requires the explicit
  `OptPolicy(default="muon")`. Muon is nesterov-momentum + quintic NS
  (flextrain-aligned coefficients; singular values land in a band near
  ~0.9 by design), and `AdamWHyper.muon_lr` sets its learning rate
  separately from the adamw fields' `lr` (the two rules want very
  different values). Muon's step math is a REGISTRY kernel
  (`muon_step`, kernels/muon.py — ported from flextrain's
  `flextrain_muon_step`: bf16 momentum arithmetic, fused NS,
  Moonshot `0.2*sqrt(max(r,c))` scaling), same as `adamw_step`
  (eager + triton); `register(...)` new implementations without
  touching the optimizer layer.

  O-object sizes follow the policy automatically (lowering asks the
  same layout fn the executable views through), so plans, transfers,
  and host pinning all shrink when a field needs fewer slots.
- `update_specials` (noaux router bias, frozen fields) remain the
  HIGHEST-priority per-field override on top of the policy.
- All step math is fp32 with storage-dtype round-trips (the AdamW
  kernel's convention), except muon's momentum which follows the
  flextrain port (momentum-dtype arithmetic).
- **Hyperparameters**: one baseline `AdamWHyper` per resolver
  (`build_resolver(dims, hyper=...)`), refined per (layer, field) by
  the SAME policy object — `hyper_overrides=((pattern, {field: value}),
  ...)`, first match wins, routed through `layer_overrides` like
  everything else:

  ```python
  OptPolicy(hyper_overrides=(("*norm*", {"weight_decay": 0.0}),
                             ("embed.*", {"lr": 1e-5})))
  ```

- **LR schedules**: `AdamWHyper.schedule = LRSchedule(kind, ...)` —
  a pure function of the optimizer step index (deterministic,
  engine-safe). Kinds: `"constant"` (the DEFAULT — debugging
  consistency: identical lr every step), `"wsd"` (the recommended
  training schedule: linear warmup, stable, linear decay over the
  last `decay_frac` to `min_lr_frac`), `"cosine"`. Decaying kinds
  additionally degenerate to warmup-then-constant until `total_steps`
  declares the run horizon. The scale multiplies `lr` and `muon_lr`
  AFTER per-field hyper overrides.
- Granularity invariant: ONE optimizer task per layer (plus embed/
  head) COMPOSES every field's step inside it, whatever mix of rules
  the policy assigns, and all of the layer's state slots pack into its
  single `O_{i}` object — task count and object grammar never depend
  on the policy (only sizes do; a fully stateless layer drops its O
  entirely).
- Gates: `tests/tasks/test_optim.py` — per-step math vs inline
  formulas, NS properties, slot layouts, and a mixed-policy model
  step through the REAL engine vs a hand replica. The all-adamw
  default is pinned byte-stable by the lowering tripwires + every
  family ladder.

## 7. The Family contract, registration, and validation

A family IS its `Family` record (`training/families.py`) — five typed
callables plus the config type, each field a `typing.Protocol` with the
exact signature documented in its docstring:

| field | contract (see the Protocol docstring for the full text) |
|---|---|
| `config_type` | frozen dataclass with preset classmethods (`tiny()` at minimum; `mini`/real-scale for benching); carries the standard knobs (`dtypes`, `opt_policy`, `optimizer_placement`) which `dims_of` forwards into the Dims |
| `dims_of: DimsOfFn` | cfg -> Dims; incompatible knob combinations raise HERE, at build time |
| `lower: LowerFn` | cfg -> Program; MUST keep the task/object naming shape `<prefix>_{step}_{round}_{layer}` / `A_ dW_ W_ O_ M_ dM_`; accepts `recompute_levels=` for planner re-lowering |
| `initial_values: InitialValuesFn` | (program, cfg, backend, seed) -> pinned host tensors; generation ORDER is part of golden comparability |
| `build_resolver: BuildResolverFn` | dims -> callable `task -> executable` (with `.launch(ctx)`); must resolve planner-inserted recompute tasks (key by compute key, never task id) |
| `golden: GoldenFn` | zero-arg -> golden CLASS with `from_packed_bytes` + `train_step` |

Builtin families register in the `_FAMILIES` table; external families
call `register_family()` from a plugin module discovered via a
`dataflow.families` entry point or the tools' `--plugin` flag
(extending_external.md — same contract, different registration).

**`validate_family("name")`** structurally checks the whole surface in
seconds, no GPU math: presets exist, lowering runs and keeps the naming
shape, the resolver covers every emitted task, the golden exposes the
harness members. `tools/verify_family.py` runs it as level 0 before the
test module; run it directly while wiring a new family — it catches
plumbing mistakes (missing resolver keys, misnamed tasks) long before a
ladder would.

## 8. New model family checklist

(Exercised end-to-end by the qwen3 family — `training/families.py` is the
registry an addition plugs into.
The qwen35 family exercises the harder variants below: heterogeneous
layer kinds, tied embeddings, third-party fused kernels.)

What adding a family (e.g. Qwen3) actually touches, in order:

1. `tasks/ops.py` + `tasks/kernels/` — any op the family adds (QK-norm,
   different activation, MoE dispatch). Ladder 1 per op.
2. `tasks/<family>_blocks.py` — `STAGES` forward, derived recompute,
   bwd, layouts. Ladder 2 + structural stage tests.
3. `models/<family>_reference.py` — golden autograd model,
   `from_packed_bytes`.
4. `training/<family>.py` — ONE declaration module: the Shaped*Config,
   the `LayerKindSpec`(s) into `build_shaped_program`, the config->dims
   mapping, and the `FamilyLayouts` into the generic lowering (§4). The
   recompute `build_variant` is just the builder re-invoked with levels.
   The config carries the standard knobs and `dims_of` FORWARDS them
   into the Dims: `dtypes` (§3) and `opt_policy` (§6) — a family that
   forgets the `opt_policy=cfg.opt_policy` forward silently pins its
   users to adamw (the Dims default).
5. `training/families.py` — register the `Family` entry
   (`resolve_family` dispatches on config type; configs must NOT subclass
   another family's config).
6. `tools/bench_train.py` `CONFIGS` — named presets: `tiny` (ladder
   scale), a `mini` (single-GPU bench scale), and the real-scale preset
   with dims verified against the published HF config (param-count match
   is the acceptance test).
7. Ladder 3 + gates (§5); sweep for quoted numbers.

Variants the qwen35 family (hybrid DeltaNet + gated attention, tied
embeddings) added to the machinery — reuse, don't reinvent:

- **Heterogeneous layer kinds**: build one `LayerKindSpec` per kind
  (sizes from the packed layouts, roofline cost seeds, distinct
  `key_prefix` for the compute-block keys) and pass `kinds=` +
  `kind_of=` to `build_shaped_program`. Task IDs stay uniform
  (`block_fwd_{s}_{r}_{i}`); only the compute keys differ per kind, so
  the train loop/NVTX/window-oracle regexes need nothing. Lowering sizes
  per-layer via `apply_exact_sizes(..., size_of=)`. The `Family` entry
  omits the gradcheck bundle (its fields default to None) and the
  per-kind ladder-2 tests live in the family's own test module.
- **Tied embeddings**: a config flag (`tied_embeddings=True`). The chain
  builder emits no `W_head`/`O_head`/`optimizer_head`; head tasks read
  `W_embed` (packed `[table | final_norm_w]` via `head_weight_layout`);
  round-0 `head_bwd` CREATES the shared `dW_embed` and `embed_bwd`
  accumulates into it. The golden takes two leaves instead of three;
  `check_model_step` branches on the config flag.
- **Third-party fused kernels** (fla, flash-attn, ...): pin the exact
  fwd/bwd contracts in the family's test module BEFORE the blocks call
  them (see tests/tasks/test_qwen35_math.py part 1), then wrap them as
  registry ops so the kernel-set stamp covers them. Every tensor handed
  to a Triton kernel must be `.contiguous()` — a strided column slice
  out of a packed context is read with the wrong stride and corrupts
  results SILENTLY (the qwen35 gate-gradient hunt).
- **MoE variants (olmoe / qwen35moe — the pluggable module)**: the MoE
  SwiGLU tail is a self-contained module (`tasks/moe/`,
  docs/notes/moe-design.md); a family opts in through FIVE points and
  writes no MoE math of its own:
  1. layout builders append `moe_weight_specs(dims, moe)` /
     `moe_context_specs(dims, moe)` (stacked expert fields
     `w13_experts (E,d,2F)` packed `[x1|x3]` / `w2_experts (E,F,d)` —
     never per-expert fields; AdamW stays 3 chunked launches);
  2. block STAGES splice `MOE_STAGES` / `MOE_SHARED_STAGES` after the
     family's ffn-norm stage (state keys `st["h2"]`/`st["h_mid"]` are the
     family-invariant seam; combine emits nothing so derived recompute
     truncates it);
  3. the block backward overrides ONLY `_mlp_bwd` ->
     `moe_mlp_tail_bwd(...)` (the M-A template split) and mixes in
     `MoEProfileFill` — REQUIRED: packed contexts carry int32 routing
     fields the profiler would otherwise feed garbage (illegal memory
     access) and routing costs must be balanced+reproducible;
  4. the golden composes `moe_mlp_reference` and autograds CE + the
     per-layer aux terms while REPORTING CE only (aux is
     gradient-injected, never in the scalar loss); block-level gradcheck
     ladders pin the discrete selection via `route_ids=` (near-tie top-k
     flips between two correct forwards are model sensitivity, not
     gradient error);
  5. the family Dims carries `moe: MoESpec` (routing mode, aux coef,
     shared expert, dispatch/combine dtype seams, `expert_ids` ownership
     — `n_experts` is ROUTING-ONLY; everything that sizes or prices
     expert state reads `n_local_experts`).
  Roofline seeds for MoE kinds: FLOPs from ACTIVE params, weight bytes
  from the FULL expert stack. Sub-noise sign-lottery params (dt_bias)
  compare via `check_model_step(field_atol=...)`.

- **DSA / sparse-attention variants (dsv32 / glm52)**: index selection is
  an M-object (never recomputed — see §2); the indexer's KL loss is
  gradient-injected like MoE aux (golden autograds it, runtime reports CE
  only); dense warm-up and frozen-indexer ablations are config knobs
  validated in `dims_of`; absorbed/MQA execution and FlashMLA live behind
  registry capability flags (`caps["flash_mla"]`).

The family test module's canonical ladder (copy the newest family's —
`tests/tasks/test_glm52_math.py` — not the oldest):

1. op-level pins (each new op: launch vs reference fwd, hand-bwd vs
   autograd; constructed tie rows for any top-k).
2. golden self-train (CE ≈ ln(vocab) at init, decreasing).
3. per-kind block ladder-2 incl. meta_views if the family has M objects.
4. stage completeness + derived-recompute truncation (structural).
5. lowering validation + TRIPWIRE HASHES in
   `tests/training/test_lowering_stability.py` (re-pin only with a
   structural-change justification in the commit message).
6. `check_model_step` (+ ga2 variant; `field_atol` envelopes ONLY for
   sub-noise sign-lottery params — zero-init biases whose first-step sign
   is decided by sub-tolerance grad noise, e.g. router bias / idx LN bias).
7. plan-invariance (different budgets, forced recompute — identical math,
   byte-compared int ctx/M fields).
8. poison-on-free + interleave stress.
9. measured-costs-replan (profiling E2E through every signature incl. the
   family's ProfileFill).
10. multistep loss-decreases + fixed-seed determinism twice (byte-compare;
    view bf16 pairs as fp32 bit patterns — `torch.equal` treats equal-byte
    NaNs as unequal).

Known llama-couplings to check when the family's TASK/OBJECT NAMES differ
(all fail loudly, none silently):

- `train_loop._STEP0_ID` — the NVTX step-renamer's family list (display
  names only);
- `tools/window_plans.py` `_TASK_RE`/`_OBJ_RE` — the seam analyzer asserts
  full name coverage and raises on unknown ids;
- `tools/bench_train.py` reads `loss_{s}_{r}` / `tokens_{s}_{r}` /
  `targets_{s}_{r}` conventions via the train loop (`round_views`,
  loss readback).

Keeping the `family-prefix_{step}_{round}_{layer}` shape (new prefixes are
fine) means only the regex alternations grow; changing the shape means
generalizing those three spots first.


See also: [The task contract](task-contract.md) — what task executables/kernels may and may not do in the launch path (no host syncs, no D2H readbacks, determinism), why each rule exists (measured incidents), the spin-audit enforcement recipe, and the sanctioned relaxation paths (capacity mode, planned host-readbacks) for host-shape vendor APIs like cublasLt.
