# Extending: custom ops, blocks, and model families

The walkthrough we ourselves follow when adding builtin families. The chain
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
details: `tasks/README.md`.

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
recompute boundary is only as tight as your last emission.

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

Gate — ladder level 2: `check_block_backward(dims)` verifies dx + every
packed dW field vs autograd, recompute-equivalence (recompute+bwd ≡
save+bwd), and 2x-accumulation semantics.

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
(per-field `[m_f | v_f]` pairs — never a flat view), and `AdamWStep`
updates per field through typed views. Embed/head tables are
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
3. throughput: `tools/m4_train.py --config <yours> --budgets ...` sweeps
   real-vs-sim (report both `real_tokens_per_s` and `wall_tokens_per_s`;
   wall is the honest number), `tools/gap_analysis.py` decomposes any gap,
   `tools/window_plans.py` checks the step seam if the family's shape
   differs materially from llama.

## 6. New model family checklist

(Exercised end-to-end by the qwen3 family — `training/families.py` is the
registry an addition plugs into; results/m5/qwen3-v1/ records what it took.
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
5. `tools/m4_train.py` `CONFIGS` — add named configs.
6. Ladder 3 + gates (§5).

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

Known llama-couplings to check when the family's TASK/OBJECT NAMES differ
(all fail loudly, none silently):

- `train_loop._STEP0_ID` — the NVTX step-renamer's family list (display
  names only);
- `tools/window_plans.py` `_TASK_RE`/`_OBJ_RE` — the seam analyzer asserts
  full name coverage and raises on unknown ids;
- `tools/m4_train.py` reads `loss_{s}_{r}` / `tokens_{s}_{r}` /
  `targets_{s}_{r}` conventions via the train loop (`round_views`,
  loss readback).

Keeping the `family-prefix_{step}_{round}_{layer}` shape (new prefixes are
fine) means only the regex alternations grow; changing the shape means
generalizing those three spots first.


See also: [The task contract](task-contract.md) — what task executables/kernels may and may not do in the launch path (no host syncs, no D2H readbacks, determinism), why each rule exists (measured incidents), the spin-audit enforcement recipe, and the sanctioned relaxation paths (capacity mode, planned host-readbacks) for host-shape vendor APIs like cublasLt.
