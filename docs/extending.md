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
bf16 state round-trips), and shares the packed-weight layouts so state is
comparable buffer-to-buffer (`from_packed_bytes` constructors). This is the
independent witness — it catches composition-graph mistakes that per-block
checks cannot. See `models/llama3_reference.py`.

## 4. Lower it (`training/`)

Lowering emits the bare task chain plus the pieces planning needs:

- **Structure**: per step, grad-accum rounds of
  `fwd → head/loss → (recompute?, bwd)` then optimizer tasks. Copy the
  conventions from `shaped_llama3.py`: task/object **naming** (step index
  first: `block_fwd_{step}_{round}_{layer}`, `A_{step}_{round}_{layer}`,
  globals `W_{layer}`/`O_{layer}` carry NO step index), the grad-accum
  mutation pattern (round 0 creates dW, later rounds mutate it), recompute
  tasks + `RecomputeRewrite`s per saved context, optimizer `step` in
  `block_params` and `group="optimizer"` (the train loop patches the step
  number by group).
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

What adding a family (e.g. Qwen3) actually touches, in order:

1. `tasks/ops.py` + `tasks/kernels/` — any op the family adds (QK-norm,
   different activation, MoE dispatch). Ladder 1 per op.
2. `tasks/<family>_blocks.py` — `STAGES` forward, derived recompute,
   bwd, layouts. Ladder 2 + structural stage tests.
3. `models/<family>_reference.py` — golden autograd model,
   `from_packed_bytes`.
4. `training/shaped_<family>.py` + `training/<family>_lowering.py` — chain
   builder (§4 conventions) + layout-exact lowering + `initial_values`.
   The recompute `build_variant` is just the builder re-invoked with
   levels.
5. `tools/m4_train.py` `CONFIGS` — add named configs.
6. Ladder 3 + gates (§5).

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
