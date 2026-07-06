# MoE module design (v1) — the implementation spec

Status: M-B (module + kernels + ladder-1), M-C (olmoe, full ladder), M-D
(qwen35moe, full ladder) landed. Plan of record:
`~/.claude/plans/please-take-your-time-transient-gosling.md`
(Shein-approved 2026-07-06); reference semantics from `refs/flextrain`
(reference-only).

## 1. Shape of the thing

The MoE MLP is a **pluggable module** (`src/dataflow/tasks/moe/`) with the
EP-facing grammar **route → dispatch → experts → combine**. Nothing outside
the package imports MoE symbols except families that opt in; the
engine/planner/sim/chain-grammar never learn MoE exists (MoE layers are
just `LayerKindSpec`s with new key_prefixes and bigger `w_bytes`).

Family plug-in contract (5 points):
1. layout builders compose `spec.moe_weight_specs` / `spec.moe_context_specs`;
2. block STAGES splice `stages.MOE_STAGES` / `MOE_SHARED_STAGES` after the
   family's ffn-norm stage (`st["h2"]` = post-ffn-norm, `st["h_mid"]` =
   post-attention residual are the family-invariant state keys);
3. the block backward's `_mlp_bwd` delegates to `stages.moe_mlp_tail_bwd`
   (+ `MoEProfileFill` mixin);
4. the family golden composes `reference.moe_mlp_reference`;
5. the family Dims carries `moe: MoESpec`.

Module map:
- `tasks/moe/spec.py` — `MoESpec` (torch-free), weight/context field specs,
  `moe_local_rows` (the EP capacity-policy landing spot).
- `tasks/moe/reference.py` — autograd-able reference forms (the correctness
  anchor for ladder-1 pins, the standalone harness, and family goldens).
- `tasks/moe/stages.py` — forward stage fns, `moe_mlp_tail_bwd`,
  `MoEProfileFill`.
- `tasks/kernels/moe_router.py`, `moe_dispatch.py`, `moe_grouped_gemm.py`,
  `swiglu.py` (packed family) — registry op families, eager fallbacks
  everywhere (`DATAFLOW_KERNELS=eager` stays a full bisection axis).

## 2. Pinned semantics

### Routing
`logits = h2 @ w_router` — bf16 GEMM, bf16 storage; ALL routing math is
fp32 from those bf16 logits. Two modes (`MoESpec.routing_mode`):
- `topk_then_softmax` (norm_topk_prob=True; qwen35moe): top-K logits,
  softmax over the K. Weights sum to 1.
- `softmax_then_topk` (norm_topk_prob=False; olmoe): full-E softmax, top-K
  probs UNnormalized. Weights sum ≤ 1.

**Tie-break = smallest expert index, both modes.** torch.topk's CUDA
kernel returns the LARGER index first (probed), so: the eager/reference
path is `torch.sort(descending=True, stable=True)` prefix (stable-desc
preserves ascending index order among equals — probed), and the fused
kernel does iterative max + `min(candidate_indices)` (flextrain's device).
bf16 logits at t·E ≈ 4M samples/layer/round make exact ties routine — the
crafted-ties ladder test is load-bearing. Dropless: no capacity factor, no
token dropping, ever (single-rank; see §7 for the EP caveat).

### Packed `[x1 | x3]` convention (repo-wide)
`w13_experts (E_local, d, 2F)` packs the up-projection: **x1 (the silu
input) in the first F output columns, x3 (the value) in the second**.
`swiglu_packed_fwd/bwd` are general registry ops pinned BIT-IDENTICAL to
the unpacked forms (silu rounds to storage dtype BEFORE the product — the
`ops.swiglu_fwd` convention). Dense MLP and QKV adopt the same packing
later; packed halves are only ever split INSIDE swiglu_packed_* (the
contiguity lesson, packed edition). NOTE flextrain packs the REVERSE
(`[value | gate]`) — relevant only to future HF-checkpoint import, which
must interleave accordingly (slot mapping in §7 covers the expert axis).

### Dispatch / permutation vocabulary
Stable sort of the (t·K) flat assignments by expert → expert-contiguous
segments (`route_offsets` = [0, cumsum(counts)]). Saved form is `route_order`
(GATHER: order[j] = flat index t·K+k occupying slot j); the SCATTER inverse
`slot_of` is derived by one unique-index scatter where needed, never saved.
xp/yp/sact/dprob are re-derived in backward from the saved order.

### Combine (single-rounding convention)
```
routed = Σ_k route_w[t,k] · yp[slot(t,k)]          # fp32 accumulate
base   = h_mid (+ shared_bf16 when n_shared_experts) # bf16 add
y      = (base.float() + routed).to(bf16)            # one rounding
```
Shared expert (S=1): `shared_bf16 = (σ(gate_pre.float()) ·
(swiglu_packed(h2@w_s13) @ w_s2).float()).to(bf16)` — ADDITIVE, not a
(1−σ) mixture (flextrain parity; their code warns about this exact
misreading).

### Aux load-balance loss — gradient-injected, never in the scalar loss
Per layer ℓ, per grad-accum round r with T_r tokens (flextrain parity):
```
f_e   = counts_e / (T_r · K)          (detached, per-round counts)
p     = softmax_fp32(saved bf16 logits)   (full E, both modes)
dz[t,e] += (α·E / T_r) · p[t,e] · (f_e − ⟨f, p_t⟩)
```
- The runtime `loss_*` object and the golden's REPORTED loss stay CE-only;
  the golden's autograd objective is `Σ_r (CE_r + Σ_ℓ aux_{ℓ,r})` with f
  detached — autograd of that expression == the injected analytic gradient
  (ladder-pinned incl. finite differences).
- Matches the per-round-mean-CE + sum-over-rounds convention
  (`head_loss total_rows = round tokens`).
- **flextrain delta**: their training passes GLOBAL tokens_per_step to the
  kernel, scaling each round's aux grad by 1/R relative to ours. Any
  future parity run must set their knob to the chunk size (or scale α).
- α: olmoe 0.01, qwen35moe 0.001. No router z-loss (flextrain has none).

### Backward tail (order pinned in `moe_mlp_tail_bwd`)
h2 = rmsnorm_apply(resid) → re-gather xp, dyp_raw → dgrad w2 (unscaled) →
`dprob_slot = ⟨dsact_raw, sact⟩` (the F-dim dot — no yp recompute, no
division; flextrain's prescale/1e-8-clamp trick avoided) → scale by slot
route_w → wgrad w2, swiglu_packed_bwd, wgrad+dgrad w13 → dispatch_bwd
(fp32 dh2) → router bwd (full-row write) + aux injection (in-place
accumulate) → w_router grads → shared-expert bwd (recompute s_act; σ fp32)
→ ffn norm_bwd → `+ dy`. Grouped wgrads write their stacked fields
directly with create-vs-accumulate inside the op (bf16-round-then-add ==
the dense `acc()` convention) — hence `_mlp_bwd` receives `dw`/`accum`
beyond the `acc` closure.

### Determinism
No atomics anywhere; every output tile/row single-owner; sort is stable;
scatter targets are unique (permutations); F.grouped_mm probed
bitwise-repeatable. Save-mode and recompute-mode backwards are
bit-identical because recompute re-runs the same kernels on the same bytes
(ladder: full-ctx bitwise reproduction test) — the plan-invariance /
poison / interleave gates inherit this. End-to-end: fixed seed + same plan
twice → identical loss bytes (family-ladder gate).

### Comparison methodology: pin the discrete selection in block ladders
Top-k selection is DISCONTINUOUS in the logits: two numerically-different-
but-both-correct forwards (kernel vs reference attention paths, bf16) flip
a few near-tie selections, and each flipped token's expert gradients then
differ at FULL magnitude — observed ~8% rel on expert dW at tiny scale;
that is model sensitivity, not gradient error. Since selection is
non-differentiable, the correct gradient comparison conditions both sides
on the SAME selection: `moe_mlp_reference(..., route_ids=)` pins the ids
(the routing WEIGHTS stay differentiable functions of the model's own
logits). Block-level ladders pass the runtime's saved `route_ids`;
end-to-end gates (model-step/ga2/multistep vs golden) run selection-free —
the one-AdamW-step damping (lr·Δgrad ≪ |W|) keeps residual flip noise
under their 3e-2 tolerances.

## 3. Kernel sourcing + measured verdicts (RTX 5090, t=16384, K=8)

Bench: `tools/bench_moe_kernels.py` (loads flextrain's kernel file
directly; ±10% acceptance per op or justified below).

| op | impl | olmoe (E=64,F=1024) | qwen35moe (E=256,F=512) | verdict |
|---|---|---|---|---|
| topk_softmax | triton (flextrain port, MODE0 kept fp32-in-registers) | 10.9µs = 0.96× | 23.2µs = 1.34× | parity / +6µs abs for exact-fp32 deviation — accepted |
| router_bwd + aux | triton ports | tiny | tiny | — |
| sort | aten argsort+bincount | 163µs = 2.6× | 153µs = 2.8× | ~100µs abs ≈ 0.1% of step — accepted (their 3-kernel count/prefix/map is a possible later port) |
| dispatch gather | aten index_select | 616µs = 1.27× | 608µs = 1.27× | generic gather vs tiled kernel; ~0.4% of step — accepted, tiled port is a follow-up |
| combine (weighted+resid) | **triton (flextrain moe_gather port)** | 480µs = 1.06× | 456µs = 0.92× | parity. Eager was 4.99ms = 12× — the port was v1-mandatory |
| dispatch_bwd | same kernel, fp32 out | 515µs = 1.35× | 468µs = 1.12× | delta ≈ the fp32-out write (+67MB) — deliberate precision choice (dh2 takes 3-4 additive joins) |
| swiglu_packed | triton | 514µs = 0.97× | 280µs = 0.91× | parity |
| grouped mm fwd/wgrad | **aten F.grouped_mm** | 1.07–1.12× | 1.09–1.13× | vs a HOST-SIZES cuBLAS loop (flextrain's backend minus its per-layer sync, which our engine cannot pay). Delta = the aten `out.copy_`/`add_` tax (no out=/accumulate variant): ≈0.6ms per 537MB result |

**Top follow-up (quantified):** custom triton grouped GEMM with in-place
epilogue + beta-accumulate wgrad removes the copy tax ≈ 5% of MoE-expert
time (~3% of step); entry bar = beat the table above. Second: tiled
gather port (~0.4%).

F.grouped_mm facts (probed): device int32 cumulative ENDS (`offsets[1:]`),
int64 offs rejected; dgrad via `w.transpose(-2,-1)` view; wgrad `(d,M)@(M,N)
→(E,d,N)` with strided mat_a; empty segments fine (wgrad zero-fills — the
create-mode empty-expert guarantee); rows past offsets[-1] zero-filled.

## 4. Profiling: the `profile_fill` hook

Packed A objects have `tensor=None`, so the profiler's int32 zero-fill
never touches ctx fields → MoE `block_bwd` would read GARBAGE gather
indices (illegal memory access), and garbage/zero logits route everything
to K experts — measured 4–30% FASTER per grouped op than balanced routing
(concentrated-in-K), i.e. an anti-conservative, allocator-history-dependent
cost bias.

Fix: `profile_program` calls `executable.profile_fill(ctx)` when present
(one-time, before the workspace/timing launches). `MoEProfileFill` seeds
every input with small deterministic pseudo-randoms (near-balanced
multinomial routing, reproducible across cache refreshes) and writes VALID
balanced identity routing into bwd signatures' ctx index fields. Existing
families define no hook → behavior unchanged for them. NOTE: registering
the moe op families grew `KernelSet.describe()` for every resolver — a
one-time profile-cache invalidation for all configs (deliberate;
import-order-dependent lazy registration would be nondeterministic).

## 5. Saved vs derived (per layer, per round)

| saved in A | shape | | derived in bwd | how |
|---|---|---|---|---|
| router_logits | (t,E) bf16 | | slot_of | unique-index scatter of order |
| route_w | (t,K) bf16 | | xp, dyp_raw | dispatch_fwd gathers |
| route_ids | (t,K) i32 | | sact | swiglu_packed_fwd(h13) |
| route_order | (t·K,) i32 | | dprob | ⟨dsact_raw, sact⟩ then unpermute |
| route_offsets | (E_loc+1,) i32 | | counts (aux) | offsets diff |
| h13 | (t·K, 2F) dispatch_dtype | | s_act, sh_each | recomputed (shared) |
| gate_pre, s13 (shared) | (t,S), (t,2F_s) bf16 | | | |

Derived recompute (staged blocks): last ctx-emitting stage =
`moe_experts13` (or `moe_shared`) → `moe_experts2_combine` is excluded by
construction, mirroring the dense down/swiglu exclusion.

## 6. Configuration surface

`MoESpec(n_experts, top_k, d_ff_expert, routing_mode, aux_coef,
n_shared_experts∈{0,1}, d_ff_shared, dispatch_dtype="bf16",
combine_dtype="fp32", expert_ids=None)`.

- Param/grad/opt dtypes: NO new mechanism — the existing `DTypePolicy`
  per-field fnmatch overrides address `w_router` / `w13_experts` /
  `w2_experts` / `w_s*` (incl. per-layer sub-policies). fp32 router
  storage becomes a policy override once cast-at-use lands; v1 = bf16
  storage + fp32 kernel math (flextrain parity).
- `dispatch_dtype`/`combine_dtype` are the quantization seam (fp8 dispatch
  later); v1 pins ("bf16","fp32") and raises loudly otherwise.
- Deliberately NOT exposed: capacity factor (dropless only), ep_size (EP
  arrives as new dispatch/combine impls + runtime work, not a config flag).

## 7. Expert parallelism (forward-looking accounting, v1-tested)

**Global-vs-local rule:** `n_experts` is ONLY routing semantics (router
width, softmax space, aux f normalization, sort id-space). Everything that
SIZES or PRICES expert state reads `n_local_experts`/`expert_ids`: weight/
grad/opt fields are `(E_local, …)` — slot j holds global expert
`expert_ids[j]`, and that ordering IS the checkpoint-shard mapping (the
router never remaps) — ctx offsets are local, rooflines use
`moe_local_rows`. `size_of_factory`/AdamW/dtype-policy inherit it all from
the layouts. Under EP:
- dispatch/combine swap to all-to-all implementations (they are adjoint
  pairs with clean tensor boundaries; expert-contiguous segments are
  A2A-native); route stays replicated; experts stay local; family code
  unchanged.
- aux f_e needs GLOBAL counts (allreduce) — single-rank counts are already
  global.
- **The named hard problem:** local RECEIVED rows are data-dependent
  (cross-rank imbalance), colliding with the static-shape IR. Future fork:
  capacity-bounded receive buffers (drops exactness) vs dynamic placement
  mode. `moe_local_rows` is the single knob where that policy lands;
  nothing in the module forecloses either.

v1 enforcement: partial `expert_ids` is fully plumbed and unit-tested at
the layout/kernel level (sharded experts stage == reference restricted to
held experts), but family lowering raises on partial ownership until a
multi-rank runtime exists.

## 8. Feasibility math (188 GB host, ~168 usable; 32 GB VRAM)

- **OLMoE-7B-A1B** (16L, d2048, E64 K8 F1024, vocab 50304, untied,
  full-row qk-norm, θ=1e4): 6.92B params → W+dW+O bf16 ≈ 55 GB pinned ✓;
  bf16 W = 13.8 GB fits VRAM at generous budgets (weights-resident plans
  possible ≥ ~22 GiB). Roofline ≈ 10–14k tok/s at 65,536 tok/step;
  save-all-restream is h2d-bound (~5.5s) vs compute (~4.3s); recompute
  flips it compute-bound.
- **Qwen3.5-MoE-35B-A3B** (40L LLLF, d2048, E256 K8 F512 + shared 512,
  vocab 248320): 34.7B → ≈ 277 GB pinned — **host-infeasible**; the config
  exists for lowering/planning + tiny-scale validation only.
- **qwen35moe_20l** (Shein-confirmed perf config): 20L (15 lin + 5 full),
  E=256, rest stock → ≈17.8B, ≈143 GB pinned (≈88% of usable — watch OS
  pressure; fallback 20L/E=128 ≈ 78 GB). Recompute-dominant plans
  expected; W_i(lin) ≈ 1.69 GB, O_i ≈ 3.37 GB objects (no structural
  limits in slab/placement/transfers; 12 GiB packings may PlacementError
  loudly — designed failure).

## 9. Known deviations / notes

- RMS eps: repo-global 1e-5; HF qwen3.5-moe says 1e-6, OLMoE 1e-5.
  Internal golden-parity unaffected (shared constant); matters only for
  future HF-checkpoint import.
- topk MODE0 fp32-in-registers (our kernel) vs flextrain's bf16 round-trip
  of selected logits: ours matches the fp32 reference tighter; +6µs at
  E=256.
- bf16 `route_w` storage: if family ladders show precision strain, the ctx
  field promotes to fp32 (+~262 KB/layer/round) — F8 in the risk table.
  (Family ladders passed at 4e-2 with bf16 storage — not promoted.)
- Registering moe ops invalidated the profile disk cache once (see §4).
- **dt_bias sign lottery at tiny scale (qwen35moe model-step gates)**: the
  true dt_bias gradient at 0.02-init tiny scale sits BELOW the fla bf16
  kernel noise floor (measured 1e-6..3e-6), and one AdamW step from zero
  init is ±lr·sign(grad) regardless of magnitude — so runtime and golden
  each land at ±1e-4 with independently-coin-flipped signs (observed: one
  flipped element → rel_l2 ≈ 1.0 on the 4-element field; the dense qwen35
  gate passes on seed luck). `check_model_step(field_atol={"dt_bias":
  2.5e-4})` compares that field against the sign-lottery envelope instead;
  the REAL dt gradient is pinned by ladder-2 at observability-scale init
  (the qwen35-design 0.06 convention). Same bf16-ULP-vs-AdamW caveat
  docs/notes/qwen35-design.md records.

## 10. Gates

Ladder-1 (`tests/tasks/test_moe_math.py`, 19 tests): topk both modes +
crafted ties (ids bitwise, eager == reference bitwise), router/aux bwd vs
autograd + finite differences, sort properties, dispatch/combine vs
einsum + bitwise repeatability, grouped vs dense loop (uneven/zero/odd
counts; create zero-fill; accumulate), swiglu_packed bitwise pins,
**standalone tail harness fwd+bwd vs reference autograd** (shared/aux
on/off) + determinism-twice, recompute ctx bitwise, eager-vs-fused tail,
EP accounting, spec validation. Family ladders (M-C/M-D) mirror
test_qwen35_math.py; `test_*_measured_costs_replan_still_golden` is the
end-to-end profiling gate.
