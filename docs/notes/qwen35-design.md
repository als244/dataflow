# Qwen3.5-dense family design (M5.2, in progress)

Correctness-first working design, extracted from the flextrain reference
(`refs/flextrain/flextrain/nn/{layers/qwen3_5.py, blocks/linear_attn.py,
blocks/attention_gated.py}`) and the Qwen3.5-9B HF config. Everything below
was read from those sources, not recalled.

## 1. Architecture contract (Qwen3.5-9B)

- 32 layers, `layer_types` = (LLLF)×8 — `full_attention_interval: 4`,
  i.e. 24 Gated-DeltaNet layers + 8 gated full-attention layers.
- Shared per layer: input RMSNorm, post-attn RMSNorm, dense SwiGLU MLP
  (d=4096 → ff=12288). rms_norm_eps 1e-6.
- **Full-attention layers** (`GQAAttentionGatedBlock`): 16 heads ×
  head_dim 256 (attn_dim 4096), 4 kv heads (kv_dim 1024).
  `w_q: (d, attn_dim*2)` — output splits into (query, gate);
  per-head qk-norm (head_dim-wide, like qwen3); **partial RoPE**
  rot_dim = 0.25·256 = 64, theta 1e7, pair-interleaved convention;
  flash attention; then `xo = (attn_result · sigmoid(gate)) @ w_o + x`.
  Gate is per-head-element (broadcast over nothing — full attn_dim).
  mRoPE collapses to standard partial RoPE for text-only (we are).
- **Linear-attention layers** (Gated DeltaNet): 16 k-heads × 128
  (key_dim 2048), 32 v-heads × 128 (value_dim 4096) — GVA, v-head i
  reads k-head i//2 INSIDE fla's kernels (never materialize the
  expansion). Pipeline:
  1. `qkvz = x_n @ W_qkvz` (d → 2·key+2·value = 12288, block-major
     [Q|K|V|Z]); `ba = x_n @ W_ba` (d → 2·HV = 64, [B|A]).
  2. depthwise causal conv1d (kernel 4) over [q|k|v] (conv_dim 8192)
     with FUSED silu — fla Triton kernel, weight (conv_dim, 4).
  3. split heads; **l2norm per head on q/k** (fp32 rstds).
  4. beta = sigmoid(b).
  5. `chunk_gated_delta_rule_fwd(q_n, k_n, v_h, a_raw, beta,
     scale=128^-0.5, use_gate_in_kernel=True, A_log, dt_bias)` — the
     decay gate g = -exp(A_log)·softplus(a+dt_bias) + chunk cumsum is
     fused in-kernel. Returns (g_post fp32 (T,HV), o, A_int bf16
     (T,HV,64), ...). fp32 recurrent state internally
     (`mamba_ssm_dtype: float32`).
  6. gated RMSNorm: `silu(z) · rmsnorm(o) · w_norm` (w over
     head_v_dim=128); `xo = o_normed @ W_out + x` (out: value_dim→d).
- Per-layer linear-attn params besides projections: conv (8192,4),
  A_log (32,), dt_bias (32,), norm (128,).
- **Tied embeddings** (a per-model CONFIG choice, not a family
  property): vocab 248,320; NO separate lm_head — the head IS the
  embedding table. Final RMSNorm before the head carries a REAL learned
  weight (model.norm.weight) — unlike our llama/qwen3 shaped families'
  weightless approximation, this one is first-class.
- MTP head (`mtp_num_hidden_layers: 1`): ignored for LM training (as
  flextrain's training path does) — recorded decision, revisit never
  unless the objective changes.
- Param count: embed 1.017B (once, tied) + 24 linear layers ×218.4M +
  8 full layers ×209.8M ≈ 7.94B (+final norm) — "9B" counts differently.

## 2. Backward contracts (from flextrain, mirrored to our style)

**DeltaNet bwd** (saved: x, qkvz, ba, g_post, A_int, core_out, both
block rstds; NOT saved: post-conv, q/k l2norms — recomputed):
1. re-run conv WITHOUT activation → pre-silu (needed by silu_bwd);
   silu → post_conv; split + l2norm → q_n,k_n,rstds (recompute).
2. o_normed = gated_rmsnorm_fwd(core_out, z) (cheap recompute);
   dW_out = o_normedᵀ dxo; do_normed = dxo @ W_outᵀ.
3. gated-rmsnorm bwd → d_core_out, dz, dW_norm.
4. `chunk_gated_delta_rule_bwd(...saved g_post, A_int...)` → dq_n,
   dk_n, dv, da, dbeta, dA_log, ddt_bias.
5. l2norm_bwd (with recomputed rstds) → dq_pre, dk_pre.
6. silu_bwd (with pre-silu) → d_post_conv; fla `causal_conv1d_bwd` →
   d_conv_in, dW_conv.
7. assemble d_qkvz = [d_conv_in | dz], d_ba = [dbeta·σ'(b) | da];
   dW_qkvz = x_nᵀ d_qkvz; dW_ba = x_nᵀ d_ba;
   dx_n = d_qkvz @ W_qkvzᵀ + d_ba @ W_baᵀ; then attn-norm bwd.

**Gated attention bwd**: saved (xk, xv, attn_result, lse, xq post-rope,
attn_gate pre-sigmoid, xo, rstds). dxo → dW_o via (attn·σ(gate)); dgated
→ d_attn = dgated·σ(gate), d_gate = dgated·attn·σ'(gate); flash bwd →
dq,dk,dv; partial-rope bwd (rotate only rot_dim); qk-norm bwd; projection
grads with the doubled [dq_rot | d_gate] concat for W_q.

## 3. Mapping to our machinery

### 3a. Heterogeneous layers (the real generalization)
`build_shaped_llama3` assumes ONE block kind. Generalize (in place,
llama/qwen3 chains must stay BIT-IDENTICAL — same ids, same order):
- The builder gains an optional per-layer kind table (default: all
  `"block"`); task ids stay `block_fwd_{s}_{r}_{i}` etc. but
  `compute_block_key` becomes kind-specific (`lin_fwd`/`attn_fwd`… or
  keep `block_fwd` + `block_params["kind"]`). DECISION: distinct
  compute_block_keys per kind (`linattn_fwd/bwd/recompute`,
  `gattn_fwd/bwd/recompute`) — profiling signatures and the kernel-set
  provenance then distinguish kinds for free; task IDS unchanged so all
  naming-dependent tooling (NVTX renamer, window oracle, m4_train
  conventions) is untouched.
- Sizes: `_mapped_size` currently maps W_/A_ uniformly. Generalize
  `apply_exact_sizes` to accept per-object-id sizes derived from a
  layer-kind table (W_{i} and A_{s}_{r}_{i} sized by kind of layer i).
- Recompute rewrites: per-layer `r_compute_block_key` by kind; saved
  bytes per kind.
- Roofline seeds: per-kind costs (linear-attn ~O(T·d²) GEMM-dominated +
  conv + delta rule ≈ treat as matmul roofline on proj+out + memory
  term; accuracy irrelevant — profiling replaces).

### 3b. Tied embeddings (second structural change — config-gated)
- One object family `W_e / dW_e / O_e` replaces the separate
  embed/head trio: lowering flag `tied_embeddings` drops
  `W_head`/`O_head`/`dW_head_*`; `embed_fwd`/`head_fwd`/`head_bwd`/
  `embed_bwd` all reference `W_e`; `dW_e_{s}` accumulates from BOTH
  `head_bwd` (round 0 creates it) AND `embed_bwd` (mutate-accumulates),
  then every later round mutates; single `optimizer_e`. Golden mirrors
  with one leaf (autograd sums the two contributions automatically).
- **Final-norm weight rides in `W_e`**: the packed embed layout gains a
  second field `[embed_table | final_norm_w(d)]` so the one-object
  contract holds — head_fwd computes `rmsnorm(x, final_norm_w) @ tableᵀ`
  and head_bwd produces the final-norm weight grad into `dW_e`'s
  matching field. (llama/qwen3 keep their weightless final norm and
  untied objects — builder changes are flag-gated, chains bit-identical.)

### 3c. Blocks (tasks/qwen35_blocks.py)
Two staged executables + shared MLP-tail stages factored from the
qwen3/llama pattern:
- `Qwen35LinBlockFwd.STAGES`: attn_norm(rstd_attn) →
  proj(qkvz, ba emitted) → conv(silu, scratch) → heads+l2norm →
  fla(g_post, A_int, core_out emitted) → gated_norm_out(xo emitted) →
  ffn_norm(rstd_ffn) → up_proj(x1,x3) → swiglu → down_resid.
  Recompute boundary = up_proj (derived).
- `Qwen35AttnBlockFwd.STAGES`: attn_norm → qkv_split_gate(xk, xv,
  attn_gate emitted) → qknorm+partial-rope(xq emitted) →
  flash(lse, attn_result) → gate_o(xo) → ffn stages.
- fla's delta-rule kernels called DIRECT (like flash-attn today, per
  tasks/README rule); gated-rmsnorm + silu_bwd as eager ops first
  (registry-fuse later); l2norm via fla's helpers; **conv as a
  kernel-REGISTRY op with two impls** — fla-triton (token-major-native,
  default) and causal-conv1d CUDA (channel-major) — measured tie at
  ~1,400 GB/s on the 5090, see §6 deps.

### 3d. Context layouts (per kind)
- Linear: rstd_attn(t), qkvz(t,12288), ba(t,64), g_post(t,32,f32),
  A_int(t,32·64), core_out(t,4096), xo(t,d), rstd_ffn(t), x1, x3.
  (No conv-state/final-state fields: we run FULL sequences per task —
  no cross-chunk machinery, `initial_state=None` always. cu_seqlens
  passed for batch>1 packed rounds: our rounds are (batch·seq) packed
  — MUST pass cu_seqlens = seq boundaries so the recurrence resets per
  sequence! llama attention got this via flash's cum_seqs; delta rule
  needs the same. TEST this explicitly.)
- Full-attn: rstd_attn, xk(t,kv), xv(t,kv), attn_gate(t,attn_dim),
  xq(t,attn_dim post-rope), lse, attn_result(t,attn_dim), xo, rstd_ffn,
  x1, x3, q/k-norm rstds? (flextrain recomputes qk-norm in bwd from
  saved pre-norm? It saves xq POST-norm-post-rope and recomputes norm
  output... follow qwen3's OUR pattern: save pre-norm q/k? head_dim 256
  norm — decide during implementation against bwd needs; prefer the
  qwen3 pattern: save pre-norm + rstds, rebuild post in bwd.)

### 3e. Golden (models/qwen35_reference.py)
Pure eager autograd, obviously correct, O(T·chunk) NOT required — use
the SEQUENTIAL recurrence at tiny scale:
  S_t = S_{t-1}·exp(g_t) + β_t·(v_t − S_{t-1}·k_t)·k_tᵀ  (per v-head,
  fp32 state, k/q l2-normalized, g_t = −exp(A_log)·softplus(a+dt_bias)),
  o_t = S_t·q_t·scale — verify exact form against fla's
  `naive`/reference implementation (fla ships
  `ops/gated_delta_rule/naive.py` — use ITS equations as the spec, our
  torch as the code).
Conv: `F.conv1d` depthwise causal + silu. Gated attention: eager sdpa
composition + sigmoid gate + partial rope reference. Tied head: single
leaf.

## 4. Shapes/config
- `ShapedQwen35Config`: 9B defaults above + `layer_kinds` derived from
  `full_attention_interval=4`; tiny() = 4 layers (LLLF), d=256,
  attn 4×64 heads (2 kv), linear 2 k-heads/4 v-heads ×32,
  conv 4, ff 512, vocab 512, partial 0.25, tied.
- Dims dataclass `Qwen35Dims` with both sub-block dim sets.

## 5. Test ladder (all before any throughput number)
1. fla-vs-naive: our golden recurrence vs `fla.ops.gated_delta_rule`
   naive reference at fp32 tiny shapes (spec-level check), then fla
   chunk kernel vs our recurrence (bf16 tolerances).
2. Ladder 1: conv+silu op, l2norm, gated-rmsnorm, partial-rope,
   gate-split ops vs references.
3. Ladder 2: `check_block_backward` for BOTH kinds (family bundle
   gains per-kind block entries — extend Family or add kind param).
4. Structural: STAGES completeness/tripwire per kind; heterogeneous
   builder invariants (kind table → sizes/keys; llama/qwen3 chains
   byte-identical to before the generalization — regression-test by
   comparing a lowered program JSON hash pre/post).
5. Ladder 3: model-step vs golden (tiny, tied embeddings); plan-
   invariance ×3; multistep golden; batch>1 cu_seqlens reset test.
6. Gates: poison-on-free + interleaving stress on the tiny config.
7. Throughput: profile + sweep qwen35-9b at 16/20 GiB.

## 6. Open items / decisions log
- Norm dtypes: flextrain keeps norm params fp32 (bf16 ULP vs AdamW lr
  bug near 1.0). Our shaped family keeps bf16 everywhere v1 (golden
  identical → self-consistent); revisit if loss curves need it.
- RoPE convention: FT pair-interleaved everywhere (matches our rope op);
  HF-weight parity is out of scope for the shaped family.
- Hopper GVA bwd workaround (fla #640): sm90-only; we're sm120 —
  verify `chunk_gated_delta_rule_bwd` works directly in ladder 1;
  fallback = flextrain's expand/reduce trick if Blackwell hits it too.
- deps — OUR OWN pins, chosen for our stack (flextrain is reference
  only): `flash-linear-attention==0.5.1` (latest; API verified — fwd
  takes `use_gate_in_kernel`/`A_log`, bwd takes `g_input`; naive
  references present; fwd smoke green on sm_120) and
  `causal-conv1d==1.6.2.post1` (built from source for sm_120 with CUDA
  13.1). Conv A/B at 9B shapes (T=8192, D=8192, W=4, silu) on the 5090:
  CUDA 0.193 ms vs fla-Triton 0.192 ms — BOTH at ~1,400 GB/s (memory-
  bound, at bandwidth; no gap on this card), both 1.7e-3 rel err vs an
  fp32 reference. Integration note: fla's kernel consumes token-major
  (T, D) natively; causal_conv1d_fn wants channel-major (B, D, T).
  DECISION: conv is a kernel-REGISTRY op with both impls registered —
  fla-triton default (layout-native), causal-conv1d as the measured
  alternate; the registry's kernel-set stamping keeps profiles honest
  either way.

## 7. Status
- [x] Recon: flextrain qwen3_5 layers/blocks read; HF config extracted;
      fwd/bwd contracts documented above.
- [x] Deps: fla 0.5.1 + causal-conv1d 1.6.2.post1 (our pins), sm_120
      smokes green, conv A/B measured (tie at bandwidth).
- [x] Heterogeneous builder + tied-embeddings lowering generalization.
- [x] Layouts + blocks + golden + lowering (families entry pending).
- [x] Ladder 1-2: kernel contracts pinned; per-kind block gradcheck green
      (fwd / recompute-equivalence / bwd all dW / 2x accumulation).
      Hard-won contract: **every tensor handed to an fla/conv Triton
      kernel must be contiguous** — the packed ``ba[:, HV:]`` column
      slice fed to ``g_input`` was read with the wrong row stride and
      silently corrupted dA_log/ddt_bias/da (~4x rel error, scaling
      with init magnitude). Not an fla version bug; 0.5.1 stands.
- [x] Families entry + ladder 3-4 + gates: model-step vs golden,
      plan-invariance (3 plans), batch=2 packed sequences (cu_seqlens
      resets pinned E2E), 3-step train vs golden, poison-on-free,
      interleaving stress, measured-cost replan. Poison found the
      layouts' alignment-padding gaps are updated by adamw from
      undefined dW padding — benign (no field reads padding); the gate
      readback masks padding and compares the model.
- [x] First sweep (pre-fusion): 1,607 wall tok/s @ 16 GiB / 1,693 @ 20,
      fidelity 0.82/1.11%, zero evictions (artifacts/m5/qwen35-first).
- [x] Fused-kernel pass: gated_rmsnorm_fwd/bwd + causal_conv1d_silu_fwd/bwd
      registry families (fla fused Triton defaults, eager reference
      fallbacks, DATAFLOW_KERNELS=eager bisection verified). fla's
      bwd recompute_output returns the PRE-GATE norm — the wrapper
      composes the silu gate (ladder caught it). The causal-conv1d
      channel-major alternate stays deferred: standalone A/B tie and a
      12-arg undocumented raw binding.
- [ ] Post-fusion sweep + results promotion.
