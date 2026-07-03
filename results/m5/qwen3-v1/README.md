# Qwen3-8B (dense, qk-norm) — first family extension (M5.1)

Qwen3-8B-shaped: 36L, d=4096, GQA 32/8, head_dim 128, d_ff 12288, vocab
151,936 (untied embed/head), qk-norm (per-head RMSNorm on q/k between
projection and rope), rope theta 1e6, bf16 AdamW. seq 1024, 65,536
tok/step (bs=8/ga=8), 3 steps, recompute planner, static placement,
interleaved optimizer, task0 pre-placement — the full llama pipeline,
reached through docs/extending.md §6.

| budget | sim tok/s | real | wall | real vs sim | fidelity | recompute | escapes/evictions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 GiB | 3,438 | 3,228 | **3,221** | −6.1% | +1.56% | 144/288 | 0 / 0 |
| 20 GiB | 3,487 | 3,295 | **3,289** | −5.5% | +1.72% | 144/288 | 0 / 0 |

Context: llama3-8B at the same budgets/tokens runs 3,341/3,365 wall —
qwen3 lands ~2-4% below with ~2% more params (8.19B vs 8.03B), 4 more
layers' worth of per-task host boundaries (809 vs 714 tasks/step), and a
larger CE/head (vocab 151,936 vs 128,256).

## What the extension actually required (extending.md §6 exercised)

- **Zero new kernels**: qk-norm is the rmsnorm registry family at
  head_dim-wide rows (tokens*heads of them) through reshaped views.
- New: `tasks/qwen3_blocks.py` (STAGES with qkv→qk_norm→rope; recompute
  DERIVED, boundary unchanged at up_proj; backward rebuilds flash's q/k
  from saved pre-norm qm/km + per-head rstds — cheap elementwise),
  `models/qwen3_reference.py`, `training/{shaped_qwen3,qwen3_lowering}.py`
  (the chain builder is family-generic and reused), qwen3 ladder tests.
- Generalized: a `training/families.py` registry now dispatches
  lowering/dims/resolver/golden/gradcheck per config type — train loop,
  gradcheck, and m4_train no longer hardcode llama.
- Latent bug the new family exposed: optimizer state was sized from
  field-sum elems while the AdamW executable derives elems from packed
  bytes — llama's fields are all 256-aligned so they agreed by luck;
  qwen3's 128-byte norm weights broke the coincidence. O_* is now sized
  from packed bytes in both lowerings.

Gates: full ladder green (qk-norm op, block bwd incl. dW_q/k_norm,
model-step vs golden, plan-invariance across 3 plans, 3-step multistep
golden with decreasing loss); suites 78 CPU + 46 GPU.
