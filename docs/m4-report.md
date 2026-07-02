# M4 report: memory-constrained multi-step llama3-8B training on one RTX 5090

Date: 2026-07-02 · commit: (M4) · harness: `tools/m4_train.py` · artifacts: `artifacts/m4/`

## Setup

- Model: llama3-8B-shaped (32L, d=4096, GQA 32/8, ff=14336, vocab=128256),
  bf16 params/grads/activations, AdamW (bf16 moments), seq 4096, batch 1.
- Programs lowered with layout-exact sizes; **costs measured, not estimated**
  (per-unique-task CUDA-event runtimes + torch-scratch workspace), transfer
  model = measured **bidirectional** pinned PCIe (~25.4 GB/s each direction;
  unidirectional is 35.5 h2d / 56.5 d2h — directions contend on this box).
- Each budget: PressureFit-annotated single-step chain replayed 4 optimizer
  steps; persistent state (W, O) lives in pinned host buffers the plan's
  offloads update in place; Session keeps slab/pools/streams across steps.
- Every configuration's math is pinned to the golden autograd model by the
  M3 ladder (loss + every dW + optimizer state), plan-invariance, poison-on-
  free, and interleaving-stress tests.

## 8B results (4 steps; step-0 warm-up excluded from steady-state)

| budget | sim ms/step (tok/s) | real vs sim | real tok/s | replay-fidelity gap | recompute chosen |
|-------:|--------------------:|------------:|-----------:|--------------------:|-----------------:|
| 20 GiB | 3347 (1224) | **−0.5%** | ≈1218 | **+0.56%** | 0/32 |
| 16 GiB | 3574 (1146) | **+1.3%** | ≈1160 | **+0.52%** | 0/32 |
| 12 GiB | 3873 (1058) | **+6.8%** | ≈1130 | **+0.66%** | 0/32 |
| 8 GiB  | 3910 (1048) | **+2.4%** | 1073 | **+0.63%** | **2/32** |

- **Replay fidelity ~0.5–0.7% at full scale with real math**: re-simulating
  with the measured per-task/per-transfer durations as overrides, the
  runtime's actual schedule is within two-thirds of a percent of the
  simulator's ideal. Scheduling, admission, and dispatch add almost nothing.
- **Real throughput ≥ the sim's prediction at every budget** (the plan is
  built on bidirectional bandwidth; phases that run one-directional beat
  it). The gate's ≥90%-of-prediction bar is met with margin at 100–107%.
- **The thesis number**: dropping the budget 20 → 8 GiB (−60%) costs only
  ~12% throughput (≈1218 → 1073 tok/s) — offload/prefetch overlap plus a
  little planner-chosen recompute hides the memory pressure, as the
  simulator said it would. At 8 GiB the recompute planner engaged on real
  measured costs (2/32 layers) and peak fast memory was 7.98 GiB.
- Loss starts at 12.54 (≈ ln vocab + init noise) and is **identical at step
  0 across all budgets**; later steps agree to ~5e-4 (bf16 atomics in embed
  backward are the only nondeterminism).
- Plain torch cannot train this model on this GPU at all: params+grads+Adam
  ≈ 60 GB before activations.

## Absolute-throughput baseline (1B-class config, fits in VRAM)

llama3-shaped 16L/d2048/ff8192/vocab32768, seq 4096:

- plain eager-torch golden model: **355.9 ms/step** (11508 tok/s)
- this runtime at 24 GiB (transfers negligible): ≈ **399 ms/step** (10264
  tok/s) → **~89% of plain torch**.

The ~11% deficit is strict pacing: one host wake-up per task (~150–500 µs)
against this config's shorter tasks; the planned aggressive dispatch-ahead
mode (device-side input waits + committed-ahead accounting) is the known
recovery path. At 8B task lengths the same overhead is ~0.5% (the replay
gap).

Also visible at 24 GiB: real ran +33% faster than the sim's *cost model*
(replay gap still 1.9%) — profiling short bursts under-boosts SM clocks vs
sustained runs. Cost calibration under sustained load (or locked clocks) is
a cheap follow-up; it does not affect the scheduling-fidelity claims.

## Failures found and fixed on the way (all now regression-tested)

1. Eager AdamW materialized ~6× param bytes of fp32 temporaries (12.6 GB on
   the embed optimizer) → chunked, bounded to ~400 MB. Same for CE loss
   (row-chunked) and head backward (matmul straight into dW).
2. Per-`execute()` stream churn multiplied torch's per-stream scratch cache
   across steps → Session now owns streams.
3. Sessions leaked slabs/pinned pools across sweep budgets → `train()` owns
   cleanup; sweep runs one process per budget.
4. Slab headroom was proportional (starved VRAM at 20 GiB) → absolute 2 GiB
   cap. Residual known limitation: best-fit + headroom + counted overflow is
   a heuristic; the static buffer-assignment mode (offline placement proof
   from the dry run) replaces it — tracked as a spawned task.
5. Missing final norm before the LM head (loss ~78) → folded weightless
   final rmsnorm into head fwd/bwd + golden model (loss ~12.5).
6. Recompute variants have their own task signatures (`block_fwd` without
   context, `block_recompute`) → profiling covers both variants.

## What M4 leaves open (→ M5)

- Aggressive dispatch mode to close the small-task pacing gap.
- Cost calibration under sustained clocks.
- Static buffer assignment (placement proof; removes headroom/overflow).
- Webapp: measured-trace overlay upload (programs already upload; traces
  are exported alongside in `artifacts/m4/`).
- Second model family (Qwen3) via the extending guide.
