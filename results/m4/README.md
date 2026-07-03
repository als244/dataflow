# M4 results: llama3-8B full-bf16 training, budget vs throughput

**Kernel set: `eager-v1`** — row-chunked eager-torch ops (+ aten flash-attention,
cuBLAS GEMMs). Costs are measured per task signature and feed the plans, so the
fused-kernel set (M4.2 registry) will move BOTH columns: real throughput directly,
sim throughput via cheaper measured costs — and may change recompute choices.
This table is the eager baseline for that A/B.

RTX 5090 (31.3 GiB) · bf16 params+grads+AdamW state · seq 4096 ·
measured task costs · plans built on measured bidirectional PCIe ·
static buffer placement (offsets packed offline from plan lifetimes,
validated against physical VRAM at planning time; 'geom. tax' = packed
extent / peak concurrent load, the price of contiguous placement) ·
steady-state excludes the warm-up step · full methodology in docs/m4-report.md


## 8b — bs=1, ga=1 (4,096 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 3856 | 3802 | 1062 | 1077 | +1.4% | +1.66% | 2/32 | 8.79 GiB | ×1.10 | 12.543, 12.588, 12.578 |
| 12 | 3783 | 3675 | 1083 | 1114 | +2.9% | +1.69% | 0/32 | 12.30 GiB | ×1.03 | 12.543, 12.587, 12.574 |
| 16 | 3599 | 3525 | 1138 | 1162 | +2.1% | +0.53% | 0/32 | 18.94 GiB | ×1.19 | 12.543, 12.589, 12.576 |
| 20 | 3343 | 3388 | 1225 | 1209 | -1.3% | +0.60% | 0/32 | 23.45 GiB | ×1.17 | 12.543, 12.588, 12.578 |

## 8b-bs4ga4 — bs=4, ga=4 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 12 | 27338 | 28846 | 2397 | 2272 | -5.2% | +0.36% | 64/128 | 13.88 GiB | ×1.16 | 12.584, 12.577, 12.576 |
| 16 | 27017 | 28120 | 2426 | 2331 | -3.9% | +0.33% | 64/128 | 17.33 GiB | ×1.09 | 12.584, 12.575, 12.576 |
| 18 | 27278 | 28107 | 2403 | 2332 | -2.9% | +0.37% | 64/128 | 19.27 GiB | ×1.08 | 12.584, 12.577, 12.576 |

## 8b-ga16 — bs=1, ga=16 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 33123 | 33757 | 1979 | 1941 | -1.9% | +2.79% | 388/512 | 8.96 GiB | ×1.12 | 12.543, 12.602, 12.559 |
| 12 | 32620 | 32236 | 2009 | 2033 | +1.2% | +4.68% | 256/512 | 13.28 GiB | ×1.11 | 12.543, 12.601, 12.561 |
| 16 | 30730 | 29554 | 2133 | 2217 | +4.0% | +3.48% | 3/512 | 18.11 GiB | ×1.13 | 12.543, 12.601, 12.559 |
| 20 | 26843 | 27551 | 2441 | 2379 | -2.6% | +1.25% | 1/512 | 21.83 GiB | ×1.09 | 12.543, 12.600, 12.560 |

## baseline1b — 1B-class baseline config

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 24 | 530 | 399 | 7730 | 10258 | +32.7% | +1.94% | 0/16 | — | — | 10.825, 10.809, 10.794, 10.795 |

## Plain-torch baseline (configs that fit in VRAM)

- **baseline1b**: plain eager torch 355.9 ms/step (11508 tok/s); this runtime at 24 GiB: 10258 tok/s (**89%** of plain torch).

## Batching comparison (real tok/s by budget)

| budget (GiB) | bs=1, ga=1 (4,096 tokens/step) | bs=4, ga=4 (65,536 tokens/step) | bs=1, ga=16 (65,536 tokens/step) |
|---:|---:|---:|---:|
| 8 | 1077 | — | 1941 |
| 12 | 1114 | 2272 | 2033 |
| 16 | 1162 | 2331 | 2217 |
| 18 | — | 2332 | — |
| 20 | 1209 | — | 2379 |

## Correctness under pressure: runtime vs plain-torch golden

Full train step through the real engine vs the golden eager-torch model (16L, d=2048, seq 4096): relative-L2 of loss + every final weight after the optimizer step. Memory pressure must not perturb the math — errors stay at bf16 noise at every budget.

| budget (GiB) | recompute | worst tensor | worst rel-L2 | status |
|---:|:---:|:---|---:|:---|
| 6 | — | W_0 | 6.76e-04 | ok |
| 4 | — | W_0 | 6.74e-04 | ok |
| 3 | — | W_0 | 6.76e-04 | ok |
| 2.5 | all | W_0 | 6.75e-04 | ok |
| 2 | all | W_0 | 6.75e-04 | ok |
