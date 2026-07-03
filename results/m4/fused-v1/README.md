# M4 results: llama3-8B full-bf16 training, budget vs throughput

**Kernel set: `fused-v1`** — registry ops all resolved to ['triton'] (aten flash-attention + cuBLAS GEMMs stay direct).
Costs are measured per task signature and feed the plans: kernel changes
move BOTH real throughput and the sim prediction.

RTX 5090 (31.3 GiB) · bf16 params+grads+AdamW state · seq 4096 ·
measured task costs · plans built on measured bidirectional PCIe ·
static buffer placement (offsets packed offline from plan lifetimes,
validated against physical VRAM at planning time; 'geom. tax' = packed
extent / peak concurrent load, the price of contiguous placement) ·
steady-state excludes the warm-up step · full methodology in docs/m4-report.md


## 8b — bs=1, ga=1 (4,096 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 3625 | 3499 | 1130 | 1171 | +3.6% | +0.96% | 13/32 | 8.97 GiB | ×1.12 | 12.544, 12.589, 12.577 |
| 12 | 3471 | 3283 | 1180 | 1248 | +5.7% | +0.65% | 6/32 | 12.26 GiB | ×1.02 | 12.544, 12.589, 12.575 |
| 16 | 3528 | 3175 | 1161 | 1290 | +11.1% | +0.93% | 0/32 | 18.28 GiB | ×1.15 | 12.544, 12.589, 12.576 |
| 20 | 3334 | 3018 | 1228 | 1357 | +10.5% | +0.68% | 0/32 | 23.45 GiB | ×1.17 | 12.544, 12.589, 12.574 |

## 8b-bs4ga4 — bs=4, ga=4 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 12 | 21000 | 22417 | 3121 | 2924 | -6.3% | +0.59% | 64/128 | 13.88 GiB | ×1.16 | 12.584, 12.576, 12.577 |
| 16 | 21027 | 21955 | 3117 | 2985 | -4.2% | +0.43% | 64/128 | 17.33 GiB | ×1.09 | 12.584, 12.576, 12.577 |
| 18 | 20914 | 21912 | 3134 | 2991 | -4.6% | +0.42% | 64/128 | 19.28 GiB | ×1.08 | 12.584, 12.575, 12.577 |

## 8b-ga16 — bs=1, ga=16 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8 | 29914 | 29141 | 2191 | 2249 | +2.7% | +3.31% | 461/512 | 9.20 GiB | ×1.15 | 12.544, 12.601, 12.559 |
| 12 | 29126 | 28018 | 2250 | 2339 | +4.0% | +3.92% | 429/512 | 13.70 GiB | ×1.14 | 12.544, 12.600, 12.559 |
| 16 | 26646 | 26431 | 2459 | 2479 | +0.8% | +2.88% | 256/512 | 17.68 GiB | ×1.11 | 12.544, 12.601, 12.561 |
| 20 | 24719 | 24666 | 2651 | 2657 | +0.2% | +2.96% | 275/512 | 21.34 GiB | ×1.07 | 12.544, 12.602, 12.560 |

## Batching comparison (real tok/s by budget)

| budget (GiB) | bs=1, ga=1 (4,096 tokens/step) | bs=4, ga=4 (65,536 tokens/step) | bs=1, ga=16 (65,536 tokens/step) |
|---:|---:|---:|---:|
| 8 | 1171 | — | 2249 |
| 12 | 1248 | 2924 | 2339 |
| 16 | 1290 | 2985 | 2479 |
| 18 | — | 2991 | — |
| 20 | 1357 | — | 2657 |

## Correctness under pressure: runtime vs plain-torch golden

Full train step through the real engine vs the golden eager-torch model (16L, d=2048, seq 4096): relative-L2 of loss + every final weight after the optimizer step. Memory pressure must not perturb the math — errors stay at bf16 noise at every budget.

| budget (GiB) | recompute | worst tensor | worst rel-L2 | status |
|---:|:---:|:---|---:|:---|
| 6 | — | W_0 | 1.08e-03 | ok |
| 4 | — | W_0 | 1.08e-03 | ok |
| 3 | — | W_0 | 1.08e-03 | ok |
| 2.5 | all | W_0 | 1.08e-03 | ok |
| 2 | all | W_0 | 1.08e-03 | ok |
