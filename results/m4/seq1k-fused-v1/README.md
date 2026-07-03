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


## 8b-s1k-bs2ga32 — seq 1K: bs=2, ga=32 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 12 | 48164 | 40521 | 1361 | 1617 | +18.9% | +4.33% | 1024/1024 | 13.17 GiB | ×1.10 | 12.569, 12.560, 12.565 |
| 16 | 43848 | 36729 | 1495 | 1784 | +19.4% | +4.51% | 1024/1024 | 16.75 GiB | ×1.05 | 12.569, 12.558, 12.566 |
| 24 | 29900 | 26099 | 2192 | 2511 | +14.6% | +7.48% | 0/1024 | 25.77 GiB | ×1.07 | 12.569, 12.558, 12.565 |
| 30 | 24904 | — | 2632 | — | — | — | 0/1024 | — | — | placement_infeasible |

## 8b-s1k-bs4ga16 — seq 1K: bs=4, ga=16 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 12 | 28581 | 27276 | 2293 | 2403 | +4.8% | +3.51% | 428/512 | 13.49 GiB | ×1.13 | 12.566, 12.578, 12.541 |
| 16 | 26727 | 26101 | 2452 | 2511 | +2.4% | +2.85% | 380/512 | 17.78 GiB | ×1.11 | 12.566, 12.579, 12.541 |
| 20 | 23518 | 23853 | 2787 | 2747 | -1.4% | +2.35% | 256/512 | 21.37 GiB | ×1.07 | 12.566, 12.580, 12.542 |
| 24 | 20482 | 21200 | 3200 | 3091 | -3.4% | +1.16% | 166/512 | 25.03 GiB | ×1.04 | 12.566, 12.579, 12.542 |
| 30 | 18897 | — | 3468 | — | — | — | 0/512 | — | — | placement_infeasible |

## 8b-s1k-bs8ga8 — seq 1K: bs=8, ga=8 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 12 | 20578 | 21783 | 3185 | 3009 | -5.5% | +0.89% | 128/256 | 13.08 GiB | ×1.10 | 12.585, 12.568, 12.578 |
| 16 | 20523 | 21496 | 3193 | 3049 | -4.5% | +0.86% | 128/256 | 17.87 GiB | ×1.13 | 12.585, 12.568, 12.579 |
| 20 | 20428 | 21209 | 3208 | 3090 | -3.7% | +0.85% | 128/256 | 21.46 GiB | ×1.07 | 12.585, 12.568, 12.579 |
| 24 | 19527 | 20164 | 3356 | 3250 | -3.2% | +0.72% | 76/256 | 26.96 GiB | ×1.12 | 12.585, 12.568, 12.580 |
| 30 | 17864 | — | 3669 | — | — | — | 0/256 | — | — | placement_infeasible |

## 8b-s1k-bs16ga4 — seq 1K: bs=16, ga=4 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 12 | 19915 | 21308 | 3291 | 3076 | -6.5% | +0.47% | 64/128 | 13.88 GiB | ×1.16 | 12.575, 12.580, 12.565 |
| 16 | 19888 | 21045 | 3295 | 3114 | -5.5% | +0.51% | 64/128 | 17.33 GiB | ×1.09 | 12.575, 12.581, 12.566 |
| 20 | 19815 | 20870 | 3307 | 3140 | -5.1% | +0.48% | 64/128 | 22.05 GiB | ×1.11 | 12.575, 12.579, 12.566 |
| 24 | 19795 | 20764 | 3311 | 3156 | -4.7% | +0.44% | 64/128 | 26.52 GiB | ×1.11 | 12.575, 12.581, 12.566 |
| 30 | 18317 | — | 3578 | — | — | — | 0/128 | — | — | placement_infeasible |

## Batching comparison (real tok/s by budget)

| budget (GiB) | seq 1K: bs=2, ga=32 (65,536 tokens/step) | seq 1K: bs=4, ga=16 (65,536 tokens/step) | seq 1K: bs=8, ga=8 (65,536 tokens/step) | seq 1K: bs=16, ga=4 (65,536 tokens/step) |
|---:|---:|---:|---:|---:|
| 12 | 1617 | 2403 | 3009 | 3076 |
| 16 | 1784 | 2511 | 3049 | 3114 |
| 20 | — | 2747 | 3090 | 3140 |
| 24 | 2511 | 3091 | 3250 | 3156 |
| 30 | — (sim 2632) | — (sim 3468) | — (sim 3669) | — (sim 3578) |
