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


## 8b-s1k-bs8ga8 — seq 1K: bs=8, ga=8 (65,536 tokens/step)

| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 12 | 19947 | 21164 | 3286 | 3097 | -5.8% | +0.93% | 128/256 | 13.18 GiB | ×1.10 | 12.585, 12.568, 12.579 |
| 16 | 19877 | 20825 | 3297 | 3147 | -4.5% | +0.78% | 128/256 | 17.87 GiB | ×1.13 | 12.585, 12.570, 12.579 |
| 20 | 19764 | 20420 | 3316 | 3209 | -3.2% | +0.76% | 128/256 | 21.72 GiB | ×1.09 | 12.585, 12.569, 12.579 |
| 24 | 19604 | — | 3343 | — | — | — | 113/256 | — | — | placement_infeasible |
