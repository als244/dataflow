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
| 12 | 19657 | 20607 | 3334 | 3180 | -4.6% | +1.61% | 128/256 | 13.24 GiB | ×1.11 | 12.557, 12.588, 12.590 |
| 16 | 19083 | 20372 | 3434 | 3217 | -6.3% | +0.95% | 128/256 | 17.87 GiB | ×1.14 | 12.557, 12.588, 12.590 |
| 20 | 19002 | 20041 | 3449 | 3270 | -5.2% | +0.79% | 128/256 | 22.68 GiB | ×1.13 | 12.557, 12.588, 12.591 |
| 24 | 18587 | — | 3526 | — | — | — | 97/256 | — | — | placement_infeasible |
