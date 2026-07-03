# seq-1K bs8/ga8 — step-boundary fix (optimizer interleave + honest head)

llama3-8B bf16, seq 1024, 65,536 tok/step, 3 steps, recompute planner,
static placement, fused kernel set. New defaults: `optimizer_placement=
interleaved` (each optimizer task at its gradient's final mutation, inside
the last backward round) + `preplace=task0` (only task 0's inputs pre-placed;
everything else planned, charged prefetches). Diagnosis + invariant analysis:
docs/notes/step-boundary.md. Baseline: results/m4/seq1k-fused-v2 (tail +
greedy, same code otherwise).

## Headline (wall tok/s = full step: fill + execute + readback)

| budget | baseline wall | NEW wall | delta | NEW real (makespan) | NEW sim | real vs sim |
|---:|---:|---:|---:|---:|---:|---:|
| 12 GiB | 3,104 | **3,331** | **+7.3%** | 3,337 | 3,471 | −3.9% |
| 16 GiB | 3,091 | **3,341** | **+8.1%** | 3,346 | 3,506 | −4.6% |
| 20 GiB | 3,147 | **3,365** | **+6.9%** | 3,370 | 3,563 | −5.4% |
| 23.75 GiB (probe-max) | 3,187 (@24) | **3,501** | **+9.9%** | 3,507 | 3,716 | −5.6% |

flextrain ceiling on this machine (same tokens, full bf16, VRAM-flat
17→29 GiB): **3,410–3,435 tok/s** → the probe-max row is **+1.9–2.7%
ABOVE flextrain**; 12–20 GiB rows are within 1.3–2.6% of it at budgets
flextrain cannot reach (its curve is VRAM-insensitive; ours holds 97% of
the flextrain ceiling at 12 GiB).

Wall−makespan seam is now ~0.03–0.06 s/step (was 0.32–0.42 s): the sim
and wall are directly comparable for the first time. Losses match the
baseline table to the printed digits at every budget (same math under a
reordered chain = plan-invariance at scale). placement_escapes = 0 and
pressure_evictions = 0 in every row.

## What changed and why (measured mechanism)

Old plans wasted ~2 s/step at the seam: a 1.5–2.0 s GPU-idle PCIe drain
(all 34 optimizer tasks after all rounds: 30 GiB O round-trip + 15 GiB W
writebacks + 13 GiB W re-prefetch with no compute left to hide under) +
0.3–0.4 s of synchronous pre-placement upload the simulator never
charged (greedy t=0 placement is free in sim time). Interleaving folds
the optimizer traffic under the last backward round (sim: GPU-idle tail
1.52–2.00 s → 0.13–1.15 s; see sim-old-vs-new.json) and task0
pre-placement turns the setup upload into planned overlapped prefetches
(14.96 → 0.98 GiB synchronous).

## Notes

- 24 GiB flat budget is placement-INFEASIBLE for the interleaved chain
  (extent 29.10 GiB > 27 physical; geometry tax ×1.21 at the top —
  interleaving overlaps more lifetimes in the last bwd round). probe-max
  lands at 23.75 GiB (extent 25.86). The tax is the known VMM-fixable
  residual.
- The planner ADAPTS per mode (recompute 176/256 at 20 GiB vs 128
  legacy; 138 at 23.75) — cost-model-driven, not hand-tuned.
- bs2ga32 @ 20 (the historical deadlock corner): 2,191 wall tok/s (was
  2,096 with 66 placement escapes/3 steps) — now ZERO escapes; the
  planner chose recompute-all (1,024/1,024) and real runs +16.5% above
  sim (the documented small-task cost bias, sign unchanged).
- real-vs-sim widened ~1–2 pts (−3.1 → −3.9…−5.6): the fixed host tax
  (~815 µs/boundary, more tasks at higher recompute counts) is a larger
  fraction of a shorter step, and the sim's tighter overlap leaves less
  slack to absorb it. Replay-fidelity gap stays 0.9–1.3%. The tax's fix
  remains CUDA-graph-per-task (M5).
