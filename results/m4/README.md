# Kernel-set A/B: `eager-v1` vs `fused-v1`

Same programs, plans re-derived per set from re-measured task costs —
kernel changes move BOTH the sim prediction and the real run (and can
shift the planner's recompute choices). tok/s = real steady-state;
(sim NNNN) = the simulator's prediction for that set's plan.


## 8b — bs=1, ga=1 (4,096 tokens/step)

| budget (GiB) | eager-v1 tok/s | fused-v1 tok/s | real speedup | eager-v1 recompute | fused-v1 recompute |
|---:|---:|---:|---:|---:|---:|
| 8 | 1077 (sim 1062) | 1171 (sim 1130) | +8.7% | 2 | 13 |
| 12 | 1114 (sim 1083) | 1248 (sim 1180) | +12.0% | 0 | 6 |
| 16 | 1162 (sim 1138) | 1290 (sim 1161) | +11.0% | 0 | 0 |
| 20 | 1209 (sim 1225) | 1357 (sim 1228) | +12.2% | 0 | 0 |

## 8b-bs4ga4 — bs=4, ga=4 (65,536 tokens/step)

| budget (GiB) | eager-v1 tok/s | fused-v1 tok/s | real speedup | eager-v1 recompute | fused-v1 recompute |
|---:|---:|---:|---:|---:|---:|
| 12 | 2272 (sim 2397) | 2924 (sim 3121) | +28.7% | 64 | 64 |
| 16 | 2331 (sim 2426) | 2985 (sim 3117) | +28.1% | 64 | 64 |
| 18 | 2332 (sim 2403) | 2991 (sim 3134) | +28.3% | 64 | 64 |

## 8b-ga16 — bs=1, ga=16 (65,536 tokens/step)

| budget (GiB) | eager-v1 tok/s | fused-v1 tok/s | real speedup | eager-v1 recompute | fused-v1 recompute |
|---:|---:|---:|---:|---:|---:|
| 8 | 1941 (sim 1979) | 2249 (sim 2191) | +15.8% | 388 | 461 |
| 12 | 2033 (sim 2009) | 2339 (sim 2250) | +15.1% | 256 | 429 |
| 16 | 2217 (sim 2133) | 2479 (sim 2459) | +11.8% | 3 | 256 |
| 20 | 2379 (sim 2441) | 2657 (sim 2651) | +11.7% | 1 | 275 |
