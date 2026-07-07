
### STATIC placement

| dev GiB | olmoe-7b |
|---|---|
| 12 | 12,220 (sim 11,956) · 11.30 GiB · bs32ga2 · rc 89% |
| 16 | 14,154 (sim 13,837) · 15.93 GiB · bs64ga1 · rc 72% |
| 20 | 14,793 (sim 14,543) · 19.96 GiB · bs64ga1 · rc 83% |
| 24 | 15,726 (sim 15,736) · 23.51 GiB · bs64ga1 · rc 44% |
| 28 | 16,222 (sim 16,592) · 27.78 GiB · bs64ga1 · rc 44% |

### Best legal per cell (mode in parens where not static)

| dev GiB | olmoe-7b |
|---|---|
| 12 | 12,220 |
| 16 | 14,154 |
| 20 | 14,793 |
| 24 | 15,726 |
| 28 | 16,222 |

Per-cell artifacts: `cells/{preset}-{dev}gib-{mode}/` — measured.json (full row), plan.json (replayable via bench_train --annotated), program.json (webapp-simulator upload). Raw bench_train output (summaries, plans, logs): `raw/`.
