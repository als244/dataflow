# Tools

One directory per purpose; every tool runs from the repo root as
`python tools/<dir>/<name>.py`.

## train/ — running training
| tool | purpose |
|---|---|
| `train_solo.py` | single-GPU pretraining: `engine` and `reference` legs (checkpoints/resume, doc-aware data, profiler bracketing), `smoke`/`parity`/`scaling` studies, `peek` (in-flight loss curve from the newest checkpoint) |
| `eval_checkpoint.py` | fineweb-VAL loss of a solo-run checkpoint (the nanogpt-comparable axis) |
| `train_fleet.py` | data-parallel fleet twin: `train` (multi-daemon launch, per-rank logs, `--profile` per-rank nsys reports), `compare` (overlay finished runs), `sweep` (single-GPU vs distributed comparison grid) |
| `dataflowd.py` | engine service daemon CLI: start / status / stop |
| `daemonize.py` | launch-and-detach (POSIX double fork) for long runs |
| `pretrain_report.py` | build the pretraining study reports (self-contained HTML) |

## bench/ — throughput: predict → measure → profile
The escalating-cost workflow ([benchmarking.md](../docs/benchmarking.md)):
| tool | purpose |
|---|---|
| `predict_step.py` | FIRST LINE OF ATTACK (CPU, instant): simulated sweeps over geometry × memory — s/step, tok/s, effective/hardware TFLOPs/s, memory peaks, PCIe traffic, recompute/idle % ([throughput.md](../docs/throughput.md)) |
| `measure_step.py` | the measured twin (GPU, minutes): same grid, each cell RUN on the engine — measured s/step beside the prediction for that cell's plan |
| `nsys_profile.py` | one Nsight Systems capture of a solo engine run, bracketed to exact warmed steps |

`bench/internal/` holds maintainer-local kernel microbenches
(gitignored — not part of the repo surface).

## verify/ — correctness gates
| tool | purpose |
|---|---|
| `verify_family.py` | one-command family correctness: canonical ladder + canon audit ([extending.md](../docs/extending.md) §8) |
| `engine_gate.py` | real-GPU synthetic execution vs simulator prediction |
| `pressure_correctness.py` | math invariance under memory pressure: engine at descending tight budgets vs the plain-torch golden trajectory |
| `deep_compare.py` | deep correctness-compare treatment for one family × shape ([correctness_compare.md](../docs/correctness_compare.md)) |
| `sweep_ladder3.py` | ladder-3 measurement sweep across families |
| `rdma_preflight.py` | RDMA peer-plane preflight checks |

## export/ — run analysis & webapp export
| tool | purpose |
|---|---|
| `export_program.py` | CPU-only end-to-end for any preset: shaped program → plan → sim → webapp exports |
| `trace_real_run.py` | a few REAL steps through the daemon → measured-vs-simulated webapp bundle ([exporting_runs.md](../docs/exporting_runs.md)) |
| `trace_program.py` | event-timeline trace of ANY program on the fake backend (reserves, transfer charges, evictions/escapes) — plan debugging without a GPU |

## gen_model_docs/ — generated docs
| tool | purpose |
|---|---|
| `gen_model_docs.py` | regenerate docs/models/ (every family × preset) |
| `gen_model_page.py` | one model page at an arbitrary run shape |
| `list_models.py` | regenerate docs/builtin_models.md |
| `list_kernels.py` | regenerate docs/kernel_registry.md |
| `list_tasks.py` | regenerate docs/task_kinds.md |

`battery.sh` (this directory) is the rsync-driven remote battery runner
for cross-box validation.
