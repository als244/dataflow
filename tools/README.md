# Tools

Flat directory; every tool runs from the repo root as
`python tools/<name>.py`. Grouped by purpose:

## Sweeps & benchmarking
| tool | purpose |
|---|---|
| `bench_frontier.py` | THE sweep driver: presets × device budgets → all-legal throughput tables with per-cell provenance ([benchmarking.md](../docs/benchmarking.md)) |
| `bench_train.py` | one (config, budget) training run: profile → plan → execute → measured row + webapp/annotated exports |
| `best_config.py` | profiling oracle: picks best batch/grad-accum shape per (preset, budget) envelope |
| `bench_tables.py` | render measured rows into the summary tables |

## Kernel & primitive microbenches
| tool | purpose |
|---|---|
| `bench_moe_kernels.py` | per-op MoE kernel head-to-heads at target shapes |
| `bench_qwen35_kernels.py` | fla delta-rule + conv A/B at qwen3.5 shapes |
| `bench_vmm.py` | VMM slab primitive latencies |

## Correctness gates
| tool | purpose |
|---|---|
| `verify_family.py` | one-command family correctness: canonical ladder + canon audit ([extending.md](../docs/extending.md) §8) |
| `engine_gate.py` | real-GPU synthetic execution vs simulator prediction |
| `pressure_correctness.py` | PressureFit plan legality regression |
| `golden_path.py` | CPU-only end-to-end: shaped program → plan → sim → webapp exports |

## Run analysis & export
| tool | purpose |
|---|---|
| `gap_analysis.py` | decompose one cell's real-vs-sim gap ([exporting_runs.md](../docs/exporting_runs.md)) |
| `export_measured_run.py` | package a traced run for webapp upload |
| `nsys_profile.py` | wrap a run in Nsight Systems capture |
| `window_plans.py` | window-oracle planning analysis for a config |
| `trace_program.py` | event-timeline trace of ANY program on the fake backend (reserves, transfer charges, evictions/escapes) — plan debugging without a GPU |

## Generated docs
| tool | purpose |
|---|---|
| `gen_model_docs.py` | regenerate docs/models/ (every family × preset) |
| `gen_model_page.py` | one model page at an arbitrary run shape |
| `list_models.py` | regenerate docs/builtin_models.md |
| `list_kernels.py` | regenerate docs/kernel_registry.md |
| `list_tasks.py` | regenerate docs/task_kinds.md |
