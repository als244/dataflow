# Tools

One directory per purpose; every tool runs from the repo root as
`python tools/<dir>/<name>.py`. Each directory's README describes its
tools' purposes and CLI arguments in full.

| directory | contents |
|---|---|
| [train/](train/README.md) | running training: `train.py` (ONE tool, every world size — zero-config solo = world-1 fleet; engine/reference legs, smoke/parity/scaling studies, `peek`, `compare`, `--profile` nsys wrap), `dataflowd`, `daemonize` (+`--kill`), `eval_checkpoint`, `fetch_dataset`, `pretrain_report` |
| [bench/](bench/README.md) | throughput, in escalating cost: `predict_step` (simulated sweeps — the first line of attack), `measure_step` (real sweeps); Nsight captures ride `train.py --profile` |
| [verify/](verify/README.md) | correctness gates: `verify_family`, `engine_gate`, `pressure_correctness`, `deep_compare`, `sweep_ladder3`, `rdma_preflight` |
| [export/](export/README.md) | run analysis & webapp export: `export_program`, `trace_real_run`, `trace_program` |
| [gen_model_docs/](gen_model_docs/README.md) | generated docs: `gen_model_docs`, `gen_model_page`, `list_models`, `list_kernels`, `list_tasks` |

`battery.sh` (this directory) is the rsync-driven remote battery
runner for cross-box validation
(`tools/battery.sh <remote-host> <remote-path>`).

The doc-side guides these tools implement:
[benchmarking.md](../docs/benchmarking.md),
[throughput.md](../docs/throughput.md),
[distributed_training.md](../docs/distributed_training.md),
[exporting_runs.md](../docs/exporting_runs.md),
[extending.md](../docs/extending.md) §8,
[engine_service.md](../docs/engine_service.md).
