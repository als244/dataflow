# tools/train/ — running training

Everything that launches, inspects, or reports on training runs. All
commands run from the repo root.

## train_solo.py — single-GPU pretraining orchestration

Subcommand-driven; every leg shares one deterministic data pipeline,
the warmup+cosine recipe, and byte-identical seeded init.

**`engine`** — engine-only long run through the service daemon.

| flag | meaning |
|---|---|
| `--preset` | any `resolve_preset` name (docs/builtin_models.md) |
| `--steps` | optimizer steps |
| `--opt {adamw,muon}` | override the preset's optimizer policy |
| `--peak-lr` | recipe peak (min = peak/10, warmup = steps/10, cosine) |
| `--muon-lr` | muon params' own PEAK lr riding the same schedule shape (default: share `--peak-lr`) |
| `--budget` | device fast-memory budget (GiB) the plan is fit to |
| `--slab` | daemon pinned-host slab (GiB) |
| `--batch` / `--ga-rounds` | round geometry overrides (tokens/step = seq × batch × rounds) |
| `--checkpoint-every` | host-local snapshot every N steps (keep-last-3) |
| `--resume` | continue from the newest complete checkpoint |
| `--measured` | plan with PROFILED task costs + measured PCIe (the `[plan]` line becomes the true-profiling sim prediction) |
| `--data SPEC` | data source spec ([data_feeds.md](../../docs/data_feeds.md)); default: the in-repo shard corpus, per-document |
| `--packing-policy {ffd,greedy}` / `--allow-round-split` / `--capture PATH` | packing policy, legacy exact-fill split, sequence capture for replay |
| `--profile` + `--profile-start-before-step N` / `--profile-stop-after-step M` | bracket a step window via the daemon's `profiler_control` (run under `tools/bench/nsys_profile.py` to capture) |
| `--out` | run-curve JSON (also names the checkpoint dir) |

**`reference`** — the pure-torch twin leg (same conventions):
`--preset --steps --opt --peak-lr --ga-rounds --grad-checkpoint
--checkpoint-every --resume --data/--packing-policy/
--allow-round-split/--capture --out`.

**`smoke`** — tiny real-vocab reference-vs-engine gate
(`--steps --budget --slab`).

**`parity`** — one preset, reference + engine at N budgets
(`--preset --steps --budgets --slab --grad-checkpoint`).

**`scaling`** — the preset ladder on one backend
(`--presets a,b,c --backend {reference,engine} --steps --budget
--slab --grad-checkpoint`).

**`peek RUN`** — read an in-flight (or finished) run's loss curve
from its newest checkpoint manifest; prints last/EMA/min and writes
`results/pretrain/<RUN>_partial.json` (`--ema`, default 0.98).

## eval_checkpoint.py — held-out val loss of a checkpoint

The published-curve axis: loads `W_*` straight from a snapshot
payload into the family's pure-torch twin and evaluates held-out val
CE.

    python tools/train/eval_checkpoint.py RUN --preset gpt2_124m [--step N]

| flag | meaning |
|---|---|
| `RUN` | run name under `results/pretrain/checkpoints/` |
| `--preset` | the run's preset (required) |
| `--step` | checkpoint step (default: newest complete) |
| `--val-tokens` / `--batch-tokens` | eval volume (default 10.5M) and eval batch |

## train_fleet.py — data-parallel fleet twin

**`train`** — multi-daemon DP run over `topology.toml`
([distributed_training.md](../../docs/distributed_training.md)).

| flag | meaning |
|---|---|
| `--preset --steps --peak-lr --seed --out` | as solo |
| `--topology` / `--group` | topology.toml path / group to train across |
| `--rounds` | per-rank round counts (rank order = member order; weighted data split) |
| `--budgets` / `--slabs` | per-rank device budgets / host slabs (GiB) |
| `--backend {hostmem,nccl,auto}` | group backend override |
| `--opt-shard {zero1,zero1rs}` | optimizer-state sharding mode |
| `--tp-mlp` | tensor-parallel MLPs through the sharding API (correctness track) |
| `--attach HOST=SOCK` | attach to pre-launched daemons (repeatable; the profiling rigs use this) |
| `--checkpoint-every / --checkpoint-redundancy / --checkpoint-keep-last / --resume` | fleet checkpointing (`--resume auto` or a step dir) |
| `--profile` + start/stop-step flags | wrap every launched daemon in the canonical nsys command, bracket per-rank, fetch reports back |

**`compare --a RUN.json --b RUN.json`** — overlay two finished runs.

**`sweep`** — single-GPU vs distributed (3:1) comparison grid:
`--preset --global-tokens 32K,64K,... --steps --budgets a,b
--backing a,b --peak-lr --seed --topology --out-dir` (writes
`sweep.json` + a comparison table).

## dataflowd.py — engine service daemon CLI

`start | status | stop` ([engine_service.md](../../docs/engine_service.md)).
`start` flags: `--socket --slab-gib --device --kernels --fake`
(CPU-only boot), `--plugin` (import a self-registering workload
module), `--no-default-workloads`, and the peer plane's
`--peer-name --peer-listen --peer-rdma-device`. Runs in the
foreground — background it with `daemonize.py`, systemd, or tmux.

## daemonize.py — launch-and-detach

POSIX double-fork detach for long runs:
`python tools/train/daemonize.py --pidfile P --logfile L [--cwd D] -- CMD ARGS...`
(the pidfile's process group is the kill handle).

## fetch_dataset.py — materialize a hub dataset locally

    python tools/train/fetch_dataset.py openai/gsm8k --config main

Writes `datasets/<name>/<split>.jsonl` (gitignored): text columns
pass through; prompt/response shapes normalize to a joined "text"
plus the kept pair. Idempotent (`--force` to redo); `--field`,
`--split`, `--revision`, `--limit`, `--target` as needed. The result
trains via `--data jsonl:datasets/<name>/<split>.jsonl,tokenizer=...`.

## pretrain_report.py — study reports

No flags: rebuilds every pretraining study report
(`results/pretrain/reports/*.html`, self-contained light+dark HTML
with inline-SVG charts) from the run curves under `results/pretrain/`.
