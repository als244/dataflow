# tools/train/ — running training

One entry point for training at every world size, plus the checkpoint
evaluator, dataset fetcher, report generator, and daemon utilities.
All commands run from the repo root.

## train.py — THE training tool

Solo is the world-1 special case of a fleet: no topology flags means
one local CHILD-process daemon (zero-config); naming a topology group
makes it a DP fleet. Checkpoints carry one record at every world size
(`checkpoint_record.json`: responsibility save plan, launch record with the exact
argv and per-rank planned programs). `--profile` wraps every launched
daemon in the canonical nsys command — one flag, any world size.

**`train`** — engine training.

| flag | meaning |
|---|---|
| `--preset` | any `resolve_preset` name (docs/builtin_models.md) |
| `--steps` / `--seed` | optimizer steps / init seed |
| `--peak-lr` / `--muon-lr` | recipe peak (min = peak/10, warmup = steps/10, cosine); muon params' own peak riding the same shape |
| `--opt {adamw,muon}` | override the preset's optimizer policy |
| `--batch` / `--ga-rounds` | round geometry overrides |
| `--fast-budget` | device fast memory GiB, comma-separated per rank |
| `--backing-budget` | host memory GiB per rank — the daemon's pinned backing store |
| `--topology` / `--group` / `--rounds` | fleet mode: topology.toml path, group, per-rank DATA shares in round units — each rank's local grad-accum count is its share (omit all three for zero-config solo) |
| `--backend` | group backend override: hostmem \| nccl \| auto |
| `--opt-shard {zero1,zero1rs,co}` | DEFAULT at world>1 is zero1rs (each rank steps params/n); `co` = the co-responsible DIAGNOSTIC lane (replicated stepping, bitwise cross-rank equality as a corruption tripwire) |
| `--tp-mlp` | tensor-parallel MLPs (correctness track; needs `--execute-padding` when rounds can be under-full) |
| `--execute-padding` | execute under-full tails as one masked segment (debug/fallback) |
| `--profile` + `--profile-start-before-step N` / `--profile-stop-after-step M` | wrap every launched daemon in nsys; bracket the step window |
| `--checkpoint-every / --checkpoint-redundancy / --checkpoint-keep-last / --resume` | checkpoint saves; `--resume auto` or a step dir |
| `--data SPEC` + `--packing-policy {ffd,greedy}` / `--allow-round-split` / `--capture PATH` | data source + packing ([data_feeds.md](../../docs/data_feeds.md)) |
| `--out` | run-curve JSON (also names the checkpoint dir and its run lock) |

**`reference`** — the pure-torch twin leg (daemon-less):
`--preset --steps --peak-lr --opt --ga-rounds --grad-checkpoint
--checkpoint-every --resume` + the data quartet + `--out`.

**`smoke`** — tiny real-vocab reference-vs-engine gate
(`--steps --fast-budget --backing-budget`).

**`parity`** — one preset, reference + engine at N budgets
(`--preset --steps --fast-budget a,b --backing-budget
--grad-checkpoint`).

**`scaling`** — the preset ladder on one backend
(`--presets a,b,c --backend {reference,engine} --steps --fast-budget
--backing-budget --grad-checkpoint`).

**`peek RUN`** — an in-flight run's loss curve from its newest
checkpoint (checkpoint_record.json, or the engine-local layout reference legs
write); prints last/EMA/min, writes
`results/pretrain/<RUN>_partial.json` (`--ema`, default 0.98).

**`compare --a RUN.json --b RUN.json`** — overlay two finished runs.

## eval_checkpoint.py — held-out val loss of a checkpoint

    python tools/train/eval_checkpoint.py RUN --preset gpt2_124m [--step N]

| flag | meaning |
|---|---|
| `RUN` | run name under `results/pretrain/checkpoints/` |
| `--preset` | the run's preset (required) |
| `--step` | checkpoint step (default: newest complete) |
| `--val-tokens` / `--batch-tokens` | eval volume (default 10.5M) and eval batch |

## dataflowd.py — engine service daemon CLI

`start | status | stop` ([engine_service.md](../../docs/engine_service.md)).
`start` flags: `--socket --backing-gib --device --kernels --fake`,
`--plugin`, `--no-default-workloads`, and the peer plane's
`--peer-name --peer-listen --peer-rdma-device`. A second daemon on a
LIVE socket refuses loudly (stale socket files are reclaimed).
Runs in the foreground — background it with `daemonize.py`, systemd,
or tmux.

## daemonize.py — launch-and-detach + the canonical kill

    python tools/train/daemonize.py --pidfile P --logfile L [--cwd D] -- CMD ARGS...
    python tools/train/daemonize.py --pidfile P --kill

`--kill` is THE way to stop a daemonized tree: signal -> wait ->
escalate -> VERIFY; exit 0 only when the whole process group is gone.

## fetch_dataset.py — materialize a hub dataset locally

    python tools/train/fetch_dataset.py openai/gsm8k --config main

Writes `datasets/<name>/<split>.jsonl` (gitignored); text columns
pass through; prompt/response shapes normalize to a joined "text".
Idempotent (`--force`); `--field --split --revision --limit --target`.
Trains via `--data jsonl:datasets/<name>/<split>.jsonl,tokenizer=...`.

## pretrain_report.py — study reports

No flags: rebuilds every pretraining study report
(`results/pretrain/reports/*.html`) from the curves under
`results/pretrain/`.

