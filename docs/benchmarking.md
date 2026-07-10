# Benchmarking guide

How to measure any model preset at any device-memory budget, and which
tool to reach for. All benchmark tooling lives in `tools/`; every tool
takes an output directory and leaves everything it produces there.

## The tools and when to use each

| tool | use case |
|---|---|
| `bench_train.py` | Run ONE preset config at one or more envelopes. The workhorse every other tool shells out to. Use directly when iterating on a single cell. |
| `best_config.py` | The shape oracle: given (family preset, seq_len, seqs/step, envelopes), profile every (bs, ga) divisor shape and sim-rank them per envelope. Use when you don't know the right batch/accum shape. |
| `bench_frontier.py` | The full matrix: presets x envelopes x placement modes. Orchestrates the oracle + bench_train subprocesses, renders the tables, emits per-cell provenance. Use for any result that will be quoted or committed. |
| `bench_tables.py` | Per-config table renderer and kernel-set A/B comparisons over raw summary directories. Prefer bench_frontier for full matrices. |
| `bench_moe_kernels.py` | Per-op kernel head-to-heads vs the flextrain reference at target shapes. Kernel work only. |
| `bench_qwen35_kernels.py` | fla delta-rule + conv A/B microbench at qwen3.5 shapes (varlen-vs-batched invocation). Kernel work only. |
| `bench_vmm.py` | VMM slab primitive microbench (map/unmap/remap latencies). Placement work only. |
| `gap_analysis.py` + `export_measured_run.py` | Decompose one cell's real-vs-sim gap; package the measured event log for webapp upload — [exporting_runs.md](exporting_runs.md). |
| `engine_gate.py` / `pressure_correctness.py` | Engine regression gates, not benchmarks. |

## The standard recipe

One command, reproducible, self-contained:

```bash
python tools/bench_frontier.py \
    --presets olmoe-7b --seq-len 1024 --seqs-per-step 64 \
    --device-gib 12,16,20,24,28 \
    --shapes oracle --run --rerun \
    --num-steps 3 --out-dir results/bench/<sweep-name>
```

- `--shapes oracle` runs `best_config` first (fresh profiling of every
  shape) and pins its per-envelope winners. Alternatives: `cached`
  (best legal shape already on disk) or an explicit map
  (`12:bs4ga4,16:bs8ga2,...`).
- A sweep with `--out-dir` is ISOLATED to its own `raw/` by default —
  it neither reuses nor renders rows from anywhere else.
  `--reuse-shared` opts into scanning the shared `artifacts/bench`
  pool (resume/compare across sweeps); `--rerun` re-executes cells
  that already exist.
- Placement defaults to `static`; pass `--placements static,vmm` for a
  mode comparison (shapes are held fixed across modes so the pair
  isolates placement).

### bench_frontier flag reference

`--help` is authoritative; this table is kept in sync:

Loss semantics in bench runs: the default data stream feeds random
token sequences with SHIFTED targets (next-token), unique across the
batch and ga rounds within a step, and REPEATED across steps. A
healthy run therefore starts at ~ln(vocab) (+ init logit variance;
~12.6 for llama3-8b) and DECLINES as the model memorizes the fixed
set — a live end-to-end learning signal in every throughput row. On
truly fresh random data the loss must sit flat at ~ln(vocab): no
model can beat the entropy floor of unpredictable targets. Every measured
train() starts FROM-INIT (weights refilled in place before each
budget row and each auto-headroom rerun), so all rows of a sweep
show the SAME loss trajectory — divergence between rows is a
correctness signal, not noise. `headroom_reruns` records plan
churn only.

`--decoupled-targets` switches the stream to independently drawn
targets (still fixed across steps): pure memorization with zero causal
structure — copy-from-context shortcuts cannot contribute, so
comparing its curve against the default shifted stream separates rote
memorization from sequence-structure exploitation.

Freeze flags (`docs/frozen_training.md`): `--freeze-layers N` freezes
layers 0..N-1 through the optimizer policy; add `--freeze-embed` to
turn the prefix from pass-through (wgrads skipped, dgrads still run)
into truncated (no backwards below layer N at all); `--freeze-head`
drops dW_head/O_head while CE still runs. The freeze shows up in the
summary json (`freeze_layers`/`freeze_embed`/`freeze_head`) and in the
log/summary filename tag (`-fz16e`).


| flag | meaning |
|---|---|
| `--presets` | comma list of family presets (bench_train config prefixes), e.g. `dsv32-mini,glm52-mini` |
| `--seq-len` | sequence length (multiple of 1024); config names use the derived tag (4096 -> `s4k`) |
| `--seqs-per-step` | sequences per optimizer step (tokens/step = this x seq-len); used by the oracle |
| `--device-gib` | comma list of device-memory envelopes to sweep |
| `--placements` | comma list of placement modes (default `static`) |
| `--shapes` | `oracle` (fresh best_config sweep) / `cached` (best legal on disk) / explicit `12:bs4ga4,...` map |
| `--num-steps` | training steps per cell (default 3; step 1 is warm-up, the rest steady-state) |
| `--run` | execute missing cells (paced bench_train subprocesses) |
| `--rerun` | re-execute cells even if rows already exist |
| `--reuse-shared` | also scan the shared `artifacts/bench` pool (default: isolated to `{out-dir}/raw`) |
| `--dry-run` | print the bench_train commands without running |
| `--pace-seconds` | settle between cell invocations (default 5; each bench_train subprocess is waited on — this is extra soak, not the wait) |
| `--backing-gib` | forwarded to bench_train: explicit pinned-host plan-cap (default: auto-derived from host MemAvailable) |
| `--backing-leeway-gib` | forwarded to bench_train: leeway subtracted from MemAvailable for the auto cap (default 10) |
| `--allow-illegal` | render envelope-busting rows, flagged, instead of dropping them |
| `--out-dir` | sweep directory: `TABLES.md` + `cells/` + `raw/` (omit for stdout tables) |
| `--plugin` | external family plugin module(s); installed entry points load automatically |

Backing (pinned-host) capacity: unless `--backing-gib` is given, bench_train derives a PLAN-cap from the host's available memory minus `--backing-leeway-gib` (default 10 GiB), plans under it, and executes with the capacity stripped — the runtime pins by plan demand, never by the cap. Saved plans record their capacities in `Program.metadata` ([program_schema.md](program_schema.md)).

Output layout under `--out-dir`:

```
TABLES.md                      # mode-pure tables + best-legal summary
oracle-<preset>-<seq>-x<n>.json
cells/<preset>-<dev>gib-<mode>/
    measured.json              # the full row: wall/sim tok/s, peak +
                               # fixed/extent/scratch decomposition,
                               # fidelity, rc, shape, step walls, losses
    plan.json                  # annotated program — replay with
                               # bench_train --annotated
    program.json               # upload to the webapp simulator
raw/                           # every bench_train output: summaries,
                               # plans, webapp programs, logs/<invocation>.log
```

## bench_train.py — one (config, budget) run in depth

The single-cell engine under the frontier: profile (cached) → recompute
planning on measured costs → PressureFit → execute N steps on the real
GPU → measured summary row + replayable exports. Invoke directly when
iterating on one cell; the frontier composes these invocations.

```bash
python tools/bench_train.py --config glm52-mini-s4k-bs8ga2 \
    --device-gib 16,20 --steps 4 --out artifacts/bench
```

| flag | meaning |
|---|---|
| `--config` | a CONFIGS name (`{preset}-s{seq}k-bs{B}ga{G}`; full list: `--help` or [builtin_models.md](builtin_models.md)) |
| `--device-gib` | HARD device envelope(s), comma list: placed extent + measured op scratch + CUDA context must all fit; the ledger is derived per envelope and the run is post-hoc verified (busting rows are flagged) |
| `--steps` | training steps (default 4; step 1 is warm-up, steady-state averages the rest) |
| `--recompute` / `--no-recompute` | sim-verified greedy recompute planning before PressureFit (default on) |
| `--force-recompute all` | skip the planner: every rewrite at level 1 (ablation tool) |
| `--placement` | `static` (packed at planning time) / `vmm` (chunk-backed) / `dynamic` |
| `--optimizer` | `interleaved` (optimizer tasks inside the chain, default) / `tail` |
| `--preplace` | initial fast placement: `task0` (default — pre-place only task 0's needs, rest travel as planned prefetches) / `greedy` (legacy) |
| `--backing-gib` | explicit pinned-host cap. UNSET (default): a PLAN-cap is derived from host MemAvailable − `--backing-leeway-gib`, planning is bounded by it, and the executed program is stripped back to capacity None so the runtime pins by plan demand (a set capacity makes the engine pin the full amount as one slab) |
| `--backing-leeway-gib` | leeway for the auto cap (default 10) |
| `--probe-max` | probe the largest feasible ledger instead of running |
| `--extent-budget` | budget the placed extent rather than the device envelope |
| `--baseline` | also time the plain-torch golden model (reference ceiling) |
| `--annotated` | REPLAY a saved plan (`*.annotated.json` / a cell's `plan.json`) instead of planning fresh |
| `--refresh-profiles` | ignore the profile cache and re-measure |
| `--out` | output dir (default `artifacts/bench`): summary JSON + per-run exports + `logs/` |
| `--plugin` | external family plugin module(s) |

Outputs per (config, budget): a row in
`{config}-{devs}dev-{placement}.summary.json` (throughput, peaks,
fidelity, recompute chosen, `envelope_ok`), plus `{stem}.annotated.json`
(the executed plan — replayable via `--annotated`, capacities stamped in
`Program.metadata`) and `{stem}.webapp.json` (upload at the
webapp simulator (webapp/) for the simulator's expected
timeline), and the full console log under `logs/`.

## What a row means (the legality contract)

Every quoted number is ENVELOPE-LEGAL: the measured device peak
(fixed + placement extent + torch reserved high-water) is <= the
quoted budget. Enforcement is bench_train's auto-headroom closing
loop — after the run, if the measured peak busts the envelope, the
ledger shrinks by the measured overage (+0.25 GiB margin, which beats
the ~±0.2 GiB run-to-run torch-scratch variance) and the row re-runs,
up to twice. There are no hand-tuned leeway constants anywhere; the
derivation's reserve values are only initial guesses that position the
first attempt. Rows that still bust carry `envelope_ok=false` and are
REFUSED by the sweep renderer unless `--allow-illegal` (rendered
with a warning flag).

Cell format in tables:

```
wall tok/s (sim tok/s) · peak GiB · bsXgaY · rc N%
```

- `sim` is the simulator's prediction for the exact plan that ran.
  Static-vs-vmm sim differs because the DERIVED LEDGERS differ (extent
  shave vs arena headroom) and the planner picks different recompute
  levels — the simulator itself has no placement term.
- `rc%` = recomputed layer-rounds / total layer-rounds.

## Placement modes

- `static` (default): contiguous slab, offline interval packing. The
  packed extent can exceed the peak concurrent load (geometry tax) —
  usually <2%, but long-lived cross-layer objects (e.g. IndexShare
  metadata) can push it to ~25% at tight envelopes.
- `vmm`: non-contiguous arena (CUDA VMM chunk mapping) — no packing
  geometry, at the cost of arena overheads and looser replay fidelity
  (2-5% vs static's 0.3-1%). Wins where geometry bites (tight
  envelopes, long-lived metadata); loses when memory is loose.

## Profile cache

Task costs are profiled once per unique signature and cached in
`artifacts/profile-cache/` (keyed by signature + environment +
`PROFILE_CACHE_REV`). The cache always reflects CURRENT kernels: if
you change any kernel or task launch path, bump `PROFILE_CACHE_REV`
in `src/dataflow/training/profiling.py` — stale costs silently skew
both sim predictions and the planner's recompute choices. Wiping
`artifacts/profile-cache/` forces a full re-profile (first run per
shape gets slower; nothing else changes).

## Operational notes

- Back-to-back invocations pin ~50-90 GB of host weights each;
  systemd-oomd can pressure-kill dense invocation chains. The sweep
  paces subprocess launches (`--pace-seconds`, default 40).
- `pytest | tail` reads tail's exit status — use `set -o pipefail` or
  trust the unpiped exit code.
- The shared `artifacts/bench` pool is scanned by default for resume
  convenience; any sweep whose numbers will be quoted should use
  `--no-legacy` to scope strictly to its own `raw/`.
