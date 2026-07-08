# Exporting measured runs (gap analysis → webapp)

Two tools turn a real training run into inspectable artifacts: a
sim-vs-real gap decomposition on disk, and a single uploadable file
the [webapp simulator](https://dataflowsim.sunshein.net/) renders as a
TRUE event timeline diffed against the simulator's prediction.

(Naming note: exporting *programs* needs no tool — every bench cell
already contains the webapp-uploadable `program.json`, and
`save_program` serializes any Program; see
[program_schema.md](program_schema.md). This doc is about exporting
measured RUNS.)

## `tools/gap_analysis.py` — decompose real-vs-sim for one cell

Runs the exact sweep pipeline for one (config, budget) point —
profile (cached) → plan on measured costs → train a few steps with
tracing — then attributes the throughput gap:

```
real vs sim = scheduling fidelity   (replay gap: re-sim with measured durations)
            + cost-model error      (planned vs measured per-task durations, by task family)
            + transfer-model error  (planned vs achieved PCIe bandwidth per direction)
```

```bash
python tools/gap_analysis.py --config mini --budget 1 --steps 2 \
    --out artifacts/gap-mini
```

(`--config` accepts any `bench_train` CONFIGS name; the tool's
docstring example `8b-bs4ga4 --budget 18` is the canonical full-scale
invocation.)

| flag | meaning |
|---|---|
| `--config` | a `bench_train` CONFIGS name (`{preset}-s{seq}k-bs{B}ga{G}`) |
| `--budget` | fast-memory LEDGER in GiB — note this is the planner budget, NOT `bench_train --device-gib`'s device envelope (no fixed/scratch derivation here; pick a cell's `planned_budget_gib` from its `measured.json` to reproduce that cell's plan; an infeasibly small ledger fails loudly at planning) |
| `--steps` | training steps to trace (default 3) |
| `--backing-gib` | pinned-host cap in GiB (default: unlimited, matching `bench_train` — a finite cap BELOW the config's backing demand makes planning infeasible) |
| `--contend` | profile with PCIe contention (matches the sweep convention) |
| `--out` | output directory |

Outputs in `--out`: `analysis.md` (the human-readable attribution),
`analysis.json` (per-task-family planned/measured totals,
per-direction achieved bandwidth, stall exposure), and the raw pair
the exporter consumes: `trace.json` (measured event intervals) +
`annotated.json` (the executed plan).

## `tools/export_measured_run.py` — package for the webapp

Consumes a gap directory and emits ONE self-contained
`dataflow-measured-run/v1` JSON containing both the measured event log
and the simulator's prediction for the same plan (both in the webapp's
native EventLog shape):

```bash
python tools/export_measured_run.py --gap-dir artifacts/gap-mini \
    --meta config=mini budget_gib=1 --out mini.measured.json
```

| flag | meaning |
|---|---|
| `--gap-dir` | output dir of `gap_analysis.py` (needs `trace.json` + `annotated.json`) |
| `--meta` | `key=value` pairs stamped into the file (run identity: config, budget, device, kernel set) |
| `--out` | the uploadable `.measured.json` |

Upload the file to the webapp: the real run renders in the same
panels used for simulations (task/transfer timelines, memory trace,
utilization summary), side by side with the sim's expectation — this
is the primary instrument for inspecting the sim-vs-real fidelity gap
beyond the single `replay_fidelity_gap_pct` number every bench row
carries.

## The three artifact kinds, disambiguated

| artifact | produced by | upload shows |
|---|---|---|
| `program.json` (per frontier cell) | `bench_frontier` | the SIMULATOR's expected timeline for that plan (measured task costs) |
| `measured.json` (per frontier cell) | `bench_frontier` | nothing to upload — the numeric summary ROW (tok/s, peaks, fidelity) |
| `*.measured.json` (event-log bundle) | `gap_analysis` → `export_measured_run` | the TRUE measured timeline, diffed against the sim's prediction |
