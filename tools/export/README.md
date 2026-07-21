# tools/export/ — run analysis & webapp export

Everything that turns programs and runs into inspectable artifacts
([exporting_runs.md](../../docs/exporting_runs.md)).

## export_program.py — CPU-only golden path

Any preset → shaped program → PressureFit (+ recompute) → sim →
exports: `<stem>.{program,annotated,webapp,summary}.json` (the
`.webapp.json` uploads to the
[webapp simulator](https://dataflowsim.sunshein.net/)).

| flag | meaning |
|---|---|
| `--preset` | any `resolve_preset` name; `--plugin` for external families |
| `--fast-gib` | device budget (required) |
| `--recompute` | let the planner choose recompute levels |
| `--seq-len --batch --grad-accum-rounds` | geometry overrides |
| `--out` | output directory (default `examples/`) |

## trace_real_run.py — real steps → measured bundle

Drives a few REAL steps of a preset on a daemon, keeps the LAST
step's trace (steady state), and writes the full webapp bundle —
`<stem>.{program,annotated,webapp,measured}.json` — where
`.measured.json` holds the measured EventLog beside the sim's
prediction of the same plan, plus a one-line real-vs-sim parity
summary.

| flag | meaning |
|---|---|
| `--preset --steps --budget --backing-gib` | the run |
| `--out` | output directory |
| `--name` | output stem (default `trace-<preset>`) |

## trace_program.py — fake-backend event timeline

Plan debugging without a GPU: event-level trace of ANY program on
the fake backend — reserves, transfer charges/enqueues, releases,
offloads, pressure evictions, placement escapes.

| flag | meaning |
|---|---|
| `--program` | a Program JSON (annotated plan / export_program output) |
| `--kinds` | comma list to filter event kinds |
| `--intervals` | also print task/transfer intervals |
| `--out` | write events as JSONL instead of a table |
