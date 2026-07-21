# Exporting measured runs (real timeline → webapp)

One tool turns a real training run into inspectable artifacts:
`tools/export/trace_real_run.py` drives a few REAL steps through the engine
daemon and writes the full webapp bundle, measured vs simulated side
by side — a single uploadable file the
[webapp simulator](https://dataflowsim.sunshein.net/) renders as a
TRUE event timeline diffed against the simulator's prediction.

(Naming note: exporting *programs* needs no run —
`tools/export/export_program.py` lowers any preset CPU-only and writes its
program/annotated/summary JSONs plus the webapp upload
(`<stem>.webapp.json`), priced from roofline costs; `save_program`
serializes any Program — see [program_schema.md](program_schema.md).
This doc is about exporting measured RUNS.)

## `tools/export/trace_real_run.py` — real steps → measured bundle

Every engine run already records a RunTrace; the daemon's run verb
returns it on request. The tool runs `--steps` steps of a preset on a
real daemon, keeps the LAST step's trace (steady state — warm
kernels, settled pools), and writes under `--out`:

```
<stem>.program.json     bare program (core IR)
<stem>.annotated.json   planner-annotated program
<stem>.webapp.json      DataflowProgram v1 (webapp upload)
<stem>.measured.json    dataflow-measured-run/v1: measured EventLog +
                        memory trace + summary, and the sim's
                        EventLog/summary for the same plan
```

plus a one-line real-vs-sim parity summary (task coverage +
makespan).

```bash
python tools/export/trace_real_run.py --preset smoke --steps 3 --out examples/
```

| flag | meaning |
|---|---|
| `--preset` | model preset (any `resolve_preset` name) |
| `--steps` | training steps to run; the last one is kept (default 3) |
| `--budget` | fast-memory plan budget in GiB |
| `--backing-gib` | daemon pinned backing size in GiB |
| `--out` | output directory |
| `--name` | output stem (default: `trace-<preset>`) |

Upload `<stem>.measured.json` to the webapp: the real run renders in
the same panels used for simulations (task/transfer timelines, memory
trace, utilization summary), side by side with the sim's expectation —
this is the primary instrument for inspecting the sim-vs-real
fidelity gap. For a capture of what the GPU actually executed
(kernels, streams, PCIe) rather than the engine's event log, escalate
to `train.py --profile` ([benchmarking.md](benchmarking.md)).

## The three artifact kinds, disambiguated

| artifact | produced by | upload shows |
|---|---|---|
| `<stem>.webapp.json` | `export_program` | the SIMULATOR's expected timeline for a plan (roofline costs) |
| run curve JSONs | `train.py --out` | nothing to upload — numeric summaries (loss, tok/s, TF/s) |
| `*.measured.json` | `trace_real_run` | the TRUE measured timeline, diffed against the sim's prediction |
