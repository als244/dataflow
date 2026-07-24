# Throughput and sim-fidelity sweep

Two questions, on whatever GPU this is run on:

1. **Where does throughput live?** Tokens/s and effective TFLOP/s across GPU
   memory budget and training geometry — the landscape a low-memory training
   runtime exists to move through.
2. **Does the simulator tell the truth?** Predicted vs measured seconds/step
   for the same plan, cell by cell.

## Quickstart

```bash
python reproducibility/throughput_fidelity/run_experiment.py
```

That is the whole thing. It picks the model, the budgets and the geometry from
the machine it finds itself on, runs the sweep, and prints the tables. Expect
one to three hours depending on the card. Results land in `data/`, figures in
`figs/`, and a summary is printed at the end.

A quick look first, on a smaller model with a coarse grid:

```bash
python reproducibility/throughput_fidelity/run_experiment.py \
    --preset l3_1b --seqs 1024,4096 --budget-step 2 --target-cells 6 --steps 4
```

Then draw the figures (any time, including mid-run):

```bash
python reproducibility/throughput_fidelity/make_plots.py adamw
python reproducibility/throughput_fidelity/analyze.py
```

## Configuration

`--help` is the reference: every knob is a flag, and every default is derived
from the machine rather than written down. Set none and the run is still
meaningful; set one and only that dimension moves.

| flag | default | what it changes |
|---|---|---|
| `--preset` | largest model whose persistent state fits the host | the model, e.g. `l3_1b` |
| `--opts` | `adamw,muon` | optimizers swept; one of them roughly halves the run |
| `--seqs` | `1024,2048,4096,8192` (capped by the preset) | sequence lengths |
| `--t-rounds` | `8192,16384,32768,65536` | tokens per round — grad-accumulation granularity |
| `--t-steps` | scaled to the device class | tokens per optimizer step |
| `--budgets` | ladder from a one-task floor to 0.85 × device | GPU memory budgets, explicitly |
| `--budget-step` | `1.414` | granularity of that ladder instead of replacing it: `2` for octaves, `1.2` for fine |
| `--host-share` | `0.8` | fraction of host memory offered as the allowance |
| `--backing-gib` | — | the allowance outright, ignoring `--host-share` |
| `--target-cells` | `18` | how many cells get real GPU runs |
| `--steps` | `6` | steps per measured cell; the first 3 are warmup, so keep ≥ 4 |
| `--stages` | all | resume or repeat part of a run, e.g. `--stages predict,select,measure` |
| `--python` | this interpreter | interpreter for the stage processes |

The grid is the cross product of `SEQS × T_ROUNDS × T_STEPS × BUDGETS`, so
prediction cost grows with all four — but only `TARGET_CELLS` and `STEPS`
drive GPU time in the measurement pass, which is the expensive one.

Geometry has one contract: `T_ROUNDS` must divide by the sequence length and
into `T_STEPS`. Combinations that do not are recorded as skips rather than
silently dropped.

## Order of operations

Each stage consumes the previous one's output; nothing is hard-coded to a
machine. Run it on a datacentre card or a desktop one and the same command
means the same thing.

### P0 · environment probe — *what can this box hold?*
`env_probe.py` → `env.json`  ·  seconds, needs the device

Reads the limits that actually apply: device memory from the driver, and the
host limit from the scheduler's grant if there is one (`SLURM_MEM_PER_NODE`),
else a cgroup cap, else physical RAM. A compute node reports its full physical
RAM even when the job owns a slice of it, so sizing a pinned slab from the node
total gets the job killed. From those it derives:

- **preset** — the largest model whose PERSISTENT state (parameters, optimizer,
  gradients) fits the host. Saved activations are deliberately not part of that
  test: they are elastic, because the host allowance is a planner input and a
  tighter one simply makes the recompute planner keep fewer contexts. A smaller
  box therefore runs the same model, deeper into the recompute regime — which
  is the regime this runtime exists for.
- **budget ladder** — √2 steps from a floor that can hold the largest single
  task, up to 0.85 × device memory. Doubling is too coarse: the whole
  transition from offload-bound to compute-bound can hide between 16 and 64 GiB
  and three points cannot show a knee.
- **host allowance** — 0.8 of what the host can offer. Fixed, not swept (see
  *The host allowance* below).
- **geometry axes** — sequence lengths, tokens per round, tokens per step,
  scaled to the device class.
- **link rate** — read from the engine's own cached measurement, taken with
  both directions in flight. That is the number plans price transfers at, and
  it is lower than either direction benchmarked alone, so re-benchmarking it
  here would report a prettier figure that nothing uses.

### P1 · predictions — *the whole grid, and where it becomes infeasible*
`run_experiment.py` stage `predict` → `data/predict_measured_{opt}.jsonl`
·  the long pole for prediction; needs the device

For each optimizer, for each sequence length, over every (tokens/round ×
tokens/step × budget) combination:

1. **profile** every unique task signature on this GPU (disk-cached per
   geometry + kernel set + device + sequence length),
2. **plan** the step at that budget and host allowance — recompute planning
   first, then placement,
3. **record** simulated s/step, tok/s, effective and hardware TFLOP/s, peak
   fast and backing bytes, transfer bytes and duty each way, recompute share of
   makespan, idle share, and the recompute levels chosen.

Cells the planner cannot fit are written as **INFEASIBLE rows carrying the
planner's reason**, never skipped, so the feasibility boundary is data rather
than a gap. Each cell also re-plans once with 25% more host memory to price the
allowance (`binding`, `host_marginal_gain`).

### P1b · cell selection — *which cells deserve real GPU time?*
`select_cells.py` → `cells.json`  ·  seconds, CPU only

Measuring every survivor would spend hours re-measuring identical behaviour.
In order:

1. keep cells feasible under **every** optimizer being swept, so the same cells
   are comparable across them;
2. **reduce over tokens-per-round** — it is a knob the operator picks, not a
   property of the hardware. For each (sequence, tokens/step, budget,
   allowance) only the best-throughput round size is on the frontier; nobody
   would run a 64K-token round on a 16 GiB budget;
3. **cluster** the frontier by what its plans DO — recompute share, transfer
   duty each way, idle share, throughput — and keep the cell nearest each
   centre, so every distinct regime is measured once;
4. force-keep a **budget spine**: one geometry across all its feasible budgets,
   because the headline result is throughput vs budget and a curve needs its
   points to share a geometry;
5. add two **dominated controls** — cells step 2 discarded. If the engine ranks
   them differently from the simulator, the reduction is unsafe and the data
   will say so rather than the assumption going unchecked.

### P2 · measurement — *what the engine actually does*
`run_experiment.py` stage `measure` → `data/measure_{opt}.jsonl`  ·  the long pole
overall; needs the device

Runs each selected cell on the real engine, per optimizer, for `STEPS` steps
with the first 3 discarded as warmup, and reports the mean of the tail beside
the simulator's prediction for **that same plan**. The cell is planned with the
same profiled costs and the same host allowance the run will actually have, so
predicted and measured describe one plan rather than two. Per cell: `pred_s`,
`meas_s`, their `ratio` (the fidelity number), tok/s, effective and hardware
TFLOP/s, recompute levels, peak backing. A cell that fails to plan or run is
recorded as a row with its error, not a crash.

### P3 · shipped-command validation — *do the documented commands still work?*
→ `logs/shipped_bench.log`  ·  minutes, needs the device

Runs the repository's own `tools/bench/predict_step.py --measured` and
`tools/bench/measure_step.py --measured-plan` at the spine geometry. The rest of
this directory drives the library directly; this stage checks the commands a
reader would actually type still work at this scale.

### Analysis
`analyze.py` runs automatically at the end: feasibility counts, host-pressure
summary, and the per-cell fidelity table. `make_plots.py <opt>` draws the
figures.

## Two ways to price a task, and why this uses one

The simulator's structure is the same either way: it schedules tasks and
transfers over a chain and reports a makespan. Only the per-task cost SEED
differs, and the difference is large.

**Roofline** (`predict_step.py` with no `--measured`) estimates each task
analytically — FLOPs against peak compute times an efficiency factor, bytes
against memory bandwidth, whichever binds. It needs no GPU and answers
instantly, which makes it the right tool for asking what a machine you do not
have would do. But its efficiency constants (`matmul_eff`, `attn_eff`,
`mem_eff`) are calibrated per device class, so on an unfamiliar card it gives
you a shape, not a number.

**Profiled** (`--measured`, and `--measured-plan` on the measure side) executes
every unique task signature on the actual GPU, times it with CUDA events after
a thermal soak, and caches the result to disk keyed by geometry, kernel set,
device and sequence length. It costs minutes on first sight of a geometry and
requires the device, and it is what makes a prediction a number rather than a
shape.

**This study is profiled everywhere** — predictions, plans, and the prediction
column of every measurement. The roofline appears nowhere, because comparing a
measured run against an analytic estimate would confound two different
questions: whether the machine model is right, and whether the scheduler is.

Profiles are keyed by geometry **and metadata including sequence length**. A
round of T tokens occupies the same buffers whether it is one long sequence or
many short ones, so without that key `batch × seq_len` combinations with equal
token counts shared one timing — while attention cost scales with the sequence,
not the token total. A new sequence length now misses the cache and re-profiles.

### Why this matters: profiling on zero-valued inputs (found and fixed here)

The first pass showed measured/predicted ≈ **1.21–1.38×** wherever compute was
on the critical path, while transfer-bound cells looked exact (0.96).
Candidates were eliminated with measurements, not argument:

| candidate | verdict | evidence |
|---|---|---|
| hidden sync / stall | ✗ | 11 device syncs in a whole run; kernels 91.6% of span |
| host launch gaps | ✗ | gaps sit *between* tasks; trace intervals match nsys kernel time to 1.3% |
| DMA / HBM contention | ✗ | kernel duration flat vs overlapping copy bytes (11.62 ms @ 0 GB, 11.86 ms @ 0.67 GB) |
| thermal / power drift | ✗ | flat across 32 s (11.64 / 11.56 / 11.64 / 11.61 ms) |
| sustained-load self-throttle | ✗ | 1000 back-to-back launches dead flat at 46.4 ms |
| **zero-valued profiling inputs** | **✓** | below |

`profile_program` filled only int32 buffers and left float payloads
uninitialized. Operand values drive switching activity → power → sustained clock:

| input fill | power | SM clock | block_bwd | flash_bwd kernel |
|---|---|---|---|---|
| uninitialized (old) | 484 W | 1811 MHz | 46.38 ms | 9.114 ms |
| explicit zeros | 474 W | 1827 MHz | 46.57 ms | — |
| **N(0,1)** | **681 W** | **1448 MHz** | **57.03 ms** | **11.624 ms** |
| *same kernel in a real step* | *690 W* | *~1560 MHz* | *59.9 ms* | *11.612 ms* |

Realistic data reproduces the real pipeline's kernel time to **0.1%**, and the
clock ratio (1.25×) explains the runtime ratio (1.23×) within 2%. Composite
operands — saved-activation contexts and packed weight layouts — carry no
element type and are the largest buffers a block reads, so seeding only the
typed ones left most bytes zero and the correction inert.

End to end on one cell, predicted vs measured makespan went from **1.20×**
(10.28 s predicted, 12.36 s real) to **0.99×** (12.06 s, 11.97 s), with idle
modelling rather than compute cost as the residual. The error was always
present but is masked wherever transfers dominate the makespan.

## The host allowance is a constraint, not a measurement

How much host memory a plan "wants" is only defined when host memory is free —
impose a ceiling and the planner picks a different plan, so the number that
justified the ceiling no longer describes what happens under it. Using demand
to set the ceiling is circular. So the allowance comes from the machine (0.8 of
what this host can offer, no plan input) and its effect is measured at that
operating point:

- **`binding`** — did the plan hit the ceiling, or did it want less?
- **`host_marginal_gain`** — the same cell re-planned with 25% more room: the
  local slope of throughput against host memory. A shadow price rather than an
  assumed level. Near zero means the allowance is irrelevant for that cell; a
  large value means the box is genuinely host-starved, and says by how much.

The counterfactual is deliberately not capped at the machine's own RAM, since
the question it answers is whether a bigger machine would help. This also
degrades gracefully: on a workload that never approaches the ceiling it reports
≈0 everywhere, which is the honest answer rather than a silently inert axis.

## Outputs

| path | contents |
|---|---|
| `env.json` | what this box chose and why: device, host limit and its source, preset, persistent state, allowance, budget ladder, geometry axes |
| `cells.json` | cells chosen for real runs, tagged `budget_spine` / `frontier` / `dominated_control`, with predicted s/step |
| `data/predict_measured_{opt}.jsonl` | one record per grid cell — s/step, tok/s, eff + hw TFLOP/s, peak fast and backing, transfer bytes and duty, recompute and idle share, `binding`, `host_marginal_gain` — or an INFEASIBLE reason |
| `data/measure_{opt}.jsonl` | per measured cell — `pred_s`, `meas_s`, `ratio`, tok/s, TFLOP/s, recompute levels, peak backing |
| `figs/` | `frontier_{opt}.png` (throughput vs GPU memory, round size optimised and labelled), `throughput_`, `eff_tflops_`, `recompute_pct_`, `idle_pct_` |
| `logs/`, `traces/` | raw stdout per stage; webapp real-vs-sim bundles and nsys captures |

`env.json` also records the link rate plans were priced at, so a run can be
read back without the machine present.

Every record carries its full identity (host, device, preset, optimizer,
geometry, budget, allowance, timestamp), so files from different boxes can be
concatenated and compared without a side channel.

## Analysis and investigation tools

| tool | what it answers |
|---|---|
| `analyze.py` | feasibility counts, host pressure, per-cell fidelity table |
| `make_plots.py <opt>` | the figures above |
| `task_cost_audit.py` | per task group, the cost the sim used vs the real duration |
| `deep_dive.py` | one cell end to end: makespan decomposition and the webapp real-vs-sim bundle |
| `throttle_probe.py` | one task, sustained or back-to-back, `--fill uninit\|zeros\|randn`, `--sets` to defeat buffer reuse |
| `nsys_gap_analysis.py` | kernel busy vs inter-kernel gaps — is time lost inside kernels or between them |
| `nsys_kernel_vs_dma.py` | kernel duration against overlapping DMA, and against wall time |
| `pcie_calib.py` | one-direction-at-a-time link bandwidth — useful only against the engine's concurrent number, which is what plans use |
| `matmul_burst.py` | a full-power reference load, for comparing power and clock behaviour |
