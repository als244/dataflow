# Throughput and sim-fidelity sweep

Two questions, on whatever GPU this is run on:

1. **Where does throughput live?** Tokens/s and effective TFLOP/s across GPU
   memory budget and training geometry — the landscape a low-memory training
   runtime exists to move through.
2. **Does the simulator tell the truth?** Predicted vs measured seconds/step
   for the same plan, cell by cell.

```bash
bash reproducibility/throughput_fidelity/run_experiment.sh
```

That is the whole interface. Overrides: `PYTHON=` (interpreter), `PRESET=`
(skip model selection), `OPTS=adamw,muon` (optimizers), `TARGET_CELLS=18` (how
many cells get real runs), `STEPS=6` (steps per measured cell, first 3 are
warmup).

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

### P1 · link calibration — *what is host↔device actually worth here?*
`pcie_calib.py` → `logs/pcie_calib.log`  ·  seconds, needs the device

Sustained pinned H2D and D2H bandwidth. Recorded for the write-up rather than
fed to the planner — the engine keeps its own cached bidirectional measurement,
taken with both directions in flight, which is the number plans should use.
This is the sanity check that the link is what the machine claims.

### P2 · predictions — *the whole grid, and where it becomes infeasible*
`sweep.py --mode predict-measured` → `data/predict_measured_{opt}.jsonl`
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

### P2b · cell selection — *which cells deserve real GPU time?*
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

### P3 · measurement — *what the engine actually does*
`sweep.py --mode measure` → `data/measure_{opt}.jsonl`  ·  the long pole
overall; needs the device

Runs each selected cell on the real engine, per optimizer, for `STEPS` steps
with the first 3 discarded as warmup, and reports the mean of the tail beside
the simulator's prediction for **that same plan**. The cell is planned with the
same profiled costs and the same host allowance the run will actually have, so
predicted and measured describe one plan rather than two. Per cell: `pred_s`,
`meas_s`, their `ratio` (the fidelity number), tok/s, effective and hardware
TFLOP/s, recompute levels, peak backing. A cell that fails to plan or run is
recorded as a row with its error, not a crash.

### P4 · shipped-command validation — *do the documented commands still work?*
→ `logs/shipped_bench.log`  ·  minutes, needs the device

Runs the repository's own `tools/bench/predict_step.py --measured` and
`tools/bench/measure_step.py --measured-plan` at the spine geometry. The rest of
this directory drives the library directly; this stage checks the commands a
reader would actually type still work at this scale.

### Analysis
`analyze.py` runs automatically at the end: feasibility counts, host-pressure
summary, and the per-cell fidelity table. `make_plots.py <opt>` draws the
figures.

## Costs are measured, never roofline

Every plan, prediction and estimate is seeded from task costs **profiled on the
GPU running the study**. The analytic roofline is not used anywhere here: it
exists for CPU-only what-ifs and its efficiency constants are calibrated to a
different device class.

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
| `pcie_calib.py`, `matmul_burst.py` | link bandwidth; a full-power reference load |
