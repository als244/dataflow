# Throughput and sim-fidelity sweep

Runs the benchmark workflow (`predict → measure → profile → export`) over the
memory-budget and geometry axes on whatever GPU it finds, and reports two
things: where throughput lives, and how closely the simulator's prediction
matches the real engine.

```bash
bash reproducibility/throughput_fidelity/run_experiment.sh
```

That is the whole interface. Overrides: `PYTHON=`, `PRESET=`, `OPTS=adamw,muon`,
`TARGET_CELLS=`, `STEPS=`.

## Nothing about the machine is hard-coded

Three stages decide the work, so the same command is meaningful on a datacentre
card and on a desktop one:

1. **`env_probe.py` — what can this box hold?** Reads device memory and the host
   limit that actually applies (a scheduler's cgroup cap, else physical RAM),
   then picks the largest preset whose PERSISTENT state — parameters, optimizer,
   gradients — fits the host, a fast-memory budget ladder from a floor that can
   hold one task up to most of the device, and geometry axes scaled to the
   device class. Saved activations are deliberately *not* part of the fit test:
   the backing ceiling is a planner input, so a host with less room simply makes
   the recompute planner keep fewer contexts and re-derive more. A smaller box
   therefore runs the same model, deeper into the recompute regime — which is
   the regime this runtime exists for.
2. **The prediction pass — which cells are feasible?** Every candidate is
   planned on costs profiled on this GPU. Cells the planner cannot fit are
   recorded as INFEASIBLE rows rather than skipped, so the boundary is data.
3. **`select_cells.py` — which are worth real runs?** Measuring every survivor
   wastes hours re-measuring the same behaviour. Cells are described by what
   their plan does (recompute share, transfer duty each way, idle share,
   throughput), clustered, and one representative per regime is kept, plus one
   budget spine — a single geometry across all its feasible budgets, because the
   headline result is throughput vs budget and a curve needs its points to share
   a geometry.

## Costs are measured, never roofline

Every plan, prediction and estimate is seeded from task costs **profiled on the
GPU running the study** (`--measured` / `--measured-plan`, disk-cached per
geometry + kernel set + device). The analytic roofline is not used anywhere
here.

### Why that matters: profiling on zero-valued inputs (found and fixed here)

The first pass of this study showed measured/predicted ≈ **1.21–1.38×** wherever
compute was on the critical path, while transfer-bound cells looked exact
(0.96). Candidates were eliminated with measurements, not argument:

| candidate | verdict | evidence |
|---|---|---|
| hidden sync / stall | ✗ | 11 device syncs in a whole run; kernels 91.6% of span |
| host launch gaps | ✗ | gaps sit *between* tasks; trace intervals match nsys kernel time to 1.3% |
| DMA / HBM contention | ✗ | kernel duration flat vs overlapping copy bytes (11.62 ms @ 0 GB, 11.86 ms @ 0.67 GB) |
| thermal / power drift | ✗ | flat across 32 s (11.64 / 11.56 / 11.64 / 11.61 ms) |
| sustained-load self-throttle | ✗ | 1000 back-to-back launches dead flat at 46.4 ms |
| **zero-valued profiling inputs** | **✓** | below |

`profile_program` filled only int32 buffers and left float payloads
uninitialized. Operand values drive switching activity → power → sustained
clock:

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

## Outputs

| path | contents |
|---|---|
| `env.json` | what this box chose, and why |
| `cells.json` | the cells selected for real runs, with their regime tags |
| `data/predict_measured_{opt}.jsonl` | one record per grid cell: s/step, tok/s, eff/hw TFLOP/s, peak fast + backing, transfer bytes and duty, recompute and idle share — or an INFEASIBLE reason |
| `data/measure_{opt}.jsonl` | per measured cell: predicted vs measured s/step, their ratio, tok/s, TFLOP/s |
| `logs/`, `traces/`, `figs/` | raw stdout, webapp + nsys bundles, figures |

Analysis: `analyze.py` (feasibility + fidelity tables), `make_plots.py <opt>`
(throughput vs budget, faceted by sequence length × tokens/step, with
recompute-time and idle companions), `task_cost_audit.py` (per task group, sim
cost vs real duration), `deep_dive.py` (one cell end to end, emits the webapp
real-vs-sim bundle).

Investigation tools kept for re-use: `throttle_probe.py` (one task, sustained or
back-to-back, `--fill uninit|zeros|randn`, `--sets` to defeat buffer reuse),
`nsys_gap_analysis.py` (kernel busy vs inter-kernel gaps),
`nsys_kernel_vs_dma.py` (duration against overlapping DMA and against wall
time), `pcie_calib.py`, `matmul_burst.py`.
