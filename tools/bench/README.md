# tools/bench/ — throughput: predict → measure → profile

The escalating-cost workflow
([benchmarking.md](../../docs/benchmarking.md)); geometry speaks
T_round (`ga` derives from `--tokens-step`; "batch" is internal
arithmetic under varlen packing).

## predict_step.py — simulated sweeps (CPU, instant)

FIRST LINE OF ATTACK: lowers the true program, plans each cell,
reads the simulator-verified schedule back as a table — s/step,
tok/s, effective/hardware TFLOPs/s, fast/backing peaks, PCIe bytes +
link %, recompute + idle %, ETA. Full guide:
[throughput.md](../../docs/throughput.md).

| flag | meaning |
|---|---|
| `--preset` | any `resolve_preset` name; `--plugin` loads external families |
| `--t-round` / `--t-round a,b,c` | round token budget (single point / sweep axis) |
| `--tokens-step` | tokens per optimizer step (ga = tokens-step / t_round) |
| `--ga-rounds` | alternative to `--tokens-step` |
| `--seq-len` / `--seq-len` | sequence length (third sweep axis) |
| `--budget` / `--budget` | device budget GiB (single / sweep) |
| `--backing` | host-backing capacity ceiling GiB (plans escalate recompute to fit; infeasible combos report as INFEASIBLE rows) |
| `--opt {adamw,muon}` | optimizer (sizes O and the NS work; roofline under-prices muon NS time — `--measured` is muon-exact) |
| `--hw {3090,5090}` + `--tflops --bw --pcie` | hardware profile / overrides |
| `--measured` | profiled task costs instead of roofline (disk-cached; needs the GPU once per geometry) |
| `--steps` | print the ETA column for this many steps |
| `--no-recompute` | pin the plan to zero recompute |
| `--top N` | single-point mode: N most expensive tasks |

## measure_step.py — real sweeps (GPU, minutes)

The measured twin: same grid interface, each cell RUNS the engine
through one shared daemon (programs unregistered + store wiped
between cells) and reports the warmed measurement beside the plan's
prediction — `pred_s meas_s ratio tok/s effTF/s hwTF/s recomp`.

| flag | meaning |
|---|---|
| grid flags | as predict_step: `--preset --plugin --opt --t-round(s) --tokens-step --seq-len(s) --budget(s)` |
| `--steps` | steps per cell (first 3 = warmup, excluded from the mean) |
| `--data SPEC` | data source spec ([data_feeds.md](../../docs/data_feeds.md)); default: the standard feed — pass the uniform-window config for plan-comparable runs |
| `--backing-gib` | daemon pinned backing GiB |
| `--peak-lr` | recipe peak for the cells |
| `--measured-plan` | prediction column from PROFILED task costs |
| `--hw` | display only — the run measures the real box |

## Nsight captures

Profiling is a `train.py` flag, not a separate tool: `--profile
--profile-start-before-step N --profile-stop-after-step M` wraps
EVERY launched daemon in the canonical nsys command (cudaProfilerApi
capture range; brackets ride the daemon's `profiler_control`) — one
flag, any world size; per-rank reports are fetched back for fleets.
See [tools/train/README.md](../train/README.md).

## compare_pressurefit_quality.py — planner quality gate (CPU)

Runs two PressureFit source versions in isolated subprocesses on identical
serialized chains, then replays both plans through one common simulator oracle.
It reports feasibility changes, selected makespan, peaks, and transfer counts
across canary, synthetic, and model/hardware/budget suites.

```bash
conda run -n dataflow python tools/bench/compare_pressurefit_quality.py \
  --baseline HEAD --candidate WORKTREE --suite all \
  --output-dir results/pressurefit-quality/head-vs-worktree
```

See
[pressurefit-quality.md](../../docs/dataflow_sim/policy/pressurefit-quality.md)
for the isolation protocol, classifications, CLI controls, report schema, and
current validation evidence.

## pressurefit_exact_oracle.py — tiny-chain optimality oracle (CPU)

Exhaustively enumerates release/offload/prefetch annotations for a tiny bare
`TaskChain`, validates every candidate, and reports the simulator-optimal plan.
The search is exponential and fixes initial placement exactly as supplied; it
is development evidence, not a production planner.

```bash
conda run -n dataflow python tools/bench/pressurefit_exact_oracle.py \
  path/to/tiny-chain.json --max-assignments 1000000 \
  --output results/pressurefit-exact/tiny.json
```

Use it to measure PressureFit's approximation gap or to prove that a proposed
portfolio simplification preserves tiny-chain optima. It intentionally contains
no workload/model semantics.
