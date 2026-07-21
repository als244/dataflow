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
| `--t-round` / `--t-rounds a,b,c` | round token budget (single point / sweep axis) |
| `--tokens-step` | tokens per optimizer step (ga = tokens-step / t_round) |
| `--ga-rounds` | alternative to `--tokens-step` |
| `--seq-len` / `--seq-lens` | sequence length (third sweep axis) |
| `--budget` / `--budgets` | device budget GiB (single / sweep) |
| `--backing` | host-slab capacity ceiling GiB (plans escalate recompute to fit; infeasible combos report as INFEASIBLE rows) |
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
| `--slab` | daemon pinned slab GiB |
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

## internal/

Maintainer-local kernel microbenches (gitignored — not part of the
repo surface): per-op MoE head-to-heads, fla delta-rule + conv A/B at
qwen3.5 shapes, VMM slab primitive latencies.
