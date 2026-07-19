# Throughput estimation — `tools/predict_step.py`

The first line of attack for "how fast should this train, and at what
memory shape?" — before any long run. It lowers the true program (all
tasks, optimizer included), plans it at a device budget, and reads the
simulator-verified schedule back as numbers: the same panels the sim
webapp draws, as table columns.

## Quick start

Single point (detail view — compute sums, model flops, top tasks):

    python tools/predict_step.py --preset gpt2_124m --ga-rounds 64 \
        --budget 14 --hw 3090 --steps 10000

Sweep — one row per (seq_len × T_round × budget) combination:

    python tools/predict_step.py --preset gpt2_124m --hw 3090 \
        --t-rounds 8192,32768,65536 --tokens-step 524288 \
        --budgets 16,8,4,2 --measured --steps 10000

Geometry speaks **T_round** (the round token budget) with `ga` derived
from `--tokens-step`; "batch" is internal arithmetic under varlen
packing (`T_round / seq_len`), never an input. `--seq-len/--seq-lens`
is the third axis; `--opt {adamw,muon}` sizes optimizer state (and its
NS work) correctly; `--backing N` sets the host-slab ceiling — the
planner escalates recompute to fit BOTH ceilings, and combos it cannot
fit report as INFEASIBLE rows, not crashes.

## Columns

    seq T_round ga tok/step budget | s/step tok/s effTF/s hwTF/s |
    fastGiB bkGiB | h2dGB h2d% d2hGB d2h% | rc% idle% | recomp | ETA_h

- `effTF/s` / `hwTF/s`: EFFECTIVE (algorithmic fwd + bwd + optimizer
  matmuls) and HARDWARE (+ recompute replays + flash-internal
  recompute) flops over the predicted step time — the conventions of
  `lowering/flops.py`, sourced from the same `cost_subops` the sim
  prices. Doc-aware feeds scale the causal-attention share by the
  actual per-round quadratic mass.
- `fastGiB` / `bkGiB`: the plan's device and host peaks. Backing is
  W+O+data plus whatever the plan offloads — at ample fast budget the
  transient share is ZERO (e.g. l3_1b@14GiB: bkGiB 6.59 == W 2.20 +
  adamw O 4.40); it grows as fast tightens.
- `h2d/d2h` GB + link %: per-direction PCIe bytes and busy fraction of
  the step (the webapp's link panels).
- `rc%` / `idle%`: compute-track time in recompute tasks / with no
  compute running.
- `recomp`: activations the planner chose to recompute, of rewritable.

## Roofline vs `--measured`

Roofline (default) is CPU-only and instant: costs from ShapedHardware
profiles (`--hw 5090|3090`, override with `--tflops/--bw/--pcie`).
`--measured` profiles every task signature on the real GPU
(disk-cached per geometry + kernel set + device; budget rows reuse one
pass) and re-costs recompute variants through the same profiles.

Calibration (gpt2_124m + l3_1b, 2026-07-19):

- measured-cost predictions: **~1–3 % of reality at fat rounds**
  (T_round ≥ 16K); ~12 % optimistic at T_round 8192 (per-task fixed
  costs the sim does not charge).
- roofline: ~12–19 % off, either direction (l3_1b measured BEAT its
  roofline).
- tight budgets add unmodeled transfer pressure: at 2 GiB expect
  ~10–15 % pessimism on top of any prediction (per-transfer fixed
  costs + kernel slowdown under concurrent DMA; the sim models both
  FIFO links and capacity-blocked queues, but moves bytes at pure
  size/bandwidth with a zero-latency scheduler).
- the two cost models can produce DIFFERENT PLANS near memory edges
  (roofline picked recompute where measured picked streaming, and
  measured rejected a combo roofline accepted) — trust `--measured`
  near the edges.

The engine driver prints the same prediction per run — the `[plan]`
line (`predicted s/step, peak fast/backing, recompute, h2d/d2h`) — and
every step logs measured `tok/s` + `eff/hw TF/s`, so prediction vs
reality is one grep.

The REFERENCE (pure-torch twin) leg has no simulator model — its
throughput is measured only (and expected well below the engine's; the
twins are correctness oracles, deliberately unoptimized).
