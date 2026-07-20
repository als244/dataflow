# Benchmarking: prediction, measurement, and profiling

Three tools cover the throughput workflow, in escalating cost:

## 1. Predict — `tools/predict_step.py` (CPU, instant)

Simulated sweeps over geometry × memory: lowers the true program,
plans each cell, reads the simulator-verified schedule back as a table
(s/step, tok/s, effective/hardware TFLOPs/s, fast/backing peaks, PCIe
bytes + link %, recompute + idle %). `--measured` swaps roofline cost
seeds for profiled task costs (disk-cached; needs the GPU once per
geometry). Full guide: [throughput.md](throughput.md).

    python tools/predict_step.py --preset gpt2_124m --hw 3090 \
        --t-rounds 8192,32768,65536 --tokens-step 524288 \
        --budgets 16,8,4,2 --steps 10000

Both sweep tools take any `resolve_preset` name (the table:
[builtin_models.md](builtin_models.md)) and `--plugin` for external
families.

## 2. Measure — `tools/measure_step.py` (GPU, ~minutes)

The measured twin: the same grid interface, but each cell RUNS the
engine for `--steps` steps through one shared daemon (store wiped
between cells) and reports the warmed measurement beside the
prediction for that cell's plan — `pred_s`, `meas_s`, their ratio,
tok/s, and TFLOPs/s from real wall time (the first 3 steps of each
cell are warmup, excluded from the mean; `--measured-plan` prices the
prediction column from profiled task costs instead of roofline). This is the ground truth the
sim calibrates against; a persistent prediction/measurement gap at
some geometry is a finding (see the calibration table in
throughput.md), not a tolerance to widen.

    python tools/measure_step.py --preset gpt2_124m \
        --t-rounds 8192,65536 --tokens-step 524288 \
        --budgets 14,4 --steps 12 --data doc

## 3. Profile — `tools/nsys_profile.py` (GPU, one capture)

Wraps a `train_solo.py engine` run in Nsight Systems with the
canonical trace set (`cuda,nvtx,osrt,cublas,cudnn` + GPU metrics) and
the cudaProfilerApi capture range: the driver brackets the requested
step window through the daemon's `profiler_control` verb, so the
report holds exactly those steps — warmed, no boot noise. Reports land
under `results/pretrain/logs/`.

    python tools/nsys_profile.py --preset gpt2_124m --ga-rounds 8 \
        --batch 64 --data doc --steps 10 --start 5 --stop 8 \
        --out gpt2_124m_ga8

Fleet-scale profiling (multi-daemon, per-rank reports) is
`tools/train_fleet.py train --profile ...` — see
[distributed_training.md](distributed_training.md).

## Discipline

- Plans and their costs are DETERMINISTIC inputs: the profile cache
  keys on task signatures + kernel set + device, and PCIe bandwidths
  are pinned once (`cached_pcie`) — re-measuring per run makes plans
  non-reproducible.
- If a kernel changes, bump `PROFILE_CACHE_REV`
  (`dataflow_training/run/profiling.py`); stale cached costs silently
  skew both the sim and the planner's recompute choices.
- Every engine run logs its own prediction (`[plan] predicted ...`)
  and per-step measured `tok/s` + `eff/hw TF/s`, so
  prediction-vs-reality on ANY run is one grep of its log.
