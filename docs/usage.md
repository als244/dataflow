# End-to-end usage: memory-constrained training

The full path from model config to multi-step training under a fast-memory
budget, as exercised by `tools/train/train_solo.py` and the drivers in
`dataflow_training/run/driver.py`.

```python
import torch
from dataclasses import replace
from dataflow.runtime.device.cuda import CudaBackend
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, derive_dims, lower_llama3
from dataflow_training.model_families.llama3.blocks import build_resolver
from dataflow_training.lowering.planning import plan_program
from dataflow_training.run.profiling import apply_measured_costs, cached_pcie, load_or_profile

cfg = ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=8, grad_accum_rounds=8)
backend = CudaBackend()

# 1. the machine, measured ONCE and disk-cached (re-measuring per run
#    makes plans non-reproducible: bandwidth noise flips recompute
#    choices). Plan with the BIDI numbers — the conservative doctrine:
#    each lane priced at its concurrent-saturation rate makes the
#    prediction a floor, and reality comes in at or better than it.
pcie = cached_pcie(backend)

# 2. lower with layout-exact sizes; install measured bandwidths
program = replace(lower_llama3(cfg),
                  bandwidth_from_slow=pcie.bidi_h2d,
                  bandwidth_to_slow=pcie.bidi_d2h)

# 3. task costs, measured and disk-cached (keyed by task signatures +
#    kernel set + device, so a kernel swap re-measures instead of lying)
profiles = load_or_profile(program, build_resolver(derive_dims(cfg)), backend)
planned = plan_program(apply_measured_costs(program, profiles),
                       fast_memory_capacity=16 * 1024**3,
                       recompute=True,
                       build_variant=lambda lv: apply_measured_costs(
                           lower_llama3(cfg, recompute_levels=lv), profiles))
```

(The maintained one-call version of steps 1-3 — measured bandwidths
threaded onto every recompute variant too — is
`dataflow_training.run.driver.plan_at_budget(cfg, budget_gib,
measured=True)`; the drivers all go through it.)

Execution goes through the ENGINE SERVICE (`dataflowd` — the runtime
engine hosted behind a store of persistent objects;
[engine_service.md](engine_service.md)): register the planned program
once with the family's resolver spec, seed W/O via init-as-program,
then `run()` per optimizer step — weights and optimizer state live in
the daemon's store and evolve in place between runs. The reference
driver wraps the whole loop:

```python
from dataflow_training.data.fineweb import make_feed
from dataflow_training.run.driver import daemon_client, run_engine
from dataflow_training.run.recipe import Recipe

recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=10, total_steps=100)
feed = make_feed(cfg.tokens)                        # deterministic fineweb rounds
with daemon_client(slab_gib=60.0) as client:        # boots an in-process dataflowd
    res = run_engine(client, cfg, recipe, feed, steps=100,
                     budget_gib=16.0)               # plans + registers + runs
print(res.losses)
# quote wall tok/s in results: res.tok_per_s times the whole run verb
# (execution + loss readback); makespan-only numbers flatter the seam
```

(`run_engine` internally does `plan_at_budget` -> `init_model` ->
`register_program` -> `run(args={"step": k, "valid_rows": ...},
fetch=[loss_...])`, with optional checkpoint/resume — read
`dataflow_training/run/driver.py` for the verb-by-verb version, or
[engine_service.md](engine_service.md) for the raw client calls.)

What the pieces guarantee:

- **Sizes are exact** (packed layouts), **costs are measured** (profiling
  harness), so the simulator's prediction for `planned.program` is an honest
  target; `dataflow_training.lowering.replay.replay_gap_pct` isolates
  scheduling overhead from any residual duration error.
- **Correctness is reference-checked**: the gradcheck ladder
  (`dataflow_training.testing.gradcheck`) pins ops, blocks, and full steps
  against the isolated pure-torch twins (`reference_models/`,
  [correctness_compare.md](correctness_compare.md)); plan-invariance,
  poison-on-free, and interleaving-stress tests guard the async machinery.
- **Steady state does zero vendor allocation**: the daemon keeps the device
  slab and pinned pools across steps; step-0 pays setup, later steps must
  report `slab_overflows == 0`.

Visualize any program in the webapp simulator:
`dataflow.core.convert.to_webapp_program(program)` produces the upload JSON
(cost subops included, so hardware sliders re-resolve runtimes). A REAL run
uploads too: `tools/export/trace_real_run.py` drives a few daemon steps with the
run verb's trace and packages the measured event log together with the
sim's prediction of the same plan into one `*.measured.json` the webapp
renders side by side (see [exporting_runs.md](exporting_runs.md)).

## The CLI instead

```
python tools/train/train_solo.py smoke                       # tiny real-vocab reference-vs-engine gate
python tools/train/train_solo.py parity --preset l3_125m ... # one preset, reference + engine at N budgets
python tools/train/train_solo.py scaling --presets l3_125m,l3_1b ...  # the ladder, loss curves
python tools/train/train_solo.py reference --preset gpt2_124m --data doc \
    --checkpoint-every 250 --out results/pretrain/ref.json   # pure-torch leg
python tools/train/train_solo.py engine --preset gpt2_124m --data doc \
    --budget 14 --resume --out results/pretrain/eng.json     # engine leg
python tools/train/train_fleet.py train --preset l3_1b --steps 1000 --rounds 6,2 \
    --out results/pretrain/fleet.json                        # data-parallel fleet
python tools/bench/measure_step.py --preset l3_1b --t-rounds 8192,32768 \
    --budgets 14,6 --steps 12          # measured throughput sweeps
```

The `reference` and `engine` legs are the long-run pretraining pair:
same recipe, same deterministic feed, overlaid loss curves. `--data
doc` selects the doc-aware feed (sequences split at document EOT
boundaries, positions restarting per document) over fixed-size block
packing; both legs support checkpointing (`--checkpoint-every`) and
`--resume`. Sweep rows from `measure_step` report the plan's predicted
s/step beside the measurement (`pred_s meas_s ratio tok/s effTF/s
hwTF/s recomp`) — tool guide: [benchmarking.md](benchmarking.md).

## Profiling a run with Nsight Systems (device metrics + NVTX)

Annotations are OFF in normal runs. The annotation layer
(`runtime/device/annotate.py`) is a switchable, vendor-portable
protocol (`range_push`/`range_pop`/`mark` plus
`start_capture`/`stop_capture`); the daemon's `profiler_control` verb
flips it on for a bracketed window of steps, and inside that window
the engine annotates every task launch (ids like
`block_bwd_{step}_{round}_{layer}`) and transfer (like
`from_slow:A_{step}_3_16`). An AMD annotator slots into the same
protocol with roctx.

`tools/bench/nsys_profile.py` packages the whole recipe: it wraps a
`train_solo.py engine` run in `nsys profile` with
`--capture-range=cudaProfilerApi`, and the run brackets steps
`--start` through `--stop` via `profiler_control` — so the report
holds exactly those warmed steps, with planning/boot noise excluded.
Open the report in the nsys GUI and use the NVTX projection rows to
read task ranges on the stream timelines. GPU metrics sampling
(`--gpu-metrics-devices`) needs perf-counter permission.

    python tools/bench/nsys_profile.py --preset gpt2_124m --steps 10 \
        --start 5 --stop 8 --out gpt2_capture

For the rest of the benchmarking workflow (predict → measure →
profile, profile-cache discipline): see
[benchmarking.md](benchmarking.md).
