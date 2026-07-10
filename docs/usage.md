# End-to-end usage: memory-constrained training

The full path from model config to multi-step training under a fast-memory
budget, as exercised by `tools/bench_train.py`.

```python
import torch
from dataclasses import replace
from dataflow.runtime.device.cuda import CudaBackend
from dataflow.tasks.models.llama3_blocks import build_resolver
from dataflow.training.llama3_lowering import dims_of, lower_llama3
from dataflow.training.planning import plan_program
from dataflow.training.profiling import apply_measured_costs, cached_pcie, load_or_profile
from dataflow.training.shaped_llama3 import ShapedLlamaConfig
from dataflow.training.train_loop import train

cfg = ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=8, grad_accum_rounds=8)
backend = CudaBackend()

# 1. the machine, measured ONCE and disk-cached (PCIe directions contend on
#    desktop platforms — plan with the bidirectional numbers; re-measuring
#    per run makes plans non-reproducible: bandwidth noise flips recompute
#    choices)
pcie = cached_pcie(backend)

# 2. lower with layout-exact sizes; install measured bandwidths
program = replace(lower_llama3(cfg),
                  bandwidth_from_slow=pcie.bidi_h2d,
                  bandwidth_to_slow=pcie.bidi_d2h)

# 3. task costs, measured and disk-cached (keyed by task signatures +
#    kernel set + device, so a kernel swap re-measures instead of lying)
profiles = load_or_profile(program, build_resolver(dims_of(cfg)), backend)
planned = plan_program(apply_measured_costs(program, profiles),
                       fast_memory_capacity=16 * 1024**3,
                       recompute=True,
                       build_variant=lambda lv: apply_measured_costs(
                           lower_llama3(cfg, recompute_levels=lv), profiles))

# 4. train: one annotated chain replayed per optimizer step; persistent
#    state lives in pinned buffers the plan's offloads update in place.
#    Static placement (default) packs every fast allocation offline and
#    raises PlacementError at PLANNING time if it cannot fit physical VRAM.
report = train(planned.program, cfg, backend, steps=100)
print(report.losses)
# quote wall_tokens_per_s in results: it covers the FULL step
# (fill + execute + readback), makespan-only numbers flatter the seam
```

What the pieces guarantee:

- **Sizes are exact** (packed layouts), **costs are measured** (profiling
  harness), so the simulator's prediction for `planned.program` is an honest
  target; `dataflow.training.replay.replay_gap_pct` isolates scheduling
  overhead from any residual duration error.
- **Correctness is reference-checked**: the gradcheck ladder
  (`dataflow.training.testing`) pins ops, blocks, and full steps to the
  golden autograd model; plan-invariance, poison-on-free, and
  interleaving-stress tests guard the async machinery.
- **Steady state does zero vendor allocation**: the Session keeps the device
  slab and pinned pools across steps; step-0 pays setup (slab, pinning,
  possible headroom overflows), later steps must report
  `step_slab_overflows == 0`.

Visualize any program in the webapp simulator (webapp/):
`dataflow.core.convert.to_webapp_program(program)` produces the upload JSON
(cost subops included, so hardware sliders re-resolve runtimes). A REAL run
uploads too: `tools/gap_analysis.py` (or a replay that persists its trace)
plus `tools/export_measured_run.py` produce a `*.measured.json` the webapp
renders in the main window next to the sim's prediction of the same plan.

## The CLI instead (what the results tables were built with)

```
python tools/bench_train.py --config 8b-s1k-bs8ga8 --device-gib 12,16,20 --steps 3 --recompute
python tools/bench_train.py --config 8b-s1k-bs8ga8 --device-gib 24 --probe-max ...   # largest feasible budget
python tools/bench_train.py --config ... --device-gib 29 ...   # HARD device envelope (actual usage <= 29)
python tools/bench_train.py --config ... --annotated <saved>.annotated.json  # exact replay of a saved plan
python tools/gap_analysis.py --config ... --budget ...      # real-vs-sim gap decomposition
python tools/window_plans.py --config ... --device-gib ...     # k-step window oracle (step-seam analysis)
```

Every summary row reports `real_tokens_per_s` (makespan) AND
`wall_tokens_per_s` (full step) plus `placement_escapes` /
`pressure_evictions` (both 0 in healthy runs). A/B flags for the planning
defaults: `--optimizer interleaved|tail`, `--preplace task0|greedy`.

## Profiling a run with Nsight Systems (device metrics + NVTX)

```
python tools/nsys_profile.py                              # bs8/ga8 @ 24 GiB, 3 steps
python tools/nsys_profile.py --config 8b-s1k-bs2ga32 --budget 16 --steps 2
python tools/nsys_profile.py --no-gpu-metrics --capture full --stats
```

Produces `artifacts/nsys/<config>-<budget>gib-<steps>steps.nsys-rep` with:
GPU metrics sampled on the timeline (`--gpu-metrics-devices=all`; needs
perf-counter permission — the script prints the fix if denied), CUDA +
NVTX + OS-runtime traces, and the runtime's own NVTX ranges: one per task
(`block_bwd_{step}_{round}_{layer}` — the display name substitutes the
GLOBAL step into the replayed plan's ids), one per transfer
(`from_slow:A_{step}_3_16`), one per optimizer step (`step:N`). Open in the nsys GUI and use the NVTX
projection rows to read ranges on the stream timelines. By default capture
is limited to the `train_steps` range, so planning/setup stay out of the
report; profiles and PCIe numbers come from the disk caches.

The annotation layer is vendor-portable (`runtime/device/annotate.py`):
the engine calls a 3-method protocol (`range_push/range_pop/mark`),
enabled by `DATAFLOW_NVTX=1`. An AMD backend implements the same protocol
with roctx and the same script structure wraps `rocprofv3`.

For benchmarking (which tool, standard recipes, row semantics, placement modes, profile cache): see [benchmarking.md](benchmarking.md).
