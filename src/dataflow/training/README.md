# dataflow.training — lowering, planning, profiling, testing

**Purpose.** Turn model definitions into executable programs and keep them
honest: lowering (structure + exact sizes), planning via dataflow_sim
(PressureFit + recompute — the ONLY layer importing the simulator),
measurement (profiling), and the correctness harness (gradcheck).

## Modules

- `shaped_llama3.py` — llama3-shaped program generator (structure, naming,
  grad-accum mutation pattern, recompute variants + rewrites). Sizes are
  analytic; used for planning/parity work at any scale.
  `optimizer_placement="interleaved"` (default) emits each optimizer task
  at its gradient's final mutation inside the last backward round —
  `"tail"` restores the legacy end-of-chain order, which drains 1.5–2 s of
  GPU-idle PCIe per step.
- `llama3_lowering.py` — execution-grade lowering: shaped structure with
  sizes rewritten to the tasks layer's packed layouts (`lower_llama3`), plus
  `initial_values()` filling pinned buffers with real weights/data.
- `planning.py` — `plan_program(program, fast_memory_capacity=, recompute=,
  build_variant=, preplace="task0")` wraps `apply_pressurefit_policy` +
  `plan_with_recompute`; `simulate_program` runs the simulator. Swapping the
  policy happens here and only here. `preplace="task0"` (the runtime
  default; the sim's own default stays "greedy") keeps t=0 pre-placement to
  task 0's inputs so head uploads are planned, charged, overlappable
  transfers instead of a silent synchronous setup copy.
- `profiling.py` — `profile_program` measures per-unique-task runtime (CUDA
  events, thermal-soaked, distribution-checked) and workspace
  (torch-allocator peak delta); `apply_measured_costs` writes them back.
  USE THE CACHES: `load_or_profile` (keyed by task signatures + kernel set
  + env + device) and `cached_pcie` — re-measuring per run makes plans
  non-reproducible (bandwidth noise flips recompute choices). Final
  planning always runs on measured costs.
- `train_loop.py` — `train(annotated, cfg, backend, steps=, placement_mode=
  "static")`: one annotated chain replayed per optimizer step; Session-owned
  slab/pools; static placement dry-run + packing by default. The report
  carries `step_wall_s` (FULL step: fill + execute + readback — quote wall
  tok/s, makespan-only numbers flatter the seam), `placement_escapes`, and
  `pressure_evictions` (both 0 in healthy runs).
- `families.py` — the model-family registry: `resolve_family(cfg)` maps a
  shaped config to its lowering, dims, resolver, golden, and gradcheck
  bundle. The train loop, gradcheck, and sweep tools dispatch through it;
  adding a family is one entry here (docs/extending.md §6). Families:
  llama3 (`shaped_llama3` + `llama3_lowering`), qwen3 (`shaped_qwen3` +
  `qwen3_lowering` — qk-norm, decoupled head_dim, vocab 151,936).
- `replay.py` — `replay_gap_pct`: re-simulate with measured durations as
  overrides; isolates scheduling fidelity from cost-model error.
- `testing/gradcheck.py` — the correctness ladder: `check_block_backward`
  (vs autograd through the golden block; recompute-equivalence; accumulation
  semantics) and `check_model_step` (full annotated program through the real
  engine vs the golden model: loss, final params, optimizer state).

## The E2E recipe (what the model-step gate runs)

```python
program = lower_llama3(cfg)                          # exact sizes
profiles = load_or_profile(program, resolver, be)    # measure (disk-cached)
measured = apply_measured_costs(program, profiles)
planned = plan_program(measured, fast_memory_capacity=cap)
values = initial_values(planned.program, cfg, be)
dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
result = Engine(be).execute(planned.program, resolver=build_resolver(dims),
                            initial_buffers=values, pool_prewarm=dry.pool_demand)
```

## Invariants the tests enforce

- **Plan-invariance**: different budgets/recompute plans ⇒ identical math.
- **Poison-on-free** (`Engine(poison_on_free=True)`): freed buffers filled
  with NaN pattern; any use-after-release explodes loudly.
- **Interleaving stress**: random device delays before each task ⇒ identical
  results (event-ordering correctness).
- **Measured-cost replanning** changes the plan, never the math.
