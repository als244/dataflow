# dataflow.training — lowering, planning, profiling, testing

**Purpose.** Turn model definitions into executable programs and keep them
honest: lowering (structure + exact sizes), planning via dataflow_sim
(PressureFit + recompute — the ONLY layer importing the simulator),
measurement (profiling), and the correctness harness (gradcheck).

## Modules

- `shaped_llama3.py` — llama3-shaped program generator (structure, naming,
  grad-accum mutation pattern, recompute variants + rewrites). Sizes are
  analytic; used for planning/parity work at any scale.
- `llama3_lowering.py` — execution-grade lowering: shaped structure with
  sizes rewritten to the tasks layer's packed layouts (`lower_llama3`), plus
  `initial_values()` filling pinned buffers with real weights/data.
- `planning.py` — `plan_program(program, fast_memory_capacity=, recompute=,
  build_variant=)` wraps `apply_pressurefit_policy` + `plan_with_recompute`;
  `simulate_program` runs the simulator. Swapping the policy happens here
  and only here.
- `profiling.py` — `profile_program` measures per-unique-task runtime (CUDA
  events, median) and workspace (torch-allocator peak delta);
  `apply_measured_costs` writes them back. Final planning runs on measured
  costs.
- `testing/gradcheck.py` — the correctness ladder: `check_block_backward`
  (vs autograd through the golden block; recompute-equivalence; accumulation
  semantics) and `check_model_step` (full annotated program through the real
  engine vs the golden model: loss, final params, optimizer state).

## The E2E recipe (what the M3 gate runs)

```python
program = lower_llama3(cfg)                          # exact sizes
profiles = profile_program(program, resolver, be)    # measure
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
