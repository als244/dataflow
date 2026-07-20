# The Dataflow Engine API

The engine executes an annotated [Program](program_schema.md) on real
hardware. It is deliberately small and name-agnostic: everything
model-specific lives in the program and the resolver.

## Construction

```python
from dataflow.runtime import Engine
from dataflow.runtime.device.cuda import CudaBackend   # or fake.FakeBackend

engine = Engine(
    backend,                       # DeviceBackend (CUDA or Fake)
    validate=True,                 # validate_program before running
    strict_final_locations=True,   # violations of the replay contract fail loudly
    poison_on_free=False,          # overwrite freed buffers (use-after-free tripwire)
    session=None,                  # Session: reuse pool/slabs across execute() calls
)
```

`FakeBackend` executes the full protocol without a GPU (no kernels, no
transfers) — use it for structural dry runs and to harvest
`pool_demand` for prewarming the real run.

## execute()

```python
result = engine.execute(
    program,                # the ANNOTATED Program (post-PressureFit)
    resolver=...,           # callable: task -> executable with .launch(ctx);
                            # dispatch on task.compute_block_key. Executables
                            # obey docs/task-contract.md (no host syncs, no
                            # D2H readbacks, deterministic kernels)
    initial_buffers=...,    # {object_id: Buffer} pinned host buffers for
                            # every initial object (weights, inputs, ...)
    pool_prewarm=...,       # {(location, size): count} from a prior run's
                            # pool_demand — kills first-step allocation churn
    placement=None,         # precomputed static placement (else the
                            # dynamic pool/slab allocation path)
    record_placement=None,  # PlacementRecorder: harvest lifetimes (dry runs)
    vmm=False,              # non-contiguous arena placement
                            # (runtime/device/vmm.py)
    run_args=None,          # opaque per-run values -> TaskContext.run_args
                            # (step, lr, seq_lens, ... — tasks interpret)
    groups=None,            # per-run peer-group handles -> TaskContext.groups
    cancel_event=None,      # threading.Event: cancel at the next task
                            # boundary (the service's cancel_run)
    annotate_rename=None,   # Callable[[str], str]: NVTX display names
)
```

### Execution semantics

- Tasks run in CHAIN ORDER — the program's task order IS the schedule.
  A task dispatches once its inputs and mutated objects are resident in
  fast memory and space is reserved for its outputs.
- Compute and the two transfer directions ride separate CUDA streams;
  the completion directives (`releases_after` / `offload_after` /
  `prefetch_after`) drive all data movement. A transfer starts only
  after destination memory is reserved.
- The engine is deterministic: same program + same buffers ⇒ same
  bytes (the correctness gates byte-compare across plans).
- Replay: with `final_locations` equal to initial locations for every
  persistent object, the SAME annotated program executes repeatedly —
  one program per optimizer step. Use a `Session` to keep the pool,
  slabs, and pinned host memory alive across `execute()` calls
  (steady-state steps perform zero vendor allocations).

## RunResult

| field | meaning |
|---|---|
| `makespan_us` | wall time of the run (compute clock) |
| `trace` | per-task/per-transfer timeline (feeds replay-fidelity + webapp export) |
| `peak_fast_bytes` / `peak_backing_bytes` | measured memory peaks (ledger view) |
| `objects` | final object table — read results (losses, updated weights) through `result.objects[id]` views after the run |
| `pool_demand` | exact `(location, size) -> count` buffer demand; feed to the next run's `pool_prewarm` |
| `final_location_violations` | replay-contract breaches (empty unless `strict_final_locations=False`) |
| `slab_overflows` / `placement_escapes` / `pressure_evictions` | pool-health counters (0 in healthy runs) |
| `buffers_allocated` / `buffers_reused` | allocation-churn accounting |

Call `result.close()` after readback when no `Session` owns the pool —
one-shot runs that skip it leak the device slab for the process
lifetime.

## The two-phase pattern

Nearly every caller (the engine service, gradcheck, the RL examples) runs:

```python
dry = Engine(FakeBackend()).execute(program, initial_buffers=values)
result = Engine(backend).execute(program, resolver=resolver,
                                 initial_buffers=values,
                                 pool_prewarm=dry.pool_demand)
```

See [extending_programs.md](extending_programs.md) for building custom
programs end to end, and [task-contract.md](task-contract.md) for what
executables may do inside `launch(ctx)`.
