# dataflow — the engine package

A CPU–GPU dataflow runtime: program IR, a generic execution engine
over a `DeviceBackend`, and a persistent engine service (`dataflowd`).
The engine executes programs and knows NOTHING about models, training,
or any workload — that vocabulary lives across the seam in
`dataflow_training`, behind the resolver registry. The seam itself is
specified in [docs/program_contract.md](../../docs/program_contract.md).

## Public API surface

What the package exports and what tools/tests actually import:

- **`dataflow.core`** (stdlib-only at import time) — the program IR:
  `Program`, `TaskSpec`, `ObjectSpec`, `OutputSpec`,
  `TransferDirective`, `RecomputeOption`, `RecomputeRewrite`,
  `TensorMeta`, `DTYPE_BITS`, `dtype_nbytes`, `validate_program` /
  `ValidationError`, `load_program` / `save_program` /
  `program_from_dict` / `program_to_dict`, `SCHEMA_VERSION`.
  `dataflow.core.convert` (sim/webapp converters: `to_sim_chain`,
  `to_webapp_program`, `apply_chain_annotations`) imports
  `dataflow_sim` lazily inside functions.
- **`dataflow.runtime`** — the generic engine: `Engine`, `RunResult`,
  `ExecutionError`, `DeadlockError`, `Executable`,
  `ExecutableResolver`, `SyntheticExecutable`, `TaskContext`,
  `synthetic_resolver`, `RunTrace`, `compare_to_sim_eventlog`.
  Off-`__init__` surfaces callers use directly:
  `dataflow.runtime.device.cuda.CudaBackend`,
  `dataflow.runtime.device.fake.FakeBackend`,
  `dataflow.runtime.interop` (`torch_view`, `TORCH_DTYPE_BY_NAME`,
  `external_stream` — the ONE place torch touches raw runtime
  pointers), `dataflow.runtime.engine.Session`,
  `dataflow.runtime.placement`, `dataflow.runtime.pool.BufferPool`.
- **`dataflow.service`** — the engine service: `EngineClient`,
  `EngineConfig`, `Server`, `ServiceError`, `SCHEMA_VERSION`,
  `DEFAULT_SOCKET`. Sanctioned submodule surfaces (the ones the
  workload boundary test allows): `dataflow.service.client`,
  `dataflow.service.registry` (`register_program_resolver`,
  `registered_kinds`, `lookup_resolver` — THE workload seam), and
  `dataflow.service.wire` (`ServiceError`).

Per-layer contracts: [core/README.md](core/README.md),
[runtime/README.md](runtime/README.md),
[runtime/device/README.md](runtime/device/README.md);
service usage: [docs/engine_service.md](../../docs/engine_service.md),
engine API: [docs/engine_api.md](../../docs/engine_api.md).

## Dependency arrows

`dataflow` depends on **torch only** (plus stdlib): vendor bindings
live in `runtime/device/cuda*.py`, torch interop in
`runtime/interop.py`, and the service pins/executes through them.
Enforced boundaries (`tests/test_import_boundaries.py`):

- `dataflow.core` imports nothing heavy at import time (no
  torch/jax/cuda/dataflow_sim); its sim converters import
  `dataflow_sim` lazily in-function.
- importing `dataflow.runtime` pulls in no torch/jax/`dataflow_sim`
  (torch enters only via the cuda device backend / `interop`, which
  callers import explicitly).
- **R1**: nothing under `src/dataflow` imports `dataflow_training` or
  `reference_models` — the engine never sees the workload or the truth
  tree. Programs arrive parsed; resolvers arrive from the registry;
  buffers are store extents.

The workload direction (`dataflow_training -> dataflow`) is likewise
restricted to the public surfaces listed above (rule R2 — see
[docs/architecture.md](../../docs/architecture.md)).
