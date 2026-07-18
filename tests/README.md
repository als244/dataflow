# Test layout

Mirrors the package split: engine suites under `tests/dataflow/`,
workload suites under `tests/dataflow_training/`, twin-only suites
under `tests/reference_models/` (the truth tree).

| directory | scope |
|---|---|
| `tests/dataflow/core/` | Program schema, validation, serialization |
| `tests/dataflow/runtime/` | engine machinery: backend, placement, VMM, engine-vs-sim parity, stress (poison-on-free, interleaving) |
| `tests/dataflow/service/` | dataflowd service surface: wire protocol, store, runs, peers, snapshots, determinism |
| `tests/dataflow_training/tasks/` | unit-level task/kernel surface: registry ops, optimizer defs + policies, dtype policy, staged block executables |
| `tests/dataflow_training/modules/` | shared building-block suites used by several families (moe, mla, dsa) — op pins + reference semantics |
| `tests/dataflow_training/models/` | one ladder per model family (`test_<family>.py`) — the 11-gate canon `tools/verify_family.py` audits |
| `tests/dataflow_training/training/` | the lowering → planning → E2E pipeline: shaped_program, planning, lowering-stability tripwires, batch/ga, multistep, varlen, plugins, webapp export |
| `tests/dataflow_training/pretrain/` | pretrain driver: presets, schedules, sharding, topology, fineweb, reference muon, engine-parity families |
| `tests/dataflow_training/data/` | data plane: sequence packing |
| `tests/reference_models/` | suites whose subject is the pure-torch twins themselves (see its README) |
| `tests/fleet/` | multi-box lane (opt-in: `pytest -m fleet`) |
| `tests/examples/` | CI gates for `examples/` |
| `tests/fixtures/` | shared fixtures (external-family plugin, program hashes) |

Top-level modules: `test_import_boundaries.py` (layering rules R1-R4),
`test_program_hashes.py`, `test_external_family.py`.
