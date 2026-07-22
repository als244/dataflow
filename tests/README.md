# Test layout

Mirrors the package split: engine suites under `tests/dataflow/`,
workload suites under `tests/dataflow_training/`, twin-only suites under
`tests/reference_models/` (the truth tree). Multi-box gates live under
`tests/fleet/` and are opt-in (`pytest -m fleet`).

Every test module documents itself: its docstring ends with a `Tests:`
block — one `- test_name: summary` line per test function — and
`tests/test_docstring_index.py` fails if that block drifts from the
file's real tests. Environment requirements are declared as markers with
central probes (`tests/conftest.py`): `gpu`, `sim`, `corpus`,
`topology_remote`, `rdma`, `ncclbind`, `vram(gib=N)`, `fleet`. A machine
that lacks something skips with a precise reason instead of failing.

| directory | scope |
|---|---|
| `tests/dataflow/core/` | Program schema, validation, serialization |
| `tests/dataflow/runtime/` | engine machinery: backend, placement, VMM, engine-vs-sim parity, stress (poison-on-free, interleaving) |
| `tests/dataflow/service/` | dataflowd service surface: wire protocol, store, runs, peers, snapshots, determinism |
| `tests/dataflow_training/tasks/` | unit-level task/kernel surface: registry ops, optimizer defs + policies, dtype policy, staged block executables |
| `tests/dataflow_training/modules/` | shared building blocks used across families (moe, mla, dsa) — op pins + reference semantics |
| `tests/dataflow_training/models/` | one ladder per model family (`test_<family>.py`) — the canon `tools/verify/verify_family.py` audits |
| `tests/dataflow_training/training/lowering/` | program construction: shaped-program builder, layout registry, group annotation, parallelism scheme, round prologue, responsibility map, bit-identical lowering tripwires |
| `tests/dataflow_training/training/planning/` | PressureFit planning and recompute selection |
| `tests/dataflow_training/training/e2e/` | full engine runs vs golden: batch/ga, ga-invariance, dtype policy, LBL modes, packed/varlen feeds, freeze plans, the train CLI |
| `tests/dataflow_training/training/surfaces/` | external seams and tooling: plugin families, webapp export, checkpoint-record format, daemonize lifecycle |
| `tests/dataflow_training/pretrain/` | pretrain driver: presets, schedules, sharding, topology, reference muon, engine-parity families |
| `tests/dataflow_training/data/` | data plane: sequence packing |
| `tests/fleet/transport/` | the peer plane: loopback and cross-box object transfers, rdma host transport, hostmem collectives, nccl, collective dtypes, grad-exchange patterns, the p2p bench |
| `tests/fleet/dp/` | data parallelism: the DP step, cross-box vs solo parity, family-generic DP, ZeRO and its byte-equal variant, the conductor smoke, the world-1 conductor |
| `tests/fleet/tp/` | tensor parallelism: the TP MLP and the llama3-TP loopback |
| `tests/fleet/checkpoint_resume/` | checkpoint/resume drills — same-box and cross-box, world 1 and world 2 |
| `tests/reference_models/` | suites whose subject is the pure-torch twins themselves (see its README) |
| `tests/examples/` | CI gates for `examples/` |
| `tests/fixtures/` | shared fixtures (the external-family plugin) |

Top-level modules: `test_import_boundaries.py` (layering rules R1-R4),
`test_program_hashes.py`, `test_external_family.py`, and
`test_docstring_index.py` (the docstring-index gate).
