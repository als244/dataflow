# Test layout

Mirrors the `src/dataflow` package split, plus model/module ladders:

| directory | scope |
|---|---|
| `tests/core/` | Program schema, validation, serialization |
| `tests/runtime/` | engine machinery: backend, placement, VMM, engine-vs-sim parity, stress (poison-on-free, interleaving) |
| `tests/tasks/` | unit-level task/kernel surface: registry ops, optimizer defs + policies, dtype policy, staged block executables |
| `tests/modules/` | shared building-block suites used by several families (moe, mla, dsa) — op pins + reference semantics |
| `tests/models/` | one ladder per model family (`test_<family>.py`) — the 11-gate canon `tools/verify_family.py` audits |
| `tests/training/` | the lowering → planning → E2E pipeline: shaped_program, planning, lowering-stability tripwires, batch/ga, multistep, varlen, plugins, webapp export |
| `tests/examples/` | CI gates for `examples/` |
