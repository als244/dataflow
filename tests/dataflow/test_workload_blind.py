"""THE engine-test boundary, enforced: tests/dataflow is workload-blind.

Every test under tests/dataflow exercises the ENGINE (core IR, runtime,
service) and MUST NOT import anything from dataflow_training — engine
behavior is specified against generic programs and resolvers, never
against a workload's lowerings, families, drivers, or presets. Tests
whose SUBJECT is workload machinery live under tests/dataflow_training
(where the client-only contract governs them instead).

Enforcement is a RATCHET, exactly like the workload suite's client-only
gate: files still importing dataflow_training are listed in
LEGACY_WORKLOAD_IMPORTS. The gate fails if any file NOT on the list
imports dataflow_training (a new violation, or a regression), and fails
if a listed file no longer does (the list stays accurate and shrinks as
tests are rewritten onto generic fixtures). Empty list = fully enforced.

Tests:
- test_engine_tests_are_workload_blind: no un-listed test under
  tests/dataflow imports dataflow_training; every listed legacy file
  still does (else it must leave the list).
"""
import ast
from pathlib import Path

ENGINE_TESTS = Path(__file__).resolve().parent

# Files still importing dataflow_training (audited 2026-07-27, Shein's
# ruling: engine tests go workload-blind). Remove a file the moment its
# fixtures are generic; when this set is empty the rule is fully
# enforced across the engine suite.
LEGACY_WORKLOAD_IMPORTS: set[str] = {
    # The profiling memory-lifecycle gate: its SUBJECT is workload-
    # agnostic (reserved memory returns to baseline per cost table) but
    # its fixture lowers a tiny family and the profiler module itself
    # lives training-side today. Leaves the list when the sweep gives
    # it a generic program fixture (or profiling a workload-blind home).
    "runtime/test_profiling_memory.py",
    "core/test_json_roundtrip.py",
    "core/test_sim_convert.py",
    "runtime/test_cuda_backend.py",
    "runtime/test_engine_stress.py",
    "runtime/test_parity_vs_sim.py",
    "runtime/test_placement.py",
    "runtime/test_trace_dict.py",
    "runtime/test_vmm.py",
    "service/test_daemon_relaunch.py",
    "service/test_engine_determinism.py",
    "service/test_error_codes.py",
    "service/test_registration.py",
    "service/test_service_packed_args.py",
    "service/test_service_runs.py",
    "service/test_service_skeleton.py",
    "service/test_service_snapshot.py",
    "service/test_service_store.py",
    "service/test_shared_server_self_heal.py",
}


def workload_imports(path: Path) -> list[int]:
    """Line numbers of every dataflow_training import in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "dataflow_training"
                   for a in node.names):
                hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "dataflow_training":
                hits.append(node.lineno)
    return sorted(hits)


def test_engine_tests_are_workload_blind():
    offenders = {}
    for path in sorted(ENGINE_TESTS.rglob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        hits = workload_imports(path)
        if hits:
            offenders[str(path.relative_to(ENGINE_TESTS))] = hits
    new = {name: lines for name, lines in offenders.items()
           if name not in LEGACY_WORKLOAD_IMPORTS}
    stale = sorted(LEGACY_WORKLOAD_IMPORTS - set(offenders))
    assert not new, (
        "engine tests must not import dataflow_training (write generic "
        "fixtures, or the test belongs under tests/dataflow_training):\n  "
        + "\n  ".join(f"{name}: lines {lines}" for name, lines in
                      sorted(new.items())))
    assert not stale, (
        "clean files still listed in LEGACY_WORKLOAD_IMPORTS (the "
        "migration ratchet must shrink):\n  " + "\n  ".join(stale))
