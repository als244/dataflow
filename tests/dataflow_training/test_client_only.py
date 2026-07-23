"""THE workload-test client contract, enforced.

Every test under tests/dataflow_training MUST drive the engine ONLY through the
client/daemon interface (daemon_client / out_of_process_daemon -> client.run ->
client.get_object). It must NOT construct an in-process Engine, use a
CudaBackend, wrap engine memory with torch_view, or reach an engine result's
object table (.objects). The client boundary hands back host copies, so no
engine device view can escape into a workload test — the whole use-after-free
class is then structurally impossible here.

Enforcement is a RATCHET: files still on the in-process path are listed in
LEGACY_INPROCESS. The gate fails if any file NOT on the list uses a forbidden
construct (a new violation, or a migrated file regressing), and fails if a
listed file no longer offends (so the list stays accurate and shrinks as the
migration proceeds). When LEGACY_INPROCESS is empty the rule is fully enforced.

Tests:
- test_workload_tests_are_client_only: no un-listed workload test constructs an
  in-process Engine, a CudaBackend, a torch_view, or reaches .objects; every
  listed legacy file still does (else it should be removed from the list).
"""
import ast
from pathlib import Path

from dataflow_training.distributed.topology import repo_root

WORKLOAD_TESTS = repo_root() / "tests" / "dataflow_training"

FORBIDDEN_IMPORT_NAMES = {"Engine", "CudaBackend", "torch_view"}
FORBIDDEN_ATTRS = {"objects"}

# Files still on the in-process engine path (audited 2026-07-23). Remove a file
# the moment it is migrated onto the client helper; when this set is empty the
# client-only rule is fully enforced across the workload suite.
LEGACY_INPROCESS: set[str] = {
    "models/test_dsv3.py",
    "models/test_dsv32.py",
    "models/test_engine_vs_reference.py",
    "models/test_glm52.py",
    "models/test_olmoe.py",
    "models/test_qwen35.py",
    "models/test_qwen35moe.py",
    "models/test_qwen3moe.py",
    "pretrain/test_client_fetch_surface.py",
    "pretrain/test_tp_layouts.py",
    "tasks/test_optim.py",
    "training/e2e/test_batch_ga.py",
    "training/e2e/test_freeze_plan.py",
    "training/e2e/test_ga_invariance.py",
    "training/e2e/test_lbl_modes.py",
    "training/e2e/test_packed_args_e2e.py",
    "training/lowering/test_round_prologue.py",
    "training/planning/test_planning.py",
}


def forbidden_uses(path: Path) -> list[tuple[int, str]]:
    """Every forbidden import name or object-table access in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORT_NAMES:
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            hits.append((node.lineno, f".{node.attr}"))
    return hits


def workload_test_files() -> list[Path]:
    return sorted(p for p in WORKLOAD_TESTS.rglob("test_*.py")
                  if "__pycache__" not in p.parts)


def test_workload_tests_are_client_only():
    offenders = {}
    for path in workload_test_files():
        hits = forbidden_uses(path)
        if hits:
            offenders[str(path.relative_to(WORKLOAD_TESTS))] = hits

    unexpected = {name: hits for name, hits in offenders.items()
                  if name not in LEGACY_INPROCESS}
    stale = sorted(LEGACY_INPROCESS - set(offenders))

    report = "\n".join(
        f"  {name}: " + ", ".join(f"{ln}:{what}" for ln, what in hits)
        for name, hits in sorted(unexpected.items()))
    assert not unexpected, (
        "workload tests must use the client/daemon interface only "
        "(no in-process Engine / CudaBackend / torch_view / .objects):\n"
        + report)
    assert not stale, (
        "these files no longer touch the engine directly — remove them from "
        "LEGACY_INPROCESS (the migration ratchet must shrink):\n  "
        + "\n  ".join(stale))
