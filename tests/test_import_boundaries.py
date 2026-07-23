"""Layering rules, enforced.

Runtime checks (fresh interpreter per check, so prior imports can't
mask a transitive leak):

- ``dataflow.core`` imports nothing heavy (no torch/jax/cuda/dataflow_sim).
- ``dataflow.runtime`` may not import torch, jax, or dataflow_sim.
- ``dataflow_training.blocks`` may not pull in dataflow_sim.

Static checks (ast scan of every import statement — docstring mentions
are naturally fine):

- R1: no module under ``src/dataflow`` imports ``dataflow_training``
  or ``reference_models``. The engine never sees the workload or the
  truth tree.
- R2: modules under ``src/dataflow_training`` import ``dataflow`` only
  through its public surfaces: ``dataflow.core.*``,
  ``dataflow.runtime.*`` (the ABIs), ``dataflow.service`` itself
  (Server/EngineConfig/EngineClient for rigs), ``dataflow.service.client``,
  ``dataflow.service.registry`` (register_program_resolver), and
  ``dataflow.service.wire`` (ServiceError).
- R3: files under ``tools/`` stay near package roots: ``dataflow``
  itself, the ``dataflow.core``/``dataflow.runtime``/``dataflow.service``
  subtrees, and ``dataflow_training`` at most two levels deep
  (``dataflow_training.x.y``) — deeper only inside
  ``dataflow_training.model_families`` (family packages) and
  ``dataflow_training.blocks``. This is the tightest rule the current
  tools pass; the gap to the "package-root public exports only" ideal
  is a docs-phase item, not a rewrite here.
- R4: no module under ``src/`` imports ``dataflow_sim`` at module top
  level except under ``dataflow_training.lowering`` (tools and tests
  are exempt by scope). Lazy in-function imports are allowed — that is
  how ``dataflow.core.convert`` keeps the simulator optional.

Static-scan limitation, accepted: ``from dataflow.service import X``
is judged by the module (``dataflow.service``), so pulling a
non-exported submodule through an allowed package would pass; the
allowed packages re-export their public surface, so the rule tracks
the real contract.

Tests:
- test_core_is_dependency_free: importing dataflow.core in a fresh interpreter drags in none of torch/jax/cuda/dataflow_sim.
- test_runtime_never_imports_torch_or_sim: importing dataflow.runtime drags in none of torch/jax/dataflow_sim.
- test_blocks_never_imports_sim: importing dataflow_training.blocks does not drag in dataflow_sim.
- test_r1_engine_never_imports_workload_or_twins: no module under src/dataflow imports dataflow_training or reference_models.
- test_r2_workload_uses_engine_public_surfaces_only: modules under src/dataflow_training reach dataflow only through the sanctioned core/runtime/service surfaces.
- test_r3_tools_stay_near_package_roots: files under tools/ import only the engine's three subtrees and dataflow_training at most two levels deep (deeper only under model_families/blocks).
- test_r4_sim_required_only_under_lowering: a module-top-level dataflow_sim import appears only under dataflow_training.lowering; elsewhere in src/ it must be a lazy in-function import.
"""
import ast
import subprocess
import sys
from pathlib import Path
from dataflow_training.distributed.topology import repo_root

REPO = repo_root()
SRC = REPO / "src"
TOOLS = REPO / "tools"

FORBIDDEN_AFTER_IMPORT = {
    "dataflow.core": ("torch", "jax", "cuda", "dataflow_sim"),
    "dataflow.runtime": ("torch", "jax", "dataflow_sim"),
    "dataflow_training.blocks": ("dataflow_sim",),
}


def _check(module: str, forbidden: tuple[str, ...]) -> None:
    code = (
        "import sys\n"
        f"import {module}\n"
        f"bad = [m for m in sys.modules if m.split('.')[0] in {forbidden!r}]\n"
        "assert not bad, f'forbidden modules imported: {bad}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"{module}: {result.stderr}"


def test_core_is_dependency_free():
    _check("dataflow.core", ("torch", "jax", "cuda", "dataflow_sim"))


def test_runtime_never_imports_torch_or_sim():
    _check("dataflow.runtime", ("torch", "jax", "dataflow_sim"))


def test_blocks_never_imports_sim():
    _check("dataflow_training.blocks", ("dataflow_sim",))


# ---------------------------------------------------------------- static scan

def python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def import_sites(path: Path) -> list[tuple[int, str, tuple[str, ...], bool]]:
    """Every absolute import in ``path`` as (lineno, module, names, top_level).

    ``names`` is non-empty only for ``from module import names``.
    ``top_level`` is False for imports nested inside a function body
    (the sanctioned lazy-import shape). Relative imports are skipped:
    they cannot leave their own package.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    sites = []
    stack = [(tree, True)]
    while stack:
        node, top_level = stack.pop()
        for child in ast.iter_child_nodes(node):
            child_top = top_level and not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(child, ast.Import):
                for alias in child.names:
                    sites.append((child.lineno, alias.name, (), top_level))
            elif isinstance(child, ast.ImportFrom):
                if child.level == 0 and child.module:
                    names = tuple(alias.name for alias in child.names)
                    sites.append((child.lineno, child.module, names, top_level))
            else:
                stack.append((child, child_top))
    return sites


def expand_refs(module: str, names: tuple[str, ...]) -> list[str]:
    """Dotted targets an import statement reaches.

    ``from dataflow import core`` must be judged as ``dataflow.core``,
    so imports FROM a bare package root expand per name; everything
    else is judged by the module itself.
    """
    if module in ("dataflow", "dataflow_training", "reference_models") and names:
        return [module if name == "*" else f"{module}.{name}" for name in names]
    return [module]


def inside(ref: str, prefixes: tuple[str, ...]) -> bool:
    return any(ref == p or ref.startswith(p + ".") for p in prefixes)


def scan_refs(root: Path) -> list[tuple[Path, int, str, bool]]:
    refs = []
    for path in python_files(root):
        for lineno, module, names, top_level in import_sites(path):
            for ref in expand_refs(module, names):
                refs.append((path, lineno, ref, top_level))
    return refs


def offender_lines(offenders: list[tuple[Path, int, str]]) -> str:
    return "\n".join(f"  {path.relative_to(REPO)}:{lineno}: {ref}"
                     for path, lineno, ref in offenders)


def test_r1_engine_never_imports_workload_or_twins():
    """R1: src/dataflow is blind to dataflow_training and reference_models."""
    offenders = [(path, lineno, ref)
                 for path, lineno, ref, top_level in scan_refs(SRC / "dataflow")
                 if inside(ref, ("dataflow_training", "reference_models"))]
    if offenders:
        print("R1 offenders:\n" + offender_lines(offenders))
    assert not offenders, f"engine imports workload/truth tree:\n{offender_lines(offenders)}"


R2_ALLOWED_SUBTREES = (
    "dataflow.core",
    "dataflow.runtime",
    "dataflow.service.client",
    "dataflow.service.registry",
    "dataflow.service.wire",
)
R2_ALLOWED_EXACT = ("dataflow.service",)


def test_r2_workload_uses_engine_public_surfaces_only():
    """R2: dataflow_training touches dataflow only via core/runtime ABIs
    and the sanctioned service surfaces."""
    offenders = []
    seen = set()
    for path, lineno, ref, top_level in scan_refs(SRC / "dataflow_training"):
        if ref.split(".")[0] != "dataflow":
            continue
        seen.add(ref)
        if inside(ref, R2_ALLOWED_SUBTREES) or ref in R2_ALLOWED_EXACT:
            continue
        offenders.append((path, lineno, ref))
    print("R2 imported dataflow modules:", sorted(seen))
    if offenders:
        print("R2 offenders:\n" + offender_lines(offenders))
    assert not offenders, f"workload imports off-surface engine modules:\n{offender_lines(offenders)}"


R3_DATAFLOW_EXACT = ("dataflow",)
R3_DATAFLOW_SUBTREES = ("dataflow.core", "dataflow.runtime", "dataflow.service")
R3_TRAINING_DEEP_SUBTREES = (
    "dataflow_training.model_families",
    "dataflow_training.blocks",
)
R3_TRAINING_MAX_PARTS = 3  # dataflow_training.x.y


def r3_allows(ref: str) -> bool:
    root = ref.split(".")[0]
    if root == "dataflow":
        return ref in R3_DATAFLOW_EXACT or inside(ref, R3_DATAFLOW_SUBTREES)
    if root == "dataflow_training":
        if len(ref.split(".")) <= R3_TRAINING_MAX_PARTS:
            return True
        return inside(ref, R3_TRAINING_DEEP_SUBTREES)
    return True  # other roots (torch, reference_models, dataflow_sim, ...) are out of scope


def test_r3_tools_stay_near_package_roots():
    """R3: tools import the engine's three subtrees and dataflow_training
    at most two levels deep (deeper only under model_families/blocks)."""
    offenders = [(path, lineno, ref)
                 for path, lineno, ref, top_level in scan_refs(TOOLS)
                 if not r3_allows(ref)]
    if offenders:
        print("R3 offenders:\n" + offender_lines(offenders))
    assert not offenders, f"tools import off-surface modules:\n{offender_lines(offenders)}"


R4_SIM_TOPLEVEL_ALLOWED = SRC / "dataflow_training" / "lowering"
R4_SIM_PACKAGE = SRC / "dataflow_sim"


def test_r4_sim_required_only_under_lowering():
    """R4: a module-top-level (required) dataflow_sim import may exist
    only under dataflow_training.lowering; everywhere else in the CONSUMER
    packages the simulator must stay a lazy in-function import. The
    dataflow_sim package itself is exempt — it IS the simulator, so its
    own internal imports are not a cross-package dependency on it."""
    offenders = []
    for path, lineno, ref, top_level in scan_refs(SRC):
        if ref.split(".")[0] != "dataflow_sim" or not top_level:
            continue
        if R4_SIM_PACKAGE in path.parents:
            continue  # the simulator's own internal imports are fine
        if R4_SIM_TOPLEVEL_ALLOWED in path.parents:
            continue
        offenders.append((path, lineno, ref))
    if offenders:
        print("R4 offenders:\n" + offender_lines(offenders))
    assert not offenders, f"required dataflow_sim import outside lowering:\n{offender_lines(offenders)}"
