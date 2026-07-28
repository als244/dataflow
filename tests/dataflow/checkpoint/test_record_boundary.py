"""The record layer is workload-blind by contract: dataflow.checkpoint
imports nothing from any workload package, ever — the boundary that
keeps records general while training policy compiles slices above it.

Tests:
- test_checkpoint_package_is_workload_blind: no module under src/dataflow/checkpoint imports dataflow_training (AST-walked, dotted forms included).
"""
import ast
import pathlib


def repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    while not (here / "src" / "dataflow").is_dir():
        here = here.parent
    return here


def test_checkpoint_package_is_workload_blind():
    pkg = repo_root() / "src" / "dataflow" / "checkpoint"
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if (name == "dataflow_training"
                        or name.startswith("dataflow_training.")):
                    offenders.append(f"{path.name}: imports {name}")
    assert not offenders, offenders
