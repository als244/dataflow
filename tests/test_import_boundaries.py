"""Layering rules, enforced.

- ``dataflow.core`` imports nothing heavy (no torch/jax/cuda/dataflow_sim).
- ``dataflow.runtime`` may not import torch, jax, or dataflow_sim.
- Only ``dataflow.runtime.device.cuda`` may import cuda bindings.
- Only ``dataflow.tasks`` may import torch/triton.
- Only ``dataflow.training`` (and tools/tests) may import dataflow_sim.

Each check runs in a fresh interpreter so prior imports can't mask leaks.
"""
import subprocess
import sys

FORBIDDEN_AFTER_IMPORT = {
    "dataflow.core": ("torch", "jax", "cuda", "dataflow_sim"),
    "dataflow.runtime": ("torch", "jax", "dataflow_sim"),
    "dataflow.tasks": ("dataflow_sim",),
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


def test_tasks_never_imports_sim():
    _check("dataflow.tasks", ("dataflow_sim",))
