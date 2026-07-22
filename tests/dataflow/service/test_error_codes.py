"""The ServiceError code registry is complete: every error code raised
anywhere in the source is registered in wire.ERROR_CODES.

ServiceError.__init__ asserts its code is registered, so an unregistered
code does not surface as itself — it becomes an AssertionError that the
daemon reports as a generic INTERNAL, masking the real condition (a peer
going unreachable, a group already existing, ...). This gate keeps the
registry ahead of every raise site, so error reporting stays honest.

Tests:
- test_every_raised_error_code_is_registered: no ServiceError code literal in the source is absent from wire.ERROR_CODES.
"""
import ast
from pathlib import Path

from dataflow.service.wire import ERROR_CODES
from dataflow_training.distributed.topology import repo_root


def service_error_codes(path):
    """Every string-literal code passed as the first argument of a
    ServiceError(...) call in one source file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    codes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "ServiceError" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            codes.append(first.value)
    return codes


def test_every_raised_error_code_is_registered():
    src = repo_root() / "src"
    unregistered: dict[str, list[str]] = {}
    for path in sorted(src.rglob("*.py")):
        for code in service_error_codes(path):
            if code not in ERROR_CODES:
                unregistered.setdefault(code, []).append(
                    path.relative_to(src).as_posix())
    assert not unregistered, (
        "ServiceError codes raised in source but missing from "
        "wire.ERROR_CODES — these AssertionError into a generic INTERNAL "
        f"when raised:\n  {unregistered}")
