"""Meta-gate: every test file carries a docstring index of its tests, and
that index stays in sync with the file's actual test functions.

Each test module's docstring ends with a ``Tests:`` block — one
``- test_name: one-line summary`` entry per test function. This gate
parses that block and compares it to the real ``def test_*`` functions,
so an added, removed, or renamed test that skips its index entry fails
loudly instead of letting the index rot.

Tests:
- test_every_file_documents_its_tests: every test module's docstring carries a Tests: block.
- test_index_matches_test_functions: each Tests: block lists exactly the file's test_* functions, else the drift is reported.
"""
import ast
import re

from dataflow_training.distributed.topology import repo_root

TESTS_ROOT = repo_root() / "tests"
TEST_FILES = sorted(TESTS_ROOT.rglob("test_*.py"))
INDEX_ENTRY = re.compile(r"^\s*-\s*(test_[A-Za-z0-9_]+)", re.MULTILINE)


def module_test_names(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    return [node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]


def documented_tests(path):
    """The test names listed in the module docstring's Tests: block, or
    None when the file has no such block."""
    tree = ast.parse(path.read_text(), filename=str(path))
    doc = ast.get_docstring(tree) or ""
    if "Tests:" not in doc:
        return None
    return INDEX_ENTRY.findall(doc.split("Tests:", 1)[1])


def relpath(path):
    return path.relative_to(TESTS_ROOT).as_posix()


def test_every_file_documents_its_tests():
    missing = [relpath(p) for p in TEST_FILES if documented_tests(p) is None]
    assert not missing, ("test files with no 'Tests:' docstring block:\n  "
                         + "\n  ".join(missing))


def test_index_matches_test_functions():
    drift = []
    for path in TEST_FILES:
        documented = documented_tests(path)
        if documented is None:
            continue
        actual = module_test_names(path)
        if set(documented) != set(actual):
            undocumented = sorted(set(actual) - set(documented))
            phantom = sorted(set(documented) - set(actual))
            drift.append(f"{relpath(path)}: undocumented={undocumented} "
                         f"phantom={phantom}")
    assert not drift, "docstring index out of sync with test functions:\n  " + \
        "\n  ".join(drift)
