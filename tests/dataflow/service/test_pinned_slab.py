"""The backing slab costs the bytes it asks for, and keeps costing them.

Tests:
- test_slab_costs_what_it_asks_for: a pinned slab consumes close to its requested size rather than the next power of two above it.
- test_slab_frees_what_it_pinned: freeing a slab returns the host memory, so a daemon restart does not leak it.
- test_host_memory_is_pinned_in_exactly_one_place: only the runtime device layer pins host memory, so a copy cannot be fixed in one place and left wrong in the other.
"""
import ast
import pathlib

import pytest

torch = pytest.importorskip("torch")

from dataflow.service.hostmem import PinnedSlab, meminfo_available_bytes

GIB = 1024 ** 3


def repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    while not (here / "src" / "dataflow").is_dir():
        here = here.parent
    return here


@pytest.mark.gpu
def test_slab_costs_what_it_asks_for():
    """cudaHostAlloc rounds up to a power of two, so a 3 GiB slab cost 4 GiB
    and a 65 GiB slab tried to take 128 -- which on a host with 113 GiB free
    is not waste but a refusal. The size asked for is the size paid for."""
    want = 3 * GIB                      # deliberately not a power of two
    before = meminfo_available_bytes()
    slab = PinnedSlab(want)
    try:
        consumed = before - meminfo_available_bytes()
        # allow generous slack for other activity on the box, but nowhere near
        # the 4 GiB the next power of two would have taken
        assert consumed < want * 1.25, (
            f"slab of {want / GIB:.1f} GiB consumed {consumed / GIB:.2f} GiB; "
            f"rounding up to a power of two would take {4 * GIB / GIB:.1f} GiB")
    finally:
        slab.free()


@pytest.mark.gpu
def test_slab_frees_what_it_pinned():
    want = 2 * GIB
    before = meminfo_available_bytes()
    PinnedSlab(want).free()
    after = meminfo_available_bytes()
    assert after > before - want // 2, (
        f"{(before - after) / GIB:.2f} GiB still held after free")


def test_host_memory_is_pinned_in_exactly_one_place():
    """Host memory used to be pinned in two modules with copied logic. The
    rounding fix landed in one of them, and the slab every daemon actually
    boots with went on paying for it -- the bug survived being fixed. So the
    gate is not "avoid cudaHostAlloc" but "there is one implementation":
    a second copy is free to be correct today and wrong after the next edit.

    The 4-byte mapped flag word in the peer transport is exempt: rounding is
    meaningless at that size and it needs device-mappable semantics."""
    root = repo_root()
    owner = root / "src" / "dataflow" / "runtime" / "device" / "cuda.py"
    exempt = {root / "src" / "dataflow" / "service" / "peer" / "comm.py"}

    offenders = []
    for path in (root / "src").rglob("*.py"):
        if path == owner or path in exempt:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in {"cudaHostAlloc", "cudaHostRegister", "cudaHostUnregister"}:
                offenders.append(f"{path.relative_to(root)}:{node.lineno} {name}")
    assert not offenders, (
        "host memory must be pinned through runtime.device.cuda.pin_region / "
        "unpin_region, which is the only place that gets to know how:\n  "
        + "\n  ".join(offenders))
