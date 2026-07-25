"""The backing slab costs the bytes it asks for, and keeps costing them.

Tests:
- test_slab_costs_what_it_asks_for: a pinned slab consumes close to its requested size rather than the next power of two above it.
- test_slab_frees_what_it_pinned: freeing a slab returns the host memory, so a daemon restart does not leak it.
- test_slab_paths_do_not_use_cudahostalloc: neither pinning path reaches for cudaHostAlloc, whose rounding is invisible until it becomes a ceiling.
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


def test_slab_paths_do_not_use_cudahostalloc():
    """Pinned memory is allocated in two places -- the service's slab and the
    runtime's device backend -- because the service does not import the
    runtime's device layer. Both must avoid cudaHostAlloc for the same reason,
    and a comment saying so is not enforcement.

    The 4-byte mapped flag word in the peer transport is exempt: rounding is
    meaningless at that size and it needs device-mappable semantics."""
    root = repo_root()
    guarded = [root / "src" / "dataflow" / "service" / "hostmem.py",
               root / "src" / "dataflow" / "runtime" / "device" / "cuda.py"]
    offenders = []
    for path in guarded:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "cudaHostAlloc":
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, (
        "cudaHostAlloc rounds the request up to a power of two, which is a "
        "hard ceiling on any host whose RAM is not just above one; use mmap "
        "+ cudaHostRegister:\n  " + "\n  ".join(offenders))
