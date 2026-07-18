"""Repo-root conftest: put the repo root on ``sys.path`` so the top-level
``reference_models/`` package (isolated ground-truth models, deliberately outside
the installed ``src/`` tree) is importable in tests without a pip reinstall.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


import pytest


@pytest.fixture(autouse=True)
def cuda_test_hygiene(request):
    """Device-memory hygiene between tests: engine slabs are freed by the
    tests themselves, but the interop view cache, the torch caching
    allocator, AND raw-cudaMalloc leaks from tests that skip
    close()/free() accumulate across a long suite until a 24GB card hits
    cudaErrorMemoryAllocation mid-run. Clearing all three after every
    test keeps the FULL suite runnable in one process (no chunking).

    The backend drain is SKIPPED for suites that keep an in-process
    THREADED EngineServer alive across tests (fleet loopback rigs,
    service suites): draining under a live server's engine frees
    buffers its next run still uses — measured as a segfault in
    memcpy_async, not a theory."""
    yield
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return
    try:
        from dataflow.tasks.interop import clear_view_cache
        clear_view_cache()
    except Exception:
        pass
    import gc

    gc.collect()
    torch.cuda.synchronize()
    nodeid = request.node.nodeid
    # ALLOWLIST: only engine-direct suites drain (they own their
    # backends); anything that may keep an in-process threaded server
    # alive across tests (fleet rigs, service suites, example bridges)
    # must not have buffers freed under it
    if nodeid.startswith(("tests/dataflow_training/training",
                          "tests/dataflow_training/models",
                          "tests/dataflow/runtime",
                          "tests/dataflow_training/pretrain")):
        try:
            # raw cudaMalloc slabs leaked by tests that skip
            # close()/free() are invisible to empty_cache — drain them
            from dataflow.runtime.device.cuda import drain_all_backends
            drain_all_backends()
        except Exception:
            pass
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True, scope="module")
def cuda_module_hygiene(request):
    """Server-hosting suites (fleet rigs, service, examples) are
    excluded from the per-test drain — freeing under a live in-process
    engine server segfaults. Their leaked slabs still starve LATER
    suites (the mirrored test order runs examples after service, whose
    held VRAM broke the RL subprocess daemons). By MODULE end every
    rig/server is torn down, so draining here is safe and returns the
    memory."""
    yield
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return
    nodeid = getattr(request.node, "nodeid", "")
    if not nodeid.startswith(("tests/dataflow/service", "tests/fleet",
                              "tests/examples")):
        return
    try:
        from dataflow.runtime.device.cuda import drain_all_backends

        torch.cuda.synchronize()
        drain_all_backends()
        torch.cuda.empty_cache()
    except Exception:
        pass


def pytest_collection_modifyitems(config, items):
    """Resource scheduling, not preference: the RL example tests boot
    4-8 GiB subprocess daemons and must run while the GPU is still
    empty. The mirrored tree ordered them AFTER the service/workload
    suites, whose in-process engines leave the parent holding enough
    VRAM (even post-drain) to starve those subprocess boots — measured
    as cudaErrorMemoryAllocation only in full-battery order. Examples
    therefore collect first."""
    front = [it for it in items if it.nodeid.startswith("tests/examples")]
    rest = [it for it in items if not it.nodeid.startswith("tests/examples")]
    items[:] = front + rest
