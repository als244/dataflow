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
    if nodeid.startswith(("tests/training", "tests/models",
                          "tests/runtime", "tests/pretrain")):
        try:
            # raw cudaMalloc slabs leaked by tests that skip
            # close()/free() are invisible to empty_cache — drain them
            from dataflow.runtime.device.cuda import drain_all_backends
            drain_all_backends()
        except Exception:
            pass
    torch.cuda.empty_cache()
