"""Repo-root conftest: put the repo root on ``sys.path`` so the top-level
``reference_models/`` package (isolated ground-truth models, deliberately outside
the installed ``src/`` tree) is importable in tests without a pip reinstall.
"""
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# The suite validates correctness and lifecycle behavior; it never publishes
# task prices.  Production profiling deliberately samples every signature for
# 2.5 seconds of sustained load, which turns an invalidated profile cache into
# roughly nine minutes of unrelated test work.  Pin the test process to the
# burst-profile path before any dataflow_training module can read the setting.
# Production processes retain profiling.py's 2.5-second default.
os.environ["DATAFLOW_SAMPLE_FLOOR_S"] = "0"
os.environ["DATAFLOW_PROFILE_SOAK_S"] = "0"
os.environ["DATAFLOW_PROFILE_CONTEND_PCIE"] = "0"
os.environ["DATAFLOW_PROFILE_REPEATS"] = "1"


import pytest  # noqa: E402

_SESSION_INTERRUPTED = False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Explain signal-timeout tracebacks without hiding their useful frame."""
    outcome = yield
    report = outcome.get_result()
    if call.when == "call":
        item._dataflow_call_failed = report.failed
    if call.when != "call" or call.excinfo is None:
        return
    if not str(call.excinfo.value).startswith("Timeout (>"):
        return
    report.sections.append((
        "timeout interpretation",
        "The test body exceeded its timeout. The traceback above identifies "
        "the frame interrupted by SIGALRM; it is not a secondary cleanup "
        "failure. The timeout timer is now canceled and fixture teardown runs "
        "afterward. Any teardown failure is reported separately.",
    ))


def pytest_keyboard_interrupt(excinfo):
    """Let teardown clean resources without masking the interruption."""
    global _SESSION_INTERRUPTED
    _SESSION_INTERRUPTED = True


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
    # Even Python GC can finalize CUDA views while a server thread is
    # concurrently closing its pools. Defer all allocator/GC hygiene until
    # the server-owning fixture has joined that thread.
    if in_process_server_live():
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
    # must not have buffers freed under it. The prefix list encodes
    # where servers are EXPECTED — the live-thread probe below covers
    # where they actually ARE: a rig inside a drained prefix whose
    # server thread is still tearing down (2026-07-28: both boxes'
    # intermittent suite segfault — the drain raced serve_forever's
    # close_all_sessions over the same pool, concurrent frees, heap
    # corruption surfacing anywhere from view invalidation to freed
    # code objects).
    if nodeid.startswith(("tests/dataflow_training/training",
                          "tests/dataflow_training/models",
                          "tests/dataflow/runtime",
                          "tests/dataflow_training/pretrain")) \
            and not in_process_server_live():
        try:
            # raw cudaMalloc slabs leaked by tests that skip
            # close()/free() are invisible to empty_cache — drain them
            from dataflow.runtime.device.cuda import drain_all_backends
            drain_all_backends()
        except Exception:
            pass
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def background_test_hygiene(request, cuda_test_hygiene):
    """Stop/reject test-owned workers before CUDA teardown begins.

    Depending on ``cuda_test_hygiene`` makes this teardown run first: no
    CUDA GC/drain may begin while test-owned background work is alive.
    """
    yield
    feed_mod = sys.modules.get("dataflow_training.data.feed")
    worker_type = getattr(feed_mod, "IngestWorker", ()) if feed_mod else ()
    leaked = [thread for thread in __import__("threading").enumerate()
              if worker_type and isinstance(thread, worker_type)]
    for worker in leaked:
        worker.stop()
    if leaked:
        message = (f"test leaked {len(leaked)} datafeed ingest worker(s); "
                   "cleanup stopped them")
        if _SESSION_INTERRUPTED or getattr(
                request.node, "_dataflow_call_failed", False):
            request.node.add_report_section("teardown", "cleanup", message)
        else:
            pytest.fail(message + "; close the Packer/DataFeed in a context "
                        "manager or finally")
    nodeid = request.node.nodeid
    if not nodeid.startswith(("tests/dataflow/service", "tests/fleet")) \
            and in_process_server_live():
        pytest.fail(
            "test leaked an in-process engine server; its owner must "
            "request shutdown and join the server thread in finally")


def in_process_server_live() -> bool:
    """True while any thread of THIS process is inside serve_forever
    (cleanup included) — draining backends under it corrupts the heap."""
    server_mod = sys.modules.get("dataflow.service.server")
    if server_mod is None:
        return False
    try:
        return server_mod.live_server_count() > 0
    except Exception:
        return False


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
    if in_process_server_live():
        pytest.fail(
            "module leaked an in-process engine server; refusing CUDA "
            "cleanup because it would race the live server thread")
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
