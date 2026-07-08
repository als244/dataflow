"""Service-owned host memory: pinned slab, capacity heuristics, views.

Self-contained on purpose (Shein): the service package does not import
the runtime's device layer. The two facts it needs are small and
copied here:
- cudaHostAlloc is page-granular and pins at ~5 GiB/s on this class of
  box regardless of chunking (measured; design note) — one boot-time
  slab, suballocated by the store;
- the safe backing budget is host MemAvailable minus a leeway.

Fake mode never touches this module.
"""
from __future__ import annotations

GIB = 1024**3


def _check(res):
    err, *rest = res
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {int(err)}")
    return rest


class PinnedSlab:
    """One cudaHostAlloc'd region; freed explicitly (daemon shutdown)."""

    def __init__(self, capacity_bytes: int, *, device: int = 0):
        import threading

        from cuda.bindings import runtime as cudart

        self._cudart = cudart
        self._free_lock = threading.Lock()
        _check(cudart.cudaSetDevice(device))
        (ptr,) = _check(cudart.cudaHostAlloc(
            capacity_bytes, cudart.cudaHostAllocDefault))
        self.ptr = int(ptr)
        self.capacity = capacity_bytes

    def free(self) -> None:
        # idempotent + thread-safe: shutdown's serve_forever finally
        # and external owners (test fixtures) may both call this — the
        # bare `if self.ptr` guard was a TOCTOU double-free (CUDA
        # error 1 in a daemon thread, found by the cancel gate's
        # teardown)
        with self._free_lock:
            ptr, self.ptr = self.ptr, 0
        if ptr:
            _check(self._cudart.cudaFreeHost(ptr))


def meminfo_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def auto_cap_bytes(reserve_gib: float = 10.0) -> int:
    return max(GIB, meminfo_available_bytes() - int(reserve_gib * GIB))


def bytes_view(ptr: int, size: int) -> memoryview:
    import ctypes

    return memoryview((ctypes.c_char * size).from_address(ptr)).cast("B")
