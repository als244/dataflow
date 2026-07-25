"""Service-owned host memory: pinned slab, capacity heuristics, views.

Pinning itself lives in the runtime's device layer (pin_region /
unpin_region) and is called from here rather than repeated: this module
carried its own copy, so when the copy in the device layer stopped rounding
allocations up to a power of two, the slab every daemon actually boots with
went on paying for it. What belongs here is the policy — how big a slab is
safe to ask for on this host — not how a page gets pinned.

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
    """One mapped-and-registered region; freed explicitly (daemon shutdown)."""

    # pinned memory is UNRECLAIMABLE and UNSWAPPABLE: exhausting the
    # host with it starves the page cache and the GPU driver long
    # before the kernel OOM killer fires (observed: full desktop
    # freeze + NVRM page-table failure + reboot). Never pin into the
    # last SYSTEM_RESERVE_GIB of available memory.
    SYSTEM_RESERVE_GIB = 24.0

    def __init__(self, capacity_bytes: int, *, device: int = 0):
        import threading

        from cuda.bindings import runtime as cudart

        from ..runtime.device.cuda import pin_region

        avail = meminfo_available_bytes()
        limit = avail - int(self.SYSTEM_RESERVE_GIB * GIB)
        if capacity_bytes > limit:
            raise RuntimeError(
                f"refusing to pin {capacity_bytes / GIB:.1f} GiB: only "
                f"{avail / GIB:.1f} GiB available and "
                f"{self.SYSTEM_RESERVE_GIB:.0f} GiB is reserved for the "
                f"system (pinned memory cannot be reclaimed or swapped)")

        self._cudart = cudart
        self._free_lock = threading.Lock()
        _check(cudart.cudaSetDevice(device))
        self.ptr = pin_region(capacity_bytes)
        self.capacity = capacity_bytes

    def free(self) -> None:
        # idempotent + thread-safe: shutdown's serve_forever finally
        # and external owners (test fixtures) may both call this — the
        # bare `if self.ptr` guard was a TOCTOU double-free (CUDA
        # error 1 in a daemon thread, found by the cancel gate's
        # teardown)
        from ..runtime.device.cuda import unpin_region

        with self._free_lock:
            ptr, self.ptr = self.ptr, 0
        if ptr:
            unpin_region(ptr, self.capacity)


def meminfo_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def auto_cap_bytes(reserve_gib: float = 30.0) -> int:
    """STORE-ONLY sizing: leaves reserve for the system but NOT for
    session pools — daemons that execute programs must size the slab
    explicitly until pools draw from the slab (design note Part V
    addendum 2). Reserve default raised 10 -> 30 after the OOM
    incident: pinned memory starves the desktop long before the
    kernel killer acts."""
    return max(GIB, meminfo_available_bytes() - int(reserve_gib * GIB))


def bytes_view(ptr: int, size: int) -> memoryview:
    import ctypes

    return memoryview((ctypes.c_char * size).from_address(ptr)).cast("B")
