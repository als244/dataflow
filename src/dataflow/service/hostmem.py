"""Service-owned host memory: pinned slab, capacity heuristics, views.

Self-contained on purpose: the service package does not import
the runtime's device layer. The two facts it needs are small and
copied here:
- pinning goes through mmap + cudaHostRegister, NOT cudaHostAlloc, which
  rounds the request up to a power of two — one boot-time slab,
  suballocated by the store;
- the safe backing budget is host MemAvailable minus a leeway.

The runtime's device layer pins the same way for the same reason; the two
implementations are separate because of the layering above, and a test
asserts neither drifts back to cudaHostAlloc.

Fake mode never touches this module.
"""
from __future__ import annotations

import ctypes

GIB = 1024**3

# Linux x86-64 ABI. mmap gives back exactly the pages asked for, where
# cudaHostAlloc rounds up to a power of two: measured on a 125.7 GiB host, a
# 33 GiB request consumed 65 GiB and anything above 64 GiB was refused
# outright with 113 GiB free, because 128 GiB does not fit. A slab must cost
# what it says, and on a host whose RAM is not near a power of two the
# rounding is the difference between fitting and not.
_PROT_READ, _PROT_WRITE = 0x1, 0x2
_MAP_PRIVATE, _MAP_ANONYMOUS = 0x02, 0x20
_MAP_FAILED = ctypes.c_void_p(-1).value
_LIBC = None


def libc():
    """ctypes libc with mmap/munmap typed for 64-bit pointers — without the
    restype the address is truncated to a C int, turning a valid high address
    into a negative number and a good mapping into a phantom failure."""
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
        _LIBC.mmap.restype = ctypes.c_void_p
        _LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_long]
        _LIBC.munmap.restype = ctypes.c_int
        _LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return _LIBC


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
        ptr = libc().mmap(None, capacity_bytes, _PROT_READ | _PROT_WRITE,
                          _MAP_PRIVATE | _MAP_ANONYMOUS, -1, 0)
        if not ptr or ptr == _MAP_FAILED:
            raise RuntimeError(
                f"could not map {capacity_bytes / GIB:.1f} GiB for the backing "
                f"slab ({avail / GIB:.1f} GiB available)")
        err = cudart.cudaHostRegister(ptr, capacity_bytes,
                                      cudart.cudaHostRegisterDefault)
        if isinstance(err, tuple):
            err = err[0]
        if int(err) != 0:
            libc().munmap(ctypes.c_void_p(ptr), capacity_bytes)
            raise RuntimeError(
                f"could not pin {capacity_bytes / GIB:.1f} GiB for the backing "
                f"slab: cudaHostRegister returned {int(err)} "
                f"({avail / GIB:.1f} GiB available)")
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
            # unregister BEFORE unmapping: handing the mapping back while the
            # driver still holds the pages pinned leaves it pinning memory
            # this process no longer owns
            _check(self._cudart.cudaHostUnregister(ptr))
            libc().munmap(ctypes.c_void_p(ptr), self.capacity)


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
