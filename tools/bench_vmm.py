"""CUDA VMM microbench: the go/no-go numbers for the VMM slab design.

Measures, on the actual device:
  1. allocation granularity;
  2. cuMemCreate / cuMemRelease cost vs physical size (handles are pooled in
     the design, so this is init-time cost — but it bounds pool growth);
  3. map+setAccess and unmap cost vs mapped size (ONE call over a large
     range vs the per-2MB-chunk alternative);
  4. fragmented mapping: one object's VA backed by K physical pieces —
     cost vs K (fragmentation in the design costs map calls, never fails);
  5. correctness: remap the same VA to different physical backing between
     writes (the per-object stable-VA lifecycle), verify isolation;
  6. interference: map/unmap churn on idle VA while a bandwidth-bound
     kernel runs — measures TLB/driver interference on compute.

Design context: the vmm-slab-design-v1 design note.
"""
from __future__ import annotations

import statistics
import time

import torch

from cuda.bindings import driver as cu

MB = 1024**2
GB = 1024**3


def check(ret):
    err = ret[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"driver error: {err}")
    rest = ret[1:]
    return rest[0] if len(rest) == 1 else rest


def timeit(fn, n=30, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), min(ts), max(ts)


def main() -> None:
    torch.cuda.init()
    torch.zeros(1, device="cuda")  # materialize the primary context
    dev = check(cu.cuCtxGetDevice())
    print(f"device: {torch.cuda.get_device_name(0)} (CUdevice {int(dev)})")

    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = int(dev)

    gran = check(cu.cuMemGetAllocationGranularity(
        prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    gran_rec = check(cu.cuMemGetAllocationGranularity(
        prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
    print(f"granularity: minimum {gran/MB:.0f} MiB, recommended {gran_rec/MB:.0f} MiB")

    access = cu.CUmemAccessDesc()
    access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = int(dev)
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    # one big VA arena, like the design's per-plan reservation
    ARENA = 64 * GB
    va = check(cu.cuMemAddressReserve(ARENA, 0, 0, 0))
    print(f"VA arena: reserved {ARENA/GB:.0f} GiB at 0x{int(va):x}")

    def align(n: int) -> int:
        return (n + gran - 1) // gran * gran

    # ---- 2) cuMemCreate / cuMemRelease vs size ----
    print("\ncuMemCreate + cuMemRelease (pooled at init in the design):")
    for size in (2 * MB, 64 * MB, 416 * MB, 1 * GB):
        size = align(size)

        def create_release(size=size):
            h = check(cu.cuMemCreate(size, prop, 0))
            check(cu.cuMemRelease(h))

        med, lo, hi = timeit(create_release, n=10)
        print(f"  {size/MB:7.0f} MiB: {med:9.1f} us  (min {lo:.1f}, max {hi:.1f})")

    # ---- 3) map + setAccess + unmap vs size (ONE call per range) ----
    print("\ncuMemMap+SetAccess / cuMemUnmap over ONE range (steady-state op):")
    results = {}
    for size in (2 * MB, 64 * MB, 416 * MB, 1 * GB):
        size = align(size)
        h = check(cu.cuMemCreate(size, prop, 0))

        def map_unmap(size=size, h=h):
            check(cu.cuMemMap(int(va), size, 0, h, 0))
            check(cu.cuMemSetAccess(int(va), size, [access], 1))
            check(cu.cuMemUnmap(int(va), size))

        med, lo, hi = timeit(map_unmap, n=20)
        results[size] = med
        print(f"  {size/MB:7.0f} MiB: {med:9.1f} us  (min {lo:.1f}, max {hi:.1f})")
        check(cu.cuMemRelease(h))

    # ---- 4) fragmented map: 416 MiB as K pieces ----
    print("\nfragmented map of 416 MiB (design: fragmentation = more map calls):")
    total = align(416 * MB)
    for k in (1, 2, 4, 16, 64):
        piece = align(total // k)
        handles = [check(cu.cuMemCreate(piece, prop, 0)) for _ in range(k)]

        def map_frag(handles=handles, piece=piece):
            off = 0
            for h in handles:
                check(cu.cuMemMap(int(va) + off, piece, 0, h, 0))
                off += piece
            check(cu.cuMemSetAccess(int(va), piece * len(handles), [access], 1))
            off = 0
            for _h in handles:
                check(cu.cuMemUnmap(int(va) + off, piece))
                off += piece

        med, lo, hi = timeit(map_frag, n=15)
        print(f"  K={k:3d} pieces of {piece/MB:6.0f} MiB: {med:9.1f} us total")
        for h in handles:
            check(cu.cuMemRelease(h))

    # ---- 5) correctness: stable VA, swapped physical backing ----
    print("\ncorrectness: remap same VA to different physical between writes")
    size = align(64 * MB)
    h1 = check(cu.cuMemCreate(size, prop, 0))
    h2 = check(cu.cuMemCreate(size, prop, 0))
    check(cu.cuMemMap(int(va), size, 0, h1, 0))
    check(cu.cuMemSetAccess(int(va), size, [access], 1))
    check(cu.cuMemsetD8(int(va), 0xAB, size))
    check(cu.cuCtxSynchronize())
    check(cu.cuMemUnmap(int(va), size))
    check(cu.cuMemMap(int(va), size, 0, h2, 0))
    check(cu.cuMemSetAccess(int(va), size, [access], 1))
    check(cu.cuMemsetD8(int(va), 0xCD, size))
    check(cu.cuCtxSynchronize())
    buf = bytearray(16)
    check(cu.cuMemcpyDtoH(buf, int(va), 16))
    assert all(b == 0xCD for b in buf), buf.hex()
    # remap h1: original bytes must have survived on their physical pages
    check(cu.cuMemUnmap(int(va), size))
    check(cu.cuMemMap(int(va), size, 0, h1, 0))
    check(cu.cuMemSetAccess(int(va), size, [access], 1))
    check(cu.cuMemcpyDtoH(buf, int(va), 16))
    assert all(b == 0xAB for b in buf), buf.hex()
    check(cu.cuMemUnmap(int(va), size))
    check(cu.cuMemRelease(h1))
    check(cu.cuMemRelease(h2))
    print("  swapped backing isolated; bytes persist per physical handle  OK")

    # ---- 6) interference with running compute ----
    print("\ninterference: bandwidth-bound kernel while map/unmap churns elsewhere")
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)

    def bw_pass():
        for _ in range(50):
            y.copy_(x)
        torch.cuda.synchronize()

    bw_pass()
    t0 = time.perf_counter()
    bw_pass()
    quiet = time.perf_counter() - t0

    size = align(64 * MB)
    h = check(cu.cuMemCreate(size, prop, 0))
    stop = False
    churn_count = 0

    import threading

    def churn():
        nonlocal churn_count
        # own the same context on this thread
        ctx = check(cu.cuCtxGetCurrent())
        if int(ctx) == 0:
            check(cu.cuCtxSetCurrent(main_ctx))
        while not stop:
            check(cu.cuMemMap(int(va), size, 0, h, 0))
            check(cu.cuMemSetAccess(int(va), size, [access], 1))
            check(cu.cuMemUnmap(int(va), size))
            churn_count += 1

    main_ctx = check(cu.cuCtxGetCurrent())
    th = threading.Thread(target=churn)
    th.start()
    t0 = time.perf_counter()
    bw_pass()
    loud = time.perf_counter() - t0
    stop = True
    th.join()
    check(cu.cuMemRelease(h))
    print(f"  copy pass quiet: {quiet*1e3:8.2f} ms")
    print(f"  copy pass churn: {loud*1e3:8.2f} ms  ({(loud/quiet-1)*100:+.1f}%)  "
          f"[{churn_count} map/unmap cycles concurrent]")

    check(cu.cuMemAddressFree(va, ARENA))
    print("\ndone.")


if __name__ == "__main__":
    main()
