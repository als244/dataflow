"""VMM arena: non-contiguous physical backing behind per-object stable VAs.

The contiguous-slab design pays an address-geometry tax (packed extent >
ledger peak, measured x1.05-1.21) and admits placement-infeasible programs
whose byte load fits at every instant. This module removes the contiguity
constraint with the CUDA VMM API (design: the vmm-slab-design-v1 design note;
microbench: tools/bench_vmm.py — map+setAccess+unmap is 39-52 us per RANGE,
size-independent, and does not perturb concurrent compute):

- virtual addresses come from growable reservations and are recycled as
  (VA, handle) SLOTS: a released slot stays mapped ("parked") and any
  same-size birth ADOPTS it wholesale with zero driver calls — the
  steady-state path of a shape-stable program. Tags do not own VAs;
  determinism comes from the deterministic get/put sequence;
- physical memory is EXACT-SIZE cuMemCreate handles (a handle must be
  mapped whole: the driver rejects sub-range maps — discovered the hard
  way, CUDA_ERROR_NOT_SUPPORTED), free-listed by size and mapped with ONE
  call per get. Shape-stable programs make size demand periodic, so free
  lists stabilize after the first round and steady state does zero
  create/release work;
- the LEDGER budget is enforced by REFLOW: when a size class misses and
  created bytes would exceed the pool, cached (free) handles of other
  sizes are released until the new handle fits (~150 us per create,
  warmup-round only). Physical occupancy therefore tracks the ledger by
  construction: no packing problem, no extent tax, no
  placement-infeasible failure class.

Unmap ordering: cuMemUnmap is HOST-ordered — in-stream address-reuse
arguments do not apply. The engine's lifecycle makes put() safe (releases
apply in the task-done handler, offload puts in transfer-completion
handlers), with ONE exception: a guard event (debug poison memset queued at
release time). Guarded buffers defer (extents, event) to a reclaim list
drained lazily; if the same object is re-allocated while its old unmap is
deferred, it simply gets a FRESH virtual range (VA is free) rather than
blocking the engine. Slot adoption supersedes per-tag stable VAs: same-
size siblings dominate rebirth order in practice, so binding VAs to tags
made almost every reuse pay an unmap+remap (measured: 4,271 steals vs 48
hits on bs8ga8). Binding VAs to HANDLES makes the same reuse free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Buffer

GRANULE = 2 * 1024 * 1024  # confirmed == minimum granularity on RTX 5090


class VmmError(RuntimeError):
    pass


def _dcheck(ret):
    from cuda.bindings import driver as cu

    err = ret[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise VmmError(f"CUDA driver error: {err}")
    rest = ret[1:]
    return rest[0] if len(rest) == 1 else rest


def _align(n: int, a: int = GRANULE) -> int:
    return (n + a - 1) // a * a


@dataclass
class _Deferred:
    tag: object
    va: int
    span: int
    handle: object
    event: object


@dataclass
class VmmArena:
    """Fast-memory allocator: stable VAs, pooled physical, mapped on demand.

    ``event_complete`` is the backend's non-blocking event probe (guards).
    All sizes in bytes. Deterministic: VA assignment and extent carving are
    pure functions of the get/put sequence.
    """

    device_index: int
    capacity_bytes: int
    event_complete: object            # Callable[[Event], bool]
    headroom_bytes: int = 512 * 1024 * 1024
    # per-RESERVATION block; the arena grows by whole blocks on exhaustion
    # (VA is free — a 9B 2-round chain wants ~320 GiB of stable ranges)
    va_bytes: int = 512 * 1024**3

    # -- state --
    _va_base: int = 0
    _va_cursor: int = 0
    _va_blocks: list = field(default_factory=list)   # (base, size) reservations
    _va_block_end: int = 0
    _free: dict = field(default_factory=dict)            # span -> [handle] (unmapped, no VA)
    _parked_slots: dict = field(default_factory=dict)    # span -> [(va, handle)] still MAPPED
    _free_vas: dict = field(default_factory=dict)        # span -> [va] (unmapped ranges)
    _free_bytes: int = 0                                 # bytes cached in _free
    created_bytes: int = 0                               # all live+cached handles
    _live: dict = field(default_factory=dict)            # id(buffer) -> Buffer
    _deferred: list = field(default_factory=list)        # [_Deferred]
    _prop: object = None
    _access: object = None
    _primary_ctx: object = None
    _retained_device: object = None
    _seq: int = 0
    closed: bool = False

    # -- stats (reported by the engine) --
    maps: int = 0
    handle_creates: int = 0
    handle_reflows: int = 0           # releases forced by the budget
    prewarmed: int = 0                # handles pre-created off the hot path
    slot_adoptions: int = 0           # births served with ZERO driver calls
    # host seconds ON THE DISPATCH PATH, by driver-call class
    t_create_s: float = 0.0
    t_destroy_s: float = 0.0
    t_map_s: float = 0.0
    t_unmap_s: float = 0.0
    reclaim_drains: int = 0
    used_bytes: int = 0
    peak_used_bytes: int = 0
    peak_created_bytes: int = 0

    def __post_init__(self) -> None:
        import os

        from cuda.bindings import driver as cu

        # cache allowance vs bytes: the pool above the ledger budget is what
        # lets freed handles STAY CACHED at peak pressure instead of being
        # destroyed+recreated (a fresh handle pays ~1.9 ms/GiB of page
        # sanitization that contends with bandwidth-bound compute). Tunable
        # while the tradeoff is being characterized.
        env = os.environ.get("DATAFLOW_VMM_HEADROOM_GIB")
        if env is not None:
            self.headroom_bytes = int(float(env) * 1024**3)

        # the driver API needs a CURRENT context; the runtime API (CudaBackend)
        # creates the primary context lazily, so it may not exist yet. Retain
        # it explicitly — torch and cudart share the same primary context.
        _dcheck(cu.cuInit(0))
        ctx = _dcheck(cu.cuCtxGetCurrent())
        if int(ctx) == 0:
            dev = _dcheck(cu.cuDeviceGet(self.device_index))
            self._primary_ctx = _dcheck(cu.cuDevicePrimaryCtxRetain(dev))
            _dcheck(cu.cuCtxSetCurrent(self._primary_ctx))
            self._retained_device = dev

        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device_index
        self._prop = prop
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device_index
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._access = access

        self.pool_bytes = _align(self.capacity_bytes + self.headroom_bytes)
        self._va_grow()
        self._va_base = self._va_blocks[0][0]

        # warmup: the first map pays a one-off driver lazy-init (~4 ms
        # measured); absorb it here instead of on the first task
        h = self._new_handle(GRANULE)
        _dcheck(cu.cuMemMap(self._va_base, GRANULE, 0, h, 0))
        _dcheck(cu.cuMemSetAccess(self._va_base, GRANULE, [self._access], 1))
        _dcheck(cu.cuMemUnmap(self._va_base, GRANULE))
        self._release_handle(h, GRANULE)

    # ------------------------------------------------------- handle pool

    def _new_handle(self, span: int):
        import time

        from cuda.bindings import driver as cu

        t0 = time.perf_counter()
        h = _dcheck(cu.cuMemCreate(span, self._prop, 0))
        self.t_create_s += time.perf_counter() - t0
        self.created_bytes += span
        self.peak_created_bytes = max(self.peak_created_bytes, self.created_bytes)
        self.handle_creates += 1
        return h

    def _release_handle(self, handle, span: int) -> None:
        self._free.setdefault(span, []).append(handle)
        self._free_bytes += span

    def _park_slot(self, va: int, span: int, handle) -> None:
        self._parked_slots.setdefault(span, []).append((va, handle))
        self._free_bytes += span

    def _reclaim_slot(self, span: int):
        """Unmap the oldest parked slot of this span: naked handle + free VA."""
        slots = self._parked_slots.get(span)
        if not slots:
            return None
        va, handle = slots.pop(0)
        self._unmap(va, span)
        self._free_bytes -= span
        self._free_vas.setdefault(span, []).append(va)
        return handle

    def _evict_parked_any(self) -> tuple[int, object] | None:
        """Reclaim the SMALLEST parked slot (unmap; return naked handle)."""
        for sp in sorted(self._parked_slots):
            if self._parked_slots[sp]:
                return sp, self._reclaim_slot(sp)
        return None

    def _destroy_handle(self, span: int, handle) -> None:
        """Release a NAKED (unmapped, uncounted-in-_free) handle."""
        import time

        from cuda.bindings import driver as cu

        t0 = time.perf_counter()
        _dcheck(cu.cuMemRelease(handle))
        self.t_destroy_s += time.perf_counter() - t0
        self.created_bytes -= span

    def _destroy_cached(self, span: int, handle) -> None:
        self._free_bytes -= span
        self._destroy_handle(span, handle)

    def _obtain_handle(self, span: int):
        """Free-listed handle of exactly `span` bytes, creating under the
        budget; REFLOW (release cached handles of other sizes, largest
        first) when creation would exceed the pool."""
        cached = self._free.get(span)
        if cached:
            self._free_bytes -= span
            return cached.pop()
        if self.created_bytes + span > self.pool_bytes:
            self.drain_reclaim()
            cached = self._free.get(span)
            if cached:
                self._free_bytes -= span
                return cached.pop()
        while self.created_bytes + span > self.pool_bytes:
            victim_span = None
            for sp, lst in self._free.items():
                if lst and (victim_span is None or sp < victim_span):
                    victim_span = sp
            if victim_span is None:
                got = self._evict_parked_any()
                if got is None:
                    pending = sum(d.span for d in self._deferred)
                    raise VmmError(
                        f"vmm cannot back {span} bytes: pool {self.pool_bytes}, "
                        f"live {self.used_bytes}, cached 0, guard-deferred {pending} — "
                        f"the ledger admitted bytes the pool cannot back "
                        f"(headroom too small?)"
                    )
                vs, vh = got  # naked already (unpark unmapped + uncounted it)
                self._destroy_handle(vs, vh)
                self.handle_reflows += 1
                continue
            self._destroy_cached(victim_span, self._free[victim_span].pop())
            self.handle_reflows += 1
        return self._new_handle(span)

    # ------------------------------------------------------------- map paths

    def _map(self, va: int, span: int, handle) -> None:
        import time

        from cuda.bindings import driver as cu

        t0 = time.perf_counter()
        _dcheck(cu.cuMemMap(va, span, 0, handle, 0))
        _dcheck(cu.cuMemSetAccess(va, span, [self._access], 1))
        self.t_map_s += time.perf_counter() - t0
        self.maps += 1

    def _unmap(self, va: int, span: int) -> None:
        import time

        from cuda.bindings import driver as cu

        t0 = time.perf_counter()
        _dcheck(cu.cuMemUnmap(va, span))
        self.t_unmap_s += time.perf_counter() - t0

    def _va_grow(self) -> None:
        from cuda.bindings import driver as cu

        base = int(_dcheck(cu.cuMemAddressReserve(self.va_bytes, 0, 0, 0)))
        self._va_blocks.append((base, self.va_bytes))
        self._va_cursor = base
        self._va_block_end = base + self.va_bytes

    def _va_for(self, span: int) -> int:
        recycled = self._free_vas.get(span)
        if recycled:
            return recycled.pop()
        if self._va_cursor + span > self._va_block_end:
            self._va_grow()  # VA is free
        va = self._va_cursor
        self._va_cursor = va + span
        return va

    # ---------------------------------------------------------------- public

    def get(self, tag: object, size_bytes: int) -> Buffer:
        if tag is None:
            self._seq += 1
            tag = ("anon", self._seq)
        span = _align(size_bytes)
        slots = self._parked_slots.get(span)
        if slots:
            # adopt a still-mapped same-size slot: ZERO driver calls — the
            # steady-state path once round one has populated the slot pool
            va, handle = slots.pop()
            self._free_bytes -= span
            self.slot_adoptions += 1
            self.used_bytes += span
            self.peak_used_bytes = max(self.peak_used_bytes, self.used_bytes)
            self._seq += 1
            buf = Buffer(
                id=f"vmm:{tag}:{self._seq}", location="fast",
                size_bytes=size_bytes, ptr=va, raw=("vmm", tag, span, handle),
            )
            self._live[id(buf)] = buf
            return buf
        handle = self._obtain_handle(span)
        va = self._va_for(span)
        self._map(va, span, handle)
        self.used_bytes += span
        self.peak_used_bytes = max(self.peak_used_bytes, self.used_bytes)
        self._seq += 1
        buf = Buffer(
            id=f"vmm:{tag}:{self._seq}",
            location="fast",
            size_bytes=size_bytes,
            ptr=va,
            raw=("vmm", tag, span, handle),
        )
        self._live[id(buf)] = buf
        return buf

    def put(self, buffer: Buffer) -> None:
        assert isinstance(buffer.raw, tuple) and buffer.raw[0] == "vmm"
        _, tag, span, handle = buffer.raw
        self._live.pop(id(buffer), None)
        self.used_bytes -= span
        guard = buffer.guard_event
        if guard is not None and not self.event_complete(guard):
            # a queued write (debug poison) may still touch this VA: keep the
            # mapping alive until the guard completes, then unmap + reclaim
            self._deferred.append(_Deferred(tag, buffer.ptr, span, handle, guard))
            return
        # PARK the slot: it stays mapped, and the next same-size birth
        # adopts it with zero driver calls; the budget reclaims parked
        # slots lazily, smallest first
        self._park_slot(buffer.ptr, span, handle)

    def prewarm(self, demand: dict) -> None:
        """Pre-create handles off the hot path from the dry run's exact
        per-size demand {(location, size): count}. Largest classes first —
        they cost the most to create later (page sanitization ~1.9 ms/GiB).
        Stops at the pool budget; steady-state gets then hit free lists."""
        want: list[list[int]] = []
        for (location, size), count in demand.items():
            if location != "fast":
                continue
            want.append([_align(size), count])
        # round-robin across classes, largest first WITHIN a pass: every
        # class gets seeded before any class gets its second handle —
        # largest-till-full starves the small classes and guarantees early
        # evictions of the expensive big handles
        want.sort(reverse=True)
        progressed = True
        while progressed:
            progressed = False
            for entry in want:
                span, count = entry
                if count <= 0 or self.created_bytes + span > self.pool_bytes:
                    continue
                self._release_handle(self._new_handle(span), span)
                self.prewarmed += 1
                entry[1] -= 1
                progressed = True

    def drain_reclaim(self) -> None:
        """Unmap + free every deferred entry whose guard has completed."""
        if not self._deferred:
            return
        self.reclaim_drains += 1
        still: list[_Deferred] = []
        for d in self._deferred:
            if self.event_complete(d.event):
                self._unmap(d.va, d.span)
                self._free_vas.setdefault(d.span, []).append(d.va)
                self._release_handle(d.handle, d.span)
            else:
                still.append(d)
        self._deferred = still

    def close(self) -> None:
        if self.closed:
            return
        from cuda.bindings import driver as cu

        _dcheck(cu.cuCtxSynchronize())
        for d in self._deferred:
            self._unmap(d.va, d.span)
            _dcheck(cu.cuMemRelease(d.handle))
        self._deferred = []
        for buf in list(self._live.values()):
            _, _tag, span, handle = buf.raw
            self._unmap(buf.ptr, span)
            _dcheck(cu.cuMemRelease(handle))
        self._live = {}
        for span, slots in self._parked_slots.items():
            for va, handle in slots:
                self._unmap(va, span)
                _dcheck(cu.cuMemRelease(handle))
        self._parked_slots = {}
        for lst in self._free.values():
            for h in lst:
                _dcheck(cu.cuMemRelease(h))
        self._free = {}
        for base, size in self._va_blocks:
            _dcheck(cu.cuMemAddressFree(base, size))
        self._va_blocks = []
        if self._primary_ctx is not None:
            _dcheck(cu.cuDevicePrimaryCtxRelease(self._retained_device))
            self._primary_ctx = None
        self.closed = True
