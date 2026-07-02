"""Buffer pool: exact-size free lists, slab-backed where a capacity exists.

Logical byte accounting lives in the Ledger; the pool recycles physical
buffers. Two regimes per location:

- **Slab-backed** (used whenever the location has a finite capacity): buffers
  are offsets carved from one upfront allocation of that capacity, so
  physical usage tracks logical usage — cross-size sharing is structural.
  Exact-size free lists remain as the fast path; a class miss carves from
  the slab; if fragmentation defeats a carve, all pooled free buffers are
  flushed back to the slab (coalescing) and the carve retried. Only then is
  it a hard invariant error.
- **Direct** (capacity None): buffers come straight from the backend
  allocator with exact-size reuse. Fine for unbounded locations; NOT safe
  for a bounded device budget (per-class maxima sum over time).

No vendor allocation happens on the steady-state path in either regime:
slabs are allocated up front; direct-regime prewarm covers known demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .device.base import Buffer, DeviceBackend, Location
from .slab import SlabAllocator, SlabError


@dataclass
class BufferPool:
    backend: DeviceBackend
    slabs: dict[str, SlabAllocator] = field(default_factory=dict)
    free_lists: dict[tuple[str, int], list[Buffer]] = field(default_factory=dict)
    allocated_count: int = 0
    reused_count: int = 0
    slab_overflows: int = 0
    _seq: int = 0
    # (location, size) -> total buffers ever created: a completed run's map is
    # the exact prewarm demand for a repeat run (direct regime only).
    allocated_by_key: dict[tuple[str, int], int] = field(default_factory=dict)

    def add_slab(self, location: Location, capacity_bytes: int) -> None:
        self.slabs[location] = SlabAllocator(
            backend=self.backend, location=location, capacity_bytes=capacity_bytes
        )

    def get(self, location: Location, size_bytes: int) -> Buffer:
        stack = self.free_lists.get((location, size_bytes))
        if stack:
            self.reused_count += 1
            return stack.pop()
        self.allocated_count += 1
        key = (location, size_bytes)
        self.allocated_by_key[key] = self.allocated_by_key.get(key, 0) + 1
        slab = self.slabs.get(location)
        if slab is None:
            return self.backend.alloc(location, size_bytes)
        return self._carve(slab, location, size_bytes)

    def _carve(self, slab: SlabAllocator, location: Location, size_bytes: int) -> Buffer:
        offset = slab.allocate(size_bytes)
        if offset is None:
            # fragmentation: flush every pooled free buffer of this location
            # back into the slab (coalesces holes) and retry once
            for (loc, _size), stack in self.free_lists.items():
                if loc != location:
                    continue
                while stack:
                    buf = stack.pop()
                    if isinstance(buf.raw, tuple) and buf.raw[0] == "slab":
                        slab.free(buf.raw[1], buf.size_bytes)
                    else:
                        self.backend.free(buf)  # overflow buffer: return to vendor
            offset = slab.allocate(size_bytes)
        if offset is None:
            # Last resort: overflow to a direct vendor allocation (counted —
            # nonzero overflow means the headroom heuristic was beaten; the
            # static-assignment mode replaces this with a planning-time
            # placement proof).
            self.slab_overflows += 1
            return self.backend.alloc(location, size_bytes)
        self._seq += 1
        return slab.make_buffer(offset, size_bytes, self._seq)

    def put(self, buffer: Buffer) -> None:
        self.free_lists.setdefault((buffer.location, buffer.size_bytes), []).append(buffer)

    def prewarm(self, demand: dict[tuple[str, int], int]) -> None:
        """Pre-create direct-regime buffers (e.g. pinned backing) so steady
        state never calls the vendor allocator. Slab-backed locations skip
        this — their single upfront allocation already happened."""
        for (location, size_bytes), count in sorted(demand.items()):
            if location in self.slabs:
                continue
            stack = self.free_lists.setdefault((location, size_bytes), [])
            while len(stack) < count:
                self.allocated_count += 1
                self.allocated_by_key[(location, size_bytes)] = (
                    self.allocated_by_key.get((location, size_bytes), 0) + 1
                )
                stack.append(self.backend.alloc(location, size_bytes))

    def drain(self) -> None:
        for (location, _size), stack in self.free_lists.items():
            while stack:
                buf = stack.pop()
                slab = self.slabs.get(location)
                if slab is not None and isinstance(buf.raw, tuple) and buf.raw[0] == "slab":
                    slab.free(buf.raw[1], buf.size_bytes)
                else:
                    self.backend.free(buf)
        self.free_lists.clear()
        for slab in self.slabs.values():
            slab.close()
        self.slabs.clear()
