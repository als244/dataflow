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
    overflow_slabs: dict[str, SlabAllocator] = field(default_factory=dict)
    free_lists: dict[tuple[str, int], list[Buffer]] = field(default_factory=dict)
    allocated_count: int = 0
    reused_count: int = 0
    slab_overflows: int = 0  # requests that escaped to the vendor allocator
    arena_carves: int = 0    # requests served by the pre-reserved overflow arena
    _seq: int = 0
    # --- static placement (fast location) ---
    # recording mode: dry runs log the instance stream for compute_placement
    recorder: object = None
    # assigned mode: instance offsets fixed offline; get() refuses ("busy")
    # while a prior overlapping instance is live — callers stall like capacity
    placement: object = None
    _placement_base: Buffer = None  # type: ignore[assignment]
    _incarnations: dict[str, int] = field(default_factory=dict)
    _live_ranges: dict[tuple, tuple[int, int]] = field(default_factory=dict)
    # (location, size) -> total buffers ever created: a completed run's map is
    # the exact prewarm demand for a repeat run (direct regime only).
    allocated_by_key: dict[tuple[str, int], int] = field(default_factory=dict)

    def add_slab(
        self, location: Location, capacity_bytes: int, *, overflow_bytes: int = 0
    ) -> None:
        self.slabs[location] = SlabAllocator(
            backend=self.backend, location=location, capacity_bytes=capacity_bytes
        )
        if overflow_bytes > 0:
            # reserved UP FRONT, before any op scratch can claim the VRAM:
            # fragmentation overflow then never competes with the torch cache
            self.overflow_slabs[location] = SlabAllocator(
                backend=self.backend, location=location,
                capacity_bytes=overflow_bytes, headroom_factor=0.0,
            )

    # --- static placement ------------------------------------------------------

    def enable_placement(self, placement) -> None:
        """Assigned mode for 'fast': one base allocation of the packing's
        proven extent; instance offsets are fixed. Replaces slab/arena/
        headroom heuristics entirely for this location."""
        self.placement = placement
        self._placement_base = self.backend.alloc("fast", placement.extent_bytes)
        self.reset_placement_epoch()

    def reset_placement_epoch(self) -> None:
        """Multi-step replay: each execute() reproduces the same instance
        keys, so incarnation counters restart per run."""
        self._incarnations = {}
        self._live_ranges = {}

    def _next_key(self, tag: str) -> tuple:
        return (tag, self._incarnations.get(tag, 0))

    def can_get(self, location: Location, size_bytes: int, tag: str | None = None) -> bool:
        """Side-effect-free admission check. Dynamic mode: always True.
        Assigned mode: False while a prior overlapping instance is live —
        callers stall exactly like a capacity block."""
        if self.placement is None or location != "fast" or tag is None:
            return True
        key = self._next_key(tag)
        offset = self.placement.offsets.get(key)
        if offset is None:
            return True  # unplanned instance: falls through to dynamic path
        end = offset + size_bytes
        for lo, hi in self._live_ranges.values():
            if lo < end and offset < hi:
                return False
        return True

    def get(self, location: Location, size_bytes: int, tag: str | None = None) -> Buffer:
        if self.recorder is not None and location == "fast" and tag is not None:
            key = self._next_key(tag)
            self._incarnations[tag] = key[1] + 1
            self.recorder.on_get(key, size_bytes)
            buf = self._get_dynamic(location, size_bytes)
            buf.tag = key
            return buf
        if self.placement is not None and location == "fast" and tag is not None:
            key = self._next_key(tag)
            offset = self.placement.offsets.get(key)
            if offset is not None:
                placed_size = self.placement.sizes[key]
                if size_bytes != placed_size:
                    raise RuntimeError(
                        f"placed instance {key} was recorded at {placed_size} bytes "
                        f"but is requested at {size_bytes}: static placement requires "
                        f"a shape-stable program (same object sizes every round/step). "
                        f"For variable-length training run with placement disabled "
                        f"(dynamic slab mode)."
                    )
                assert self.can_get(location, size_bytes, tag), (
                    f"assigned-mode get for {key} while offset range is live — "
                    f"callers must gate on can_get"
                )
                self._incarnations[tag] = key[1] + 1
                self._live_ranges[key] = (offset, offset + size_bytes)
                self._seq += 1
                return Buffer(
                    id=f"placed:{key[0]}:{key[1]}",
                    location="fast",
                    size_bytes=size_bytes,
                    ptr=self._placement_base.ptr + offset,
                    raw=("placed",),
                    tag=key,
                )
        return self._get_dynamic(location, size_bytes)

    def _get_dynamic(self, location: Location, size_bytes: int) -> Buffer:
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
            # fragmentation: flush pooled SLAB-BACKED free buffers back into
            # the slab (coalesces holes) and retry once. Overflow buffers stay
            # pooled — freeing them mid-run would call the vendor allocator
            # (sync risk) and forfeit their reuse.
            for (loc, _size), stack in self.free_lists.items():
                if loc != location:
                    continue
                keep = [b for b in stack if not (isinstance(b.raw, tuple) and b.raw[0] == "slab")]
                for buf in stack:
                    if isinstance(buf.raw, tuple) and buf.raw[0] == "slab":
                        buf.raw[2].free(buf.raw[1], buf.size_bytes)
                stack[:] = keep
            offset = slab.allocate(size_bytes)
        if offset is None:
            # Fragmentation beat the headroom heuristic (the static-assignment
            # mode replaces all of this with a planning-time placement proof).
            # An overflow-ARENA carve is cheap and safe (pre-reserved VRAM,
            # host-side offset math) and is counted separately; only escaping
            # to a VENDOR allocation counts as slab_overflows — the metric the
            # steady-state zero-alloc invariant asserts on.
            over = self.overflow_slabs.get(location)
            if over is not None:
                over_offset = over.allocate(size_bytes)
                if over_offset is not None:
                    self.arena_carves += 1
                    self._seq += 1
                    return over.make_buffer(over_offset, size_bytes, self._seq)
            self.slab_overflows += 1
            return self.backend.alloc(location, size_bytes)
        self._seq += 1
        return slab.make_buffer(offset, size_bytes, self._seq)

    def put(self, buffer: Buffer) -> None:
        if self.recorder is not None and buffer.tag is not None:
            self.recorder.on_put(buffer.tag)
        if isinstance(buffer.raw, tuple) and buffer.raw and buffer.raw[0] == "placed":
            self._live_ranges.pop(buffer.tag, None)
            return  # placed offsets are identity-managed, never free-listed
        self.free_lists.setdefault((buffer.location, buffer.size_bytes), []).append(buffer)

    def prewarm(self, demand: dict[tuple[str, int], int]) -> None:
        """Pre-create direct-regime buffers (e.g. pinned backing) so steady
        state never calls the vendor allocator. Slab-backed locations skip
        this — their single upfront allocation already happened."""
        for (location, size_bytes), count in sorted(demand.items()):
            if location in self.slabs:
                continue
            if location == "fast" and self.placement is not None:
                continue  # placed instances live in the placement base
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
                if isinstance(buf.raw, tuple) and buf.raw[0] == "slab":
                    buf.raw[2].free(buf.raw[1], buf.size_bytes)
                else:
                    self.backend.free(buf)
        self.free_lists.clear()
        for slab in self.slabs.values():
            slab.close()
        for slab in self.overflow_slabs.values():
            slab.close()
        self.slabs.clear()
        self.overflow_slabs.clear()
        if self._placement_base is not None:
            self.backend.free(self._placement_base)
            self._placement_base = None
            self.placement = None
