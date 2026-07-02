"""Slab sub-allocator: one device allocation sized to the fast budget.

Why: exact-size free lists alone never share memory across size classes, so
their physical high-water is the SUM of per-class maxima over time — which
can far exceed the ledger's logical peak (436MB weight-class maxima and
420MB context-class maxima never coexist logically, but pooled buffers of
both classes persist). Backing every buffer with offsets carved from a
single slab of `capacity` bytes makes physical usage track logical usage:
whatever the ledger admits, the slab can hold — up to fragmentation, which
best-fit + immediate coalescing keeps benign for transformer programs (few,
repeating sizes). If fragmentation ever defeats an admitted request even
after reclaiming all pooled buffers, that's a loud invariant error (the VMM
remap upgrade path in the plan).

Alignment: 256-byte offsets (safe for all vector loads and copy engines).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .device.base import Buffer, DeviceBackend, Location

_ALIGN = 256


class SlabError(RuntimeError):
    pass


@dataclass
class SlabAllocator:
    backend: DeviceBackend
    location: Location
    capacity_bytes: int
    # Physical slack beyond the logical budget: the ledger enforces the
    # LOGICAL capacity exactly (sim semantics); headroom only absorbs
    # placement fragmentation + alignment. High-occupancy plans (~95%+ of
    # budget) need it. The static-assignment mode (planned follow-up) will
    # replace this heuristic with an offline placement proof.
    headroom_factor: float = 0.125
    base: Buffer | None = None
    # free holes as parallel sorted-by-offset lists
    _hole_offsets: list[int] = field(default_factory=list)
    _hole_sizes: list[int] = field(default_factory=list)
    used_bytes: int = 0

    def __post_init__(self) -> None:
        # absolute cap on headroom: at large budgets a proportional pad would
        # consume the VRAM the torch scratch lane and overflow fallback need
        headroom = min(int(self.capacity_bytes * self.headroom_factor), 2 * 1024**3)
        padded = self.capacity_bytes + headroom
        self.capacity_bytes = (padded + _ALIGN - 1) // _ALIGN * _ALIGN
        self.base = self.backend.alloc(self.location, self.capacity_bytes)
        self._hole_offsets = [0]
        self._hole_sizes = [self.capacity_bytes]

    def allocate(self, size_bytes: int) -> int | None:
        """Best-fit carve; returns offset or None if no hole fits."""
        size = (size_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        best = -1
        best_size = None
        for i, hole in enumerate(self._hole_sizes):
            if hole >= size and (best_size is None or hole < best_size):
                best, best_size = i, hole
                if hole == size:
                    break
        if best < 0:
            return None
        offset = self._hole_offsets[best]
        remaining = self._hole_sizes[best] - size
        if remaining:
            self._hole_offsets[best] = offset + size
            self._hole_sizes[best] = remaining
        else:
            del self._hole_offsets[best]
            del self._hole_sizes[best]
        self.used_bytes += size
        return offset

    def free(self, offset: int, size_bytes: int) -> None:
        size = (size_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        i = bisect.bisect_left(self._hole_offsets, offset)
        # coalesce with next hole
        if i < len(self._hole_offsets) and offset + size == self._hole_offsets[i]:
            self._hole_offsets[i] = offset
            self._hole_sizes[i] += size
        else:
            self._hole_offsets.insert(i, offset)
            self._hole_sizes.insert(i, size)
        # coalesce with previous hole
        if i > 0 and self._hole_offsets[i - 1] + self._hole_sizes[i - 1] == self._hole_offsets[i]:
            self._hole_sizes[i - 1] += self._hole_sizes[i]
            del self._hole_offsets[i]
            del self._hole_sizes[i]
        self.used_bytes -= size

    def make_buffer(self, offset: int, size_bytes: int, seq: int) -> Buffer:
        assert self.base is not None
        return Buffer(
            id=f"slab:{self.location}:{seq}",
            location=self.location,
            size_bytes=size_bytes,
            ptr=self.base.ptr + offset,
            raw=("slab", offset),
        )

    def close(self) -> None:
        if self.base is not None:
            self.backend.free(self.base)
            self.base = None
