"""Buffer pool: exact-size free lists over DeviceBackend allocations.

Logical byte accounting lives in the Ledger; the pool only recycles physical
buffers. Exact-size reuse makes physical availability equal logical
availability whenever sizes repeat (transformer chains repeat a handful of
sizes). On the fake backend allocation is free; the real backend (M2) adds
preallocation/warmup so steady state never calls the vendor allocator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .device.base import Buffer, DeviceBackend, Location


@dataclass
class BufferPool:
    backend: DeviceBackend
    free_lists: dict[tuple[str, int], list[Buffer]] = field(default_factory=dict)
    allocated_count: int = 0
    reused_count: int = 0

    def get(self, location: Location, size_bytes: int) -> Buffer:
        stack = self.free_lists.get((location, size_bytes))
        if stack:
            self.reused_count += 1
            return stack.pop()
        self.allocated_count += 1
        return self.backend.alloc(location, size_bytes)

    def put(self, buffer: Buffer) -> None:
        self.free_lists.setdefault((buffer.location, buffer.size_bytes), []).append(buffer)

    def drain(self) -> None:
        for stack in self.free_lists.values():
            while stack:
                self.backend.free(stack.pop())
        self.free_lists.clear()
