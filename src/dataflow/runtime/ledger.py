"""Host-authoritative byte ledger.

One counter per location, mirroring the simulator's accounting: bytes are
charged when a task's outputs are reserved or a transfer *starts* (never at
enqueue), and freed when a release retires or an offload completes. The
ledger never queries the device; it changes only inside completion-token
handlers and dispatch decisions, so its trajectory in virtual time is
directly comparable to the simulator's pool totals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .device.base import Location


class LedgerError(RuntimeError):
    pass


@dataclass
class Ledger:
    fast_capacity: int | None = None
    backing_capacity: int | None = None
    used: dict[str, int] = field(default_factory=lambda: {"fast": 0, "backing": 0})
    peak_fast_bytes: int = 0
    peak_backing_bytes: int = 0
    # observer(t_us, used_fast) for memory tracing; wired by the engine
    on_change: Callable[[str, int], None] | None = None

    def capacity(self, location: Location) -> int | None:
        return self.fast_capacity if location == "fast" else self.backing_capacity

    def free_bytes(self, location: Location) -> int | None:
        cap = self.capacity(location)
        if cap is None:
            return None
        return cap - self.used[location]

    def can_reserve(self, location: Location, size_bytes: int) -> bool:
        if size_bytes <= 0:
            return True
        cap = self.capacity(location)
        return cap is None or self.used[location] + size_bytes <= cap

    def charge(self, location: Location, size_bytes: int) -> None:
        if size_bytes <= 0:
            return
        cap = self.capacity(location)
        if cap is not None and self.used[location] + size_bytes > cap:
            raise LedgerError(
                f"over-commit on {location}: used={self.used[location]} + "
                f"{size_bytes} > capacity={cap} (admission bug — every charge "
                f"must have been admitted by can_reserve)"
            )
        self.used[location] += size_bytes
        if location == "fast":
            self.peak_fast_bytes = max(self.peak_fast_bytes, self.used["fast"])
        if location == "backing":
            self.peak_backing_bytes = max(self.peak_backing_bytes, self.used["backing"])
        if self.on_change is not None:
            self.on_change(location, self.used[location])

    def release(self, location: Location, size_bytes: int) -> None:
        if size_bytes <= 0:
            return
        self.used[location] -= size_bytes
        if self.used[location] < 0:
            raise LedgerError(f"negative usage on {location}: {self.used[location]}")
        if self.on_change is not None:
            self.on_change(location, self.used[location])
