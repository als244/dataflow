"""Virtual-clock DeviceBackend.

Streams carry virtual clocks; enqueued work advances them; events capture
stream time; completion tokens are delivered from a heap ordered by
(time, priority, seq) — exactly the simulator's processing order. Drives the
M1 parity gate and all CI-without-GPU testing.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from .base import Buffer, Event, Location, Stream, StreamKind


@dataclass
class FakeBackend:
    name: str = "fake"
    _host_us: float = 0.0
    _seq: int = 0
    _pending: list[tuple[float, int, int, Any]] = field(default_factory=list)
    _alloc_bytes: dict[str, int] = field(default_factory=lambda: {"fast": 0, "backing": 0})
    _next_ptr: int = 16  # synthetic pointers, never dereferenced

    # --- streams & events ---------------------------------------------------
    def create_stream(self, kind: StreamKind) -> Stream:
        self._seq += 1
        return Stream(id=f"{kind}:{self._seq}", kind=kind)

    def record_event(self, stream: Stream) -> Event:
        self._seq += 1
        return Event(id=f"ev{self._seq}", time_us=stream.clock_us, completed=False)

    def stream_wait_event(self, stream: Stream, event: Event) -> None:
        stream.clock_us = max(stream.clock_us, event.time_us)

    # --- memory ---------------------------------------------------------------
    def alloc(self, location: Location, size_bytes: int) -> Buffer:
        self._seq += 1
        ptr = self._next_ptr
        self._next_ptr += max(size_bytes, 1)
        self._alloc_bytes[location] += size_bytes
        return Buffer(id=f"buf{self._seq}", location=location, size_bytes=size_bytes, ptr=ptr)

    def free(self, buffer: Buffer) -> None:
        self._alloc_bytes[buffer.location] -= buffer.size_bytes

    # --- async work -----------------------------------------------------------
    def memcpy_async(
        self,
        dst: Buffer,
        src: Buffer,
        size_bytes: int,
        stream: Stream,
        *,
        duration_us: float | None = None,
    ) -> tuple[float, float]:
        if duration_us is None:
            raise ValueError("fake backend requires duration_us for memcpy_async")
        return self.advance_stream(stream, duration_us)

    def advance_stream(self, stream: Stream, duration_us: float) -> tuple[float, float]:
        start = max(stream.clock_us, self._host_us)
        stream.clock_us = start + duration_us
        return start, stream.clock_us

    # --- completion tokens ------------------------------------------------------
    def notify_after(self, stream: Stream, event: Event, token: Any, *, priority: int) -> None:
        self._seq += 1
        heapq.heappush(self._pending, (event.time_us, priority, self._seq, token))

    def next_completion(self) -> Any | None:
        if not self._pending:
            return None
        time_us, _prio, _seq, token = heapq.heappop(self._pending)
        self._host_us = max(self._host_us, time_us)
        return token

    def host_now_us(self) -> float:
        return self._host_us

    def sync_all(self) -> None:
        while self._pending:
            self.next_completion()
