"""DeviceBackend: the runtime's only window onto a vendor GPU runtime.

Design constraints:

- Surface restricted to the CUDA∩HIP common subset so an AMD backend is a
  mechanical addition (`cudaStreamCreate`↔`hipStreamCreate`,
  `cudaEventRecord`↔`hipEventRecord`, `cudaMemcpyAsync`↔`hipMemcpyAsync`,
  `cudaLaunchHostFunc`↔`hipLaunchHostFunc`, ...).
- **Completion tokens** are the engine's only progress signal: the engine
  registers an opaque token against a stream point (`notify_after`) and later
  consumes tokens in completion order (`next_completion`). The fake backend
  orders tokens by virtual time; the cuda backend delivers them via host
  callbacks. The engine never polls device state and never sleeps.
- Virtual-time hooks (`advance_stream`, clamping to the host clock) exist so
  synthetic executables can model task durations on the fake backend; the
  cuda backend implements `advance_stream` as a no-op (real work takes real
  time).

Ordering contract for `next_completion`: tokens become ready when their
stream point completes; ties (same completion time on different streams) are
delivered in `priority` order (lower first). The fake backend implements this
exactly; the cuda backend approximates ties by arrival order, which is
harmless because real timestamps never tie exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Location = Literal["fast", "backing"]
StreamKind = Literal["compute", "h2d", "d2h"]

# Tie-break priorities at equal completion times, mirroring the simulator's
# processing order: from_slow completions, then to_slow, then task end.
PRIORITY_H2D_DONE = 0
PRIORITY_D2H_DONE = 1
PRIORITY_TASK_DONE = 2


@dataclass
class Stream:
    id: str
    kind: StreamKind
    raw: Any = None          # vendor handle (fake: None)
    clock_us: float = 0.0    # virtual clock (fake backend only)


@dataclass
class Event:
    id: str
    raw: Any = None
    time_us: float = 0.0     # virtual completion time (fake backend only)
    completed: bool = False


@dataclass
class Buffer:
    id: str
    location: Location
    size_bytes: int
    ptr: int = 0             # device/pinned pointer (fake: synthetic)
    raw: Any = None


class DeviceBackend(Protocol):
    """Minimal vendor-runtime surface. See module docstring."""

    name: str
    # True when allocations consume real device/host memory (drives slab
    # sub-allocation); the fake backend's notional allocations set False.
    physical: bool

    # --- streams & events ---------------------------------------------------
    def create_stream(self, kind: StreamKind) -> Stream: ...

    def record_event(self, stream: Stream) -> Event: ...

    def stream_wait_event(self, stream: Stream, event: Event) -> None:
        """Make future work on `stream` wait for `event` (device-side)."""
        ...

    def align_stream_to_host(self, stream: Stream) -> None:
        """Fake backend: clamp the stream's virtual clock to the host clock
        (work enqueued now cannot start in the past). Real backends: no-op."""
        ...

    def event_time_us(self, event: Event) -> float:
        """Timestamp of a COMPLETED event on the run's shared timebase.
        Fake: the virtual completion time. Real: elapsed time from the run's
        origin event. Only valid once the event has completed (i.e. from a
        token handler at or after its completion)."""
        ...

    # --- memory ---------------------------------------------------------------
    def alloc(self, location: Location, size_bytes: int) -> Buffer:
        """Device alloc for 'fast', pinned-host alloc for 'backing'.
        Setup/teardown paths only — never on the steady-state hot path."""
        ...

    def free(self, buffer: Buffer) -> None: ...

    # --- async work -----------------------------------------------------------
    def memcpy_async(
        self,
        dst: Buffer,
        src: Buffer,
        size_bytes: int,
        stream: Stream,
        *,
        duration_us: float | None = None,
    ) -> tuple[float, float] | None:
        """Enqueue an async copy. `duration_us` models the copy on the fake
        backend (which returns virtual (start_us, end_us)); real backends
        ignore it and return None — real intervals come from event timings."""
        ...

    def advance_stream(self, stream: Stream, duration_us: float) -> tuple[float, float]:
        """Model `duration_us` of device work on `stream` (fake backend);
        returns (start_us, end_us) in virtual time after clamping the stream
        clock to the host clock. Real backends return (nan, nan) no-ops —
        real executables enqueue real work instead."""
        ...

    # --- completion tokens ------------------------------------------------------
    def notify_after(self, stream: Stream, event: Event, token: Any, *, priority: int) -> None:
        """Deliver `token` via `next_completion` once `event` (just recorded
        on `stream`) completes."""
        ...

    def next_completion(self) -> Any | None:
        """Consume the next completion token in completion order, advancing
        the host clock to its completion time. Returns None when nothing is
        pending (the engine treats a blocked state + None as deadlock)."""
        ...

    def host_now_us(self) -> float: ...

    def mark_origin(self) -> None:
        """Reset the run timebase to 'now' — called by the engine after setup
        (pool prewarm, initial-object load) so traces measure execution, not
        allocation. Fake backend: no-op (setup takes zero virtual time)."""
        ...

    def sync_all(self) -> None:
        """Drain everything (shutdown/error paths only)."""
        ...
