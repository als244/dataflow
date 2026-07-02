"""Per-direction transfer engines (from_slow = h2d prefetch, to_slow = d2h offload).

Simulator-exact semantics:

- FIFO queue per direction, at most one transfer in flight;
- a queued transfer consumes NO destination bytes until it starts;
- the queue head blocks while destination capacity is insufficient and is
  re-attempted whenever bytes free (the engine pokes `try_start`);
- to_slow may overwrite an existing live backing entry of the same size
  (no new charge); completion frees the fast source and makes backing live;
- from_slow charges fast bytes at start; completion makes the fast slot live.

Transfer duration mirrors the simulator's formula exactly:
per-trigger override, else `max((size + bw - 1) // bw, 1)`.

A blocked or waiting queue never blocks the dispatcher — that asymmetry
(compute dispatch continues while transfers wait) is the core fix over the
prior attempt.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from .device.base import (
    PRIORITY_D2H_DONE,
    PRIORITY_H2D_DONE,
    DeviceBackend,
    Event,
    Stream,
)
from .ledger import Ledger
from .objects import ObjectTable, Slot
from .pool import BufferPool
from .trace import Interval, RunTrace, TraceEvent

Direction = Literal["from_slow", "to_slow"]


class TransferError(RuntimeError):
    pass


@dataclass
class TransferJob:
    object_id: str
    size_bytes: int
    runtime_override: float | None
    anchor_event: Event                 # task-end event that fired the directive
    fired_by_task: str
    start_event: Event | None = None
    done_event: Event | None = None
    interval_name: str = ""
    version: int = 0                    # source version captured at start


@dataclass(frozen=True)
class TransferDone:
    direction: Direction
    job: TransferJob


@dataclass
class TransferEngine:
    direction: Direction
    backend: DeviceBackend
    stream: Stream
    ledger: Ledger
    pool: BufferPool
    table: ObjectTable
    trace: RunTrace
    bandwidth: int | None
    poison: object = None  # optional debug hook: poison(buffer) before pooling
    queue: deque[TransferJob] = field(default_factory=deque)
    inflight: TransferJob | None = None
    _name_seq: dict[str, int] = field(default_factory=dict)

    @property
    def dst_location(self) -> str:
        return "fast" if self.direction == "from_slow" else "backing"

    def transfer_runtime_us(self, size_bytes: int, override: float | None) -> float:
        if override is not None:
            return max(float(override), 0.0)
        if self.bandwidth is None or self.bandwidth <= 0:
            raise TransferError(
                f"{self.direction} transfer needs bandwidth on the program or a "
                f"per-trigger runtime override"
            )
        return float(max((size_bytes + self.bandwidth - 1) // self.bandwidth, 1))

    def enqueue(self, job: TransferJob) -> None:
        self.queue.append(job)
        self.trace.events.append(
            TraceEvent(
                t=self.backend.host_now_us(),
                kind="transfer_enqueue",
                object_id=job.object_id,
                task_id=job.fired_by_task,
                detail=self.direction,
            )
        )

    def _interval_name(self, object_id: str) -> str:
        seq = self._name_seq.get(object_id, 0)
        self._name_seq[object_id] = seq + 1
        base = f"{self.direction}:{object_id}"
        return base if seq == 0 else f"{base}#{seq}"

    def try_start(self) -> None:
        """Start the queue head if the destination has room (sim semantics:
        destination bytes are charged HERE, at start — never at enqueue)."""
        if self.inflight is not None or not self.queue:
            return
        job = self.queue[0]
        rec = self.table.get(job.object_id)

        if self.direction == "from_slow":
            src_slot = rec.backing
            if src_slot is None:
                raise TransferError(
                    f"prefetch of {job.object_id!r} has no backing source at start"
                )
            if not self.ledger.can_reserve("fast", job.size_bytes):
                return  # head blocks; re-attempted on frees
            self.queue.popleft()
            self.ledger.charge("fast", job.size_bytes)
            dst_buffer = self.pool.get("fast", job.size_bytes)
            rec.fast = Slot(buffer=dst_buffer, state="inbound", version=src_slot.version)
            job.version = src_slot.version
            src_buffer = src_slot.buffer
            src_ready = src_slot.ready_event
        else:
            src_slot = rec.fast
            if src_slot is None:
                raise TransferError(
                    f"offload of {job.object_id!r} has no fast source at start"
                )
            existing = rec.backing
            need = 0 if existing is not None else job.size_bytes
            if not self.ledger.can_reserve("backing", need):
                return  # head blocks
            self.queue.popleft()
            job.version = src_slot.version
            if existing is None:
                self.ledger.charge("backing", job.size_bytes)
                dst_buffer = self.pool.get("backing", job.size_bytes)
                rec.backing = Slot(buffer=dst_buffer, state="inbound", version=src_slot.version)
            else:
                existing.state = "inbound"
                existing.version = src_slot.version
                dst_buffer = existing.buffer
            src_slot.state = "outbound"
            src_buffer = src_slot.buffer
            src_ready = src_slot.ready_event

        self.backend.stream_wait_event(self.stream, job.anchor_event)
        if src_ready is not None:
            self.backend.stream_wait_event(self.stream, src_ready)
        if dst_buffer.guard_event is not None:
            # a debug poison touched these bytes on another stream; the copy
            # must not race it
            self.backend.stream_wait_event(self.stream, dst_buffer.guard_event)
            dst_buffer.guard_event = None
        self.backend.align_stream_to_host(self.stream)
        job.start_event = self.backend.record_event(self.stream)
        duration = self.transfer_runtime_us(job.size_bytes, job.runtime_override)
        self.backend.memcpy_async(
            dst_buffer, src_buffer, job.size_bytes, self.stream, duration_us=duration
        )
        job.done_event = self.backend.record_event(self.stream)
        job.interval_name = self._interval_name(job.object_id)
        priority = PRIORITY_H2D_DONE if self.direction == "from_slow" else PRIORITY_D2H_DONE
        self.backend.notify_after(
            self.stream, job.done_event, TransferDone(self.direction, job), priority=priority
        )
        self.inflight = job

    def complete(self, job: TransferJob) -> None:
        """Retire the in-flight transfer (called from its completion token)."""
        assert self.inflight is job, "completion for a job that is not in flight"
        rec = self.table.get(job.object_id)
        if self.direction == "from_slow":
            slot = rec.fast
            assert slot is not None
            slot.state = "live"
            slot.ready_event = job.done_event
        else:
            fast = rec.fast
            assert fast is not None and fast.state == "outbound"
            self.ledger.release("fast", job.size_bytes)
            if self.poison is not None:
                self.poison(fast.buffer)  # type: ignore[operator]
            self.pool.put(fast.buffer)
            rec.fast = None
            backing = rec.backing
            assert backing is not None
            backing.state = "live"
            backing.ready_event = job.done_event
        self.inflight = None
        assert job.start_event is not None and job.done_event is not None
        self.trace.intervals.append(
            Interval(
                task_id=job.interval_name,
                start=self.backend.event_time_us(job.start_event),
                end=self.backend.event_time_us(job.done_event),
                track=self.direction,
            )
        )
        self.trace.events.append(
            TraceEvent(
                t=self.backend.host_now_us(),
                kind="transfer_end",
                object_id=job.object_id,
                detail=self.direction,
            )
        )
