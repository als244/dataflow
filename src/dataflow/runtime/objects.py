"""Object table: per-object location slots and version state.

State machine per (object, location) slot — mirrors the simulator's
MemoryState vocabulary:

    reserved          output of a dispatched task; bytes charged, not yet written
    live              readable; `ready_event` guards device-side consumers
    pending_outbound  offload enqueued, not yet started (source stays readable
                      for nothing — the sim forbids compute use in this state)
    outbound          offload in flight; bytes still charged until completion
    pending_inbound   prefetch destination created at transfer start (transient)
    inbound           prefetch in flight into this slot
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dataflow.core import TensorMeta
from .device.base import Buffer, Event

SlotState = Literal[
    "reserved",
    "live",
    "pending_outbound",
    "outbound",
    "pending_inbound",
    "inbound",
]


@dataclass
class Slot:
    buffer: Buffer
    state: SlotState
    version: int
    ready_event: Event | None = None


@dataclass
class ObjectRecord:
    id: str
    size_bytes: int
    role: str = "other"
    tensor: TensorMeta | None = None
    version: int = 0
    fast: Slot | None = None
    backing: Slot | None = None

    def slot(self, location: str) -> Slot | None:
        return self.fast if location == "fast" else self.backing

    def set_slot(self, location: str, slot: Slot | None) -> None:
        if location == "fast":
            self.fast = slot
        else:
            self.backing = slot


@dataclass
class ObjectTable:
    records: dict[str, ObjectRecord] = field(default_factory=dict)

    def get(self, object_id: str) -> ObjectRecord:
        rec = self.records.get(object_id)
        if rec is None:
            raise KeyError(f"unknown object {object_id!r}")
        return rec

    def add(self, record: ObjectRecord) -> ObjectRecord:
        existing = self.records.get(record.id)
        if existing is not None:
            return existing
        self.records[record.id] = record
        return record

    def fast_state(self, object_id: str) -> SlotState | None:
        rec = self.records.get(object_id)
        if rec is None or rec.fast is None:
            return None
        return rec.fast.state
