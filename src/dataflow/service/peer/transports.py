"""Payload transports. P1.0 ships ``mem`` (the deterministic test
double); ``socket`` arrives with the loopback daemons; ``rdma-host``
with the two-box phase. The Transport contract at this layer is tiny:
an ORDERED message stream per direction with (msg, payload) frames —
exactly what PeerCore.handle consumes.

MemTransport faults are SCRIPTED, never random: drop/duplicate/delay
are keyed by 0-based delivery index of the direction's stream, and
``corrupt`` flips a payload byte — every adversarial interleaving in
the battery is reproducible by construction.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class FaultPlan:
    drop: set = field(default_factory=set)        # indices to swallow
    duplicate: set = field(default_factory=set)   # indices delivered twice
    corrupt: set = field(default_factory=set)     # payload byte flipped


class MemEndpoint:
    """One direction's sender handle: call it like PeerCore's ``send``."""

    def __init__(self, wire: deque, faults: FaultPlan):
        self.wire = wire
        self.faults = faults
        self.sent = 0

    def __call__(self, msg: dict, payload=None) -> None:
        idx = self.sent
        self.sent += 1
        if idx in self.faults.drop:
            return
        frame = (dict(msg), None if payload is None else bytes(payload))
        if idx in self.faults.corrupt and frame[1]:
            data = bytearray(frame[1])
            data[0] ^= 0xFF
            frame = (frame[0], bytes(data))
        self.wire.append(frame)
        if idx in self.faults.duplicate:
            self.wire.append((dict(msg),
                              None if payload is None else bytes(payload)))


class MemLink:
    """A bidirectional in-memory link between two PeerCores.

    Usage: a_send = link.endpoint_ab; deliveries happen only when the
    test pumps (deterministic single-threaded scheduling)."""

    def __init__(self, faults_ab: FaultPlan | None = None,
                 faults_ba: FaultPlan | None = None):
        self.wire_ab: deque = deque()
        self.wire_ba: deque = deque()
        self.endpoint_ab = MemEndpoint(self.wire_ab, faults_ab or FaultPlan())
        self.endpoint_ba = MemEndpoint(self.wire_ba, faults_ba or FaultPlan())

    def pump(self, core_a, core_b, *, max_steps: int = 10_000) -> int:
        """Deliver queued frames alternately until both wires drain.
        Returns frames delivered."""
        delivered = 0
        for _ in range(max_steps):
            progressed = False
            if self.wire_ab:
                msg, payload = self.wire_ab.popleft()
                core_b.handle(msg, payload)
                delivered += 1
                progressed = True
            if self.wire_ba:
                msg, payload = self.wire_ba.popleft()
                core_a.handle(msg, payload)
                delivered += 1
                progressed = True
            if not progressed:
                return delivered
        raise RuntimeError("pump did not drain (livelock?)")
