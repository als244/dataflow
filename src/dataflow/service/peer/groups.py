"""Peer groups (spec §4): named collective handles, conductor-created,
engine-tracked. This is the BOOKKEEPING + control-star layer — the
comm backends (hostmem staging, raw nccl) attach to these handles in
the follow-up; v1 handles carry {name, members, rank, world, backend}
and the coordinator-member CONTROL STAR that group_error fan-out and
bootstrap ride.

Shape: create_peer_group lands on the COORDINATOR (rank 0's daemon,
by the conductor's choice of target); the coordinator must already
hold control links to every member (the conductor connects the star
first — minimal topology, spec §2). The verb pushes GROUP_JOIN over
each star link, collects GROUP_ACKs (the init barrier), and only then
returns. Members learn their rank from the join frame. group_error
propagates member -> coordinator -> members (two hops, spec §7).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class GroupRecord:
    name: str
    members: tuple                # peer names, rank = index
    backend: str
    self_rank: int
    coordinator: str              # peer name of rank 0's daemon
    ready: bool = False
    acks: set = field(default_factory=set)
    error: str | None = None

    def handle_dict(self) -> dict:
        return {"name": self.name, "rank": self.self_rank,
                "world": len(self.members), "backend": self.backend,
                "members": list(self.members),
                "coordinator": self.coordinator}


class GroupTable:
    """The daemon's live-groups table (one per daemon; whole table is
    exposed to every run via TaskContext.groups)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.groups: dict[str, GroupRecord] = {}
        self.barriers: dict[str, threading.Event] = {}

    def create(self, rec: GroupRecord) -> None:
        with self.lock:
            if rec.name in self.groups:
                raise ValueError(f"group {rec.name} exists")
            self.groups[rec.name] = rec
            self.barriers[rec.name] = threading.Event()

    def ack(self, name: str, member: str) -> bool:
        """Record a member's GROUP_ACK; True when the barrier is full."""
        with self.lock:
            rec = self.groups.get(name)
            if rec is None:
                return False
            rec.acks.add(member)
            done = rec.acks >= set(rec.members) - {rec.members[rec.self_rank]}
            if done:
                rec.ready = True
                self.barriers[name].set()
            return done

    def wait_ready(self, name: str, timeout: float) -> bool:
        ev = self.barriers.get(name)
        return bool(ev and ev.wait(timeout))

    def adopt(self, rec: GroupRecord) -> None:
        """Member side: install a group announced by its coordinator."""
        with self.lock:
            self.groups[rec.name] = rec
            self.barriers.setdefault(rec.name, threading.Event()).set()
            rec.ready = True

    def drop(self, name: str) -> bool:
        with self.lock:
            self.barriers.pop(name, None)
            return self.groups.pop(name, None) is not None

    def mark_error(self, name: str, why: str) -> None:
        with self.lock:
            rec = self.groups.get(name)
            if rec is not None:
                rec.error = why

    def handles(self) -> dict:
        """{name -> handle dict} for TaskContext injection — READY,
        non-errored groups only (absent => task skip semantics)."""
        with self.lock:
            return {n: r.handle_dict() for n, r in self.groups.items()
                    if r.ready and r.error is None}

    def infos(self) -> list:
        with self.lock:
            return [{**r.handle_dict(), "ready": r.ready,
                     "error": r.error} for r in self.groups.values()]
