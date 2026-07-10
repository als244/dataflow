"""Rendezvous transfer state machines — deterministic, thread-free.

``PeerCore`` is ONE side of one peer link: it initiates outbound sends
(sender machines) and serves inbound ones (receiver machines). It never
sleeps and never spawns threads — time enters through a caller-supplied
``clock()`` and elapses only when the caller invokes ``tick()``; wire
messages leave via the injected ``send(msg, payload)`` callable and
enter via ``handle(msg, payload)``. The threaded NetworkManager (P1)
pumps real sockets into exactly this interface; the protocol battery
pumps the in-memory transport with scripted faults.

Store integration is abstracted behind ``ReceiverEnv`` (reserve /
commit / release / lease check): the real embedder backs it with the
locked allocator + queued dispatcher commits; tests back it with an
in-memory fake. The core NEVER touches a catalog.

Failure discipline (fail-stop v1): NACK BUSY/CAPACITY retries with
bounded exponential backoff, COLLISION never retries; a transfer with
no progress for INACTIVITY_TIMEOUT_S aborts and releases its
reservation; CHUNK seqs must arrive strictly ascending; a checksum
mismatch at DONE aborts without committing (two-phase: nothing torn).
Completed send_ids live in a dedup ring so retransmitted RTS/DONE are
re-acked idempotently, never re-committed.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

from . import protocol as P
from .protocol import NackCode, TransferState


@dataclass
class Reservation:
    """A writable landing extent for one inbound transfer. The real env
    hands a view over the FINAL store extent (zero-copy landing); the
    fake hands a bytearray."""

    dest_id: str
    buffer: bytearray
    raddr: int = 0
    rkey: int = 0


class ReceiverEnv:
    """What the receiver side needs from its embedder. The real
    implementation routes reserve/release through the LOCKED allocator
    and commit through the dispatcher (metadata-only bind)."""

    def object_exists(self, dest_id: str) -> bool:
        raise NotImplementedError

    def object_size(self, dest_id: str) -> int:
        raise NotImplementedError

    def object_leased(self, dest_id: str) -> bool:
        raise NotImplementedError

    def reserve(self, dest_id: str, nbytes: int) -> Reservation | None:
        """None => CAPACITY refusal (allocator or in-flight quota)."""
        raise NotImplementedError

    def commit(self, res: Reservation, meta: dict) -> None:
        raise NotImplementedError

    def release(self, res: Reservation) -> None:
        raise NotImplementedError


@dataclass
class SenderTicket:
    """Caller-visible outcome of one send (the queued-verb ticket)."""

    send_id: str
    dest_id: str
    state: TransferState = TransferState.NEGOTIATING
    error: str | None = None
    bytes_done: int = 0
    bytes_total: int = 0


@dataclass
class SenderMachine:
    ticket: SenderTicket
    payload: bytes
    overwrite: bool
    meta: dict
    tries: int = 0
    next_rts_at: float = 0.0
    awaiting_ack_since: float | None = None
    last_progress: float = 0.0
    cleared: bool = False
    done_resends: int = 0


@dataclass
class ReceiverMachine:
    send_id: str
    res: Reservation
    meta: dict
    expect_bytes: int
    next_seq: int = 0
    received: int = 0
    last_progress: float = 0.0


class PeerCore:
    """One peer link's transfer logic (both directions)."""

    def __init__(self, env: ReceiverEnv, send: Callable, clock: Callable,
                 *, chunk_bytes: int = P.CHUNK_BYTES_DEFAULT,
                 inactivity_s: float = P.INACTIVITY_TIMEOUT_S):
        self.env = env
        self.send = send
        self.clock = clock
        self.chunk_bytes = chunk_bytes
        self.inactivity_s = inactivity_s
        self.senders: dict[str, SenderMachine] = {}
        self.receivers: dict[str, ReceiverMachine] = {}
        # per-dest FIFO: one in-flight send per dest_id, rest queued
        self.dest_queues: dict[str, deque] = {}
        self.completed: OrderedDict[str, bool] = OrderedDict()  # dedup ring
        self.seq = 0

    # ------------------------------------------------ sender side

    def start_send(self, dest_id: str, payload: bytes, *,
                   overwrite: bool = False, meta: dict | None = None,
                   ) -> SenderTicket:
        self.seq += 1
        send_id = f"s{self.seq}"
        ticket = SenderTicket(send_id=send_id, dest_id=dest_id,
                              bytes_total=len(payload))
        machine = SenderMachine(ticket=ticket, payload=bytes(payload),
                                overwrite=overwrite, meta=dict(meta or {}),
                                last_progress=self.clock())
        self.senders[send_id] = machine
        queue = self.dest_queues.setdefault(dest_id, deque())
        queue.append(send_id)
        if len(queue) == 1:
            self.open_send(machine)
        return ticket

    def open_send(self, machine: SenderMachine) -> None:
        t = machine.ticket
        eager = t.bytes_total <= P.EAGER_MAX_BYTES
        msg = P.rts(t.send_id, t.dest_id, t.bytes_total,
                    meta=machine.meta, overwrite=machine.overwrite,
                    eager=eager)
        machine.last_progress = self.clock()
        if eager:
            self.send(msg, machine.payload)
            machine.awaiting_ack_since = self.clock()
        else:
            self.send(msg, None)

    def finish_sender(self, machine: SenderMachine, state: TransferState,
                      error: str | None = None) -> None:
        t = machine.ticket
        t.state = state
        t.error = error
        self.senders.pop(t.send_id, None)
        queue = self.dest_queues.get(t.dest_id)
        if queue and queue[0] == t.send_id:
            queue.popleft()
            if queue:
                self.open_send(self.senders[queue[0]])

    def handle_clearance(self, machine: SenderMachine) -> None:
        """CTS or ADDR: stream the payload + DONE."""
        machine.cleared = True
        t = machine.ticket
        data = machine.payload
        n = len(data)
        seq = 0
        for lo in range(0, n, self.chunk_bytes):
            hi = min(lo + self.chunk_bytes, n)
            self.send(P.chunk(t.send_id, seq, hi == n), data[lo:hi])
            t.bytes_done = hi
            seq += 1
        if n == 0:
            self.send(P.chunk(t.send_id, 0, True), b"")
        self.send(P.done(t.send_id, P.payload_checksum(data)), None)
        machine.awaiting_ack_since = self.clock()
        machine.last_progress = self.clock()
        t.state = TransferState.COMMITTING

    def handle_nack(self, machine: SenderMachine, code: str) -> None:
        if code == NackCode.COLLISION.value:
            self.finish_sender(machine, TransferState.ERROR,
                               f"NACK {code}")
            return
        machine.tries += 1
        if machine.tries >= P.NACK_BACKOFF_TRIES:
            self.finish_sender(machine, TransferState.ERROR,
                               f"TRANSFER_ABORTED after {machine.tries} "
                               f"NACK {code}")
            return
        delay = P.NACK_BACKOFF_BASE_S * (2 ** (machine.tries - 1))
        machine.next_rts_at = self.clock() + delay
        machine.last_progress = self.clock()

    # ------------------------------------------------ receiver side

    def handle_rts(self, msg: dict, payload) -> None:
        send_id = msg["send_id"]
        if send_id in self.completed:
            self.send(P.done_ack(send_id), None)   # idempotent re-ack
            return
        if send_id in self.receivers:
            return                                  # duplicate mid-flight
        dest_id = msg["dest_id"]
        nbytes = int(msg["bytes"])
        if self.env.object_leased(dest_id):
            self.send(P.nack(send_id, NackCode.BUSY), None)
            return
        if self.env.object_exists(dest_id):
            if not msg.get("overwrite"):
                self.send(P.nack(send_id, NackCode.COLLISION), None)
                return
            if self.env.object_size(dest_id) != nbytes:
                self.send(P.nack(send_id, NackCode.COLLISION), None)
                return
        res = self.env.reserve(dest_id, nbytes)
        if res is None:
            self.send(P.nack(send_id, NackCode.CAPACITY), None)
            return
        if msg.get("eager"):
            res.buffer[:] = payload or b""
            self.env.commit(res, msg.get("meta") or {})
            self.remember_completed(send_id)
            self.send(P.done_ack(send_id), None)
            return
        self.receivers[send_id] = ReceiverMachine(
            send_id=send_id, res=res, meta=msg.get("meta") or {},
            expect_bytes=nbytes, last_progress=self.clock())
        if res.raddr:
            self.send(P.addr(send_id, res.raddr, res.rkey), None)
        else:
            self.send(P.cts(send_id), None)

    def handle_chunk(self, msg: dict, payload) -> None:
        machine = self.receivers.get(msg["send_id"])
        if machine is None:
            return                                  # aborted/unknown: drop
        if msg["seq"] != machine.next_seq:
            self.abort_receiver(machine,
                                f"CHUNK seq {msg['seq']} != "
                                f"{machine.next_seq} (ordered-stream "
                                f"violation)")
            return
        data = payload or b""
        lo = machine.received
        machine.res.buffer[lo:lo + len(data)] = data
        machine.received += len(data)
        machine.next_seq += 1
        machine.last_progress = self.clock()

    def handle_done(self, msg: dict) -> None:
        send_id = msg["send_id"]
        if send_id in self.completed:
            self.send(P.done_ack(send_id), None)   # lost-ack resend
            return
        machine = self.receivers.get(send_id)
        if machine is None:
            return
        if machine.received != machine.expect_bytes:
            self.abort_receiver(machine,
                                f"DONE at {machine.received}/"
                                f"{machine.expect_bytes} bytes")
            return
        if P.payload_checksum(machine.res.buffer) != msg["checksum"]:
            self.abort_receiver(machine, "checksum mismatch")
            return
        self.env.commit(machine.res, machine.meta)
        self.receivers.pop(send_id, None)
        self.remember_completed(send_id)
        self.send(P.done_ack(send_id), None)

    def abort_receiver(self, machine: ReceiverMachine, why: str) -> None:
        self.env.release(machine.res)
        self.receivers.pop(machine.send_id, None)
        self.last_abort_reason = why

    def remember_completed(self, send_id: str) -> None:
        self.completed[send_id] = True
        while len(self.completed) > P.DEDUP_RING_SIZE:
            self.completed.popitem(last=False)

    # ------------------------------------------------ dispatch + time

    def handle(self, msg: dict, payload=None) -> None:
        kind = msg["kind"]
        if kind == "RTS":
            self.handle_rts(msg, payload)
        elif kind == "CHUNK":
            self.handle_chunk(msg, payload)
        elif kind == "DONE":
            self.handle_done(msg)
        elif kind in ("CTS", "ADDR"):
            machine = self.senders.get(msg["send_id"])
            if machine is not None and not machine.cleared:
                machine.ticket.state = TransferState.MOVING
                self.handle_clearance(machine)
        elif kind == "NACK":
            machine = self.senders.get(msg["send_id"])
            if machine is not None:
                self.handle_nack(machine, msg["code"])
        elif kind == "DONE_ACK":
            machine = self.senders.get(msg["send_id"])
            if machine is not None:
                self.finish_sender(machine, TransferState.DONE)
        else:
            raise ValueError(f"unknown peer message kind {kind!r}")

    def tick(self) -> None:
        """Advance time-driven behavior: due NACK retries, DONE resends,
        inactivity aborts. Call after advancing the clock (tests) or
        periodically (the threaded NM)."""
        now = self.clock()
        for machine in list(self.senders.values()):
            t = machine.ticket
            if machine.next_rts_at and now >= machine.next_rts_at \
                    and not machine.cleared:
                machine.next_rts_at = 0.0
                self.open_send(machine)
                continue
            if machine.awaiting_ack_since is not None \
                    and now - machine.awaiting_ack_since >= self.inactivity_s:
                if t.state == TransferState.COMMITTING:
                    if machine.done_resends >= 1:
                        # two silent windows: the receiver aborted (or
                        # died) — fail-stop, don't resend forever
                        self.finish_sender(machine, TransferState.ERROR,
                                           "TRANSFER_ABORTED no DONE_ACK")
                        continue
                    machine.done_resends += 1
                    self.send(P.done(t.send_id,
                                     P.payload_checksum(machine.payload)),
                              None)
                    machine.awaiting_ack_since = now
                    continue
                if t.state == TransferState.NEGOTIATING \
                        and not machine.next_rts_at:
                    self.finish_sender(machine, TransferState.ERROR,
                                       "TRANSFER_ABORTED inactivity")
                    continue
            if now - machine.last_progress >= self.inactivity_s \
                    and t.state == TransferState.NEGOTIATING \
                    and not machine.next_rts_at:
                self.finish_sender(machine, TransferState.ERROR,
                                   "TRANSFER_ABORTED inactivity")
        for machine in list(self.receivers.values()):
            if now - machine.last_progress >= self.inactivity_s:
                self.abort_receiver(machine, "inactivity")


class CollectiveQueue:
    """Per-group FIFO of collective jobs (spec §4.4 / §6): hostfn posts
    append; the NM consumes strictly in order. Pure ordering semantics —
    the reduction itself is the P3 backend's business."""

    def __init__(self):
        self.pending: deque = deque()
        self.completed: list = []

    def post(self, job) -> None:
        self.pending.append(job)

    def take(self):
        return self.pending.popleft() if self.pending else None

    def complete(self, job) -> None:
        self.completed.append(job)
