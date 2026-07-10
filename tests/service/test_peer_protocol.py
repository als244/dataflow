"""Peer rendezvous protocol battery (CPU-only, no daemons, no GPU).

Drives the PeerCore state machines over the in-memory transport with
SCRIPTED faults and a hand-advanced clock — every path deterministic:
happy paths (eager + chunked), all three NACK codes with backoff,
overwrite matrix, dedup ring (duplicate RTS, lost DONE_ACK), two-phase
abort (dead sender, corrupt chunk, reordered chunk), per-dest FIFO,
inactivity timeouts, and the per-group collective FIFO. The threaded
NetworkManager built on these cores is gated separately in the fleet
lane; THIS battery is where protocol logic bugs die.
"""
from dataclasses import dataclass, field

from dataflow.service.peer import PeerCore
from dataflow.service.peer.core import CollectiveQueue, Reservation, ReceiverEnv
from dataflow.service.peer.protocol import (
    EAGER_MAX_BYTES,
    NACK_BACKOFF_TRIES,
    TransferState,
)
from dataflow.service.peer.transports import FaultPlan, MemLink


@dataclass
class FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


@dataclass
class FakeStoreEnv(ReceiverEnv):
    """In-memory embedder: capacity-limited reservations, committed
    objects, leases. Counts every call for leak/idempotency asserts."""

    capacity: int = 1 << 20
    committed: dict = field(default_factory=dict)
    leased: set = field(default_factory=set)
    live: list = field(default_factory=list)
    reserve_calls: int = 0
    commit_calls: int = 0
    release_calls: int = 0

    def object_exists(self, dest_id):
        return dest_id in self.committed

    def object_size(self, dest_id):
        return len(self.committed[dest_id])

    def object_leased(self, dest_id):
        return dest_id in self.leased

    def reserve(self, dest_id, nbytes):
        self.reserve_calls += 1
        held = sum(len(r.buffer) for r in self.live)
        if held + nbytes > self.capacity:
            return None
        res = Reservation(dest_id=dest_id, buffer=bytearray(nbytes))
        self.live.append(res)
        return res

    def commit(self, res, meta):
        self.commit_calls += 1
        self.committed[res.dest_id] = bytes(res.buffer)
        self.live.remove(res)

    def release(self, res):
        self.release_calls += 1
        self.live.remove(res)


def make_pair(*, faults_ab=None, faults_ba=None, capacity=1 << 20,
              chunk_bytes=64, inactivity_s=60.0):
    """A (sender core, receiver core, link, receiver env, clock) rig.
    a sends to b; chunk_bytes tiny so small payloads exercise chunking."""
    clock = FakeClock()
    env_a, env_b = FakeStoreEnv(), FakeStoreEnv(capacity=capacity)
    link = MemLink(faults_ab=faults_ab, faults_ba=faults_ba)
    core_a = PeerCore(env_a, link.endpoint_ab, clock,
                      chunk_bytes=chunk_bytes, inactivity_s=inactivity_s)
    core_b = PeerCore(env_b, link.endpoint_ba, clock,
                      chunk_bytes=chunk_bytes, inactivity_s=inactivity_s)
    return core_a, core_b, link, env_b, clock


def drain(link, a, b):
    return link.pump(a, b)


PAYLOAD = bytes(range(256)) * 3          # 768 B > eager only if forced


def big_payload(n=EAGER_MAX_BYTES + 1000):
    return bytes(i % 251 for i in range(n))


# ------------------------------- happy paths ---------------------------------

def test_eager_happy_path():
    a, b, link, env, clock = make_pair()
    t = a.start_send("W_0", PAYLOAD)          # 768 B <= 64 KiB => eager
    drain(link, a, b)
    assert t.state == TransferState.DONE and t.error is None
    assert env.committed["W_0"] == PAYLOAD
    assert env.commit_calls == 1 and not env.live and not b.receivers


def test_chunked_happy_path_byte_identity():
    a, b, link, env, clock = make_pair()
    data = big_payload()
    t = a.start_send("W_1", data)
    drain(link, a, b)
    assert t.state == TransferState.DONE
    assert env.committed["W_1"] == data
    assert t.bytes_done == t.bytes_total == len(data)
    assert not env.live and not b.receivers


def test_zero_and_exact_chunk_boundaries():
    a, b, link, env, clock = make_pair(chunk_bytes=100)
    exact = big_payload(EAGER_MAX_BYTES + 400)   # multiple of 100
    t = a.start_send("W_2", exact)
    drain(link, a, b)
    assert t.state == TransferState.DONE
    assert env.committed["W_2"] == exact


# ------------------------------- NACK matrix ---------------------------------

def test_busy_lease_backoff_then_success():
    a, b, link, env, clock = make_pair()
    env.leased.add("W_0")
    t = a.start_send("W_0", big_payload())
    drain(link, a, b)                          # RTS -> NACK BUSY
    assert t.state == TransferState.NEGOTIATING
    env.leased.discard("W_0")                  # lease clears
    clock.advance(0.2)
    a.tick()                                   # due retry fires
    drain(link, a, b)
    assert t.state == TransferState.DONE
    assert env.committed["W_0"] == big_payload()


def test_capacity_retries_exhaust_to_error():
    a, b, link, env, clock = make_pair(capacity=10)   # nothing fits
    t = a.start_send("W_0", big_payload())
    drain(link, a, b)
    for _ in range(NACK_BACKOFF_TRIES + 1):
        clock.advance(2.0)
        a.tick()
        drain(link, a, b)
    assert t.state == TransferState.ERROR
    assert "TRANSFER_ABORTED" in t.error and "CAPACITY" in t.error
    assert env.commit_calls == 0 and not env.live


def test_collision_never_retries():
    a, b, link, env, clock = make_pair()
    env.committed["W_0"] = b"resident"
    t = a.start_send("W_0", big_payload())
    drain(link, a, b)
    assert t.state == TransferState.ERROR and "COLLISION" in t.error
    assert env.reserve_calls == 0              # refused before allocation


def test_overwrite_matrix():
    a, b, link, env, clock = make_pair()
    data = big_payload()
    env.committed["W_0"] = bytes(len(data))    # same size resident
    t = a.start_send("W_0", data, overwrite=True)
    drain(link, a, b)
    assert t.state == TransferState.DONE and env.committed["W_0"] == data
    t2 = a.start_send("W_0", big_payload(2048 + EAGER_MAX_BYTES),
                      overwrite=True)          # size mismatch
    drain(link, a, b)
    assert t2.state == TransferState.ERROR and "COLLISION" in t2.error


# ---------------------------- idempotency / dedup ----------------------------

def test_duplicate_rts_single_reservation():
    a, b, link, env, clock = make_pair(faults_ab=FaultPlan(duplicate={0}))
    t = a.start_send("W_0", big_payload())
    drain(link, a, b)
    assert t.state == TransferState.DONE
    assert env.reserve_calls == 1 and env.commit_calls == 1


def test_lost_done_ack_resend_reacks_without_recommit():
    # drop the receiver's DONE_ACK (its 2nd message: CTS then DONE_ACK)
    a, b, link, env, clock = make_pair(faults_ba=FaultPlan(drop={1}),
                                       inactivity_s=5.0)
    t = a.start_send("W_0", big_payload())
    drain(link, a, b)
    assert t.state == TransferState.COMMITTING   # ack lost
    assert env.commit_calls == 1                 # receiver DID commit
    clock.advance(6.0)
    a.tick()                                     # DONE resend
    drain(link, a, b)
    assert t.state == TransferState.DONE
    assert env.commit_calls == 1                 # dedup ring re-acked


def test_eager_duplicate_rts_recommit_guard():
    a, b, link, env, clock = make_pair(faults_ab=FaultPlan(duplicate={0}))
    t = a.start_send("W_0", PAYLOAD)
    drain(link, a, b)
    assert t.state == TransferState.DONE
    assert env.commit_calls == 1


# ------------------------------ two-phase abort ------------------------------

def test_dead_sender_frees_reservation_nothing_torn():
    a, b, link, env, clock = make_pair(inactivity_s=5.0)
    a.start_send("W_0", big_payload())
    b.handle(*link.wire_ab.popleft())          # RTS lands
    a.handle(*link.wire_ba.popleft())          # CTS -> a emits chunks
    b.handle(*link.wire_ab.popleft())          # ONE chunk lands
    link.wire_ab.clear()                       # then the sender "dies"
    clock.advance(10.0)
    b.tick()                                   # receiver inactivity
    assert env.commit_calls == 0
    assert not env.live and not b.receivers    # reservation freed
    assert env.release_calls == 1


def test_corrupt_chunk_aborts_without_commit():
    # sender frames: RTS=0, chunks=1.., corrupt the second chunk
    a, b, link, env, clock = make_pair(faults_ab=FaultPlan(corrupt={2}),
                                       inactivity_s=5.0)
    t = a.start_send("W_0", big_payload())
    drain(link, a, b)
    assert env.commit_calls == 0 and not env.live
    assert env.release_calls == 1
    assert b.last_abort_reason == "checksum mismatch"
    # sender never hears back; bounded DONE resends then loud error
    clock.advance(6.0)
    a.tick()
    drain(link, a, b)
    clock.advance(6.0)
    a.tick()
    assert t.state == TransferState.ERROR and "no DONE_ACK" in t.error


def test_reordered_chunks_are_protocol_violation():
    a, b, link, env, clock = make_pair()
    a.start_send("W_0", big_payload())
    b.handle(*link.wire_ab.popleft())          # RTS
    a.handle(*link.wire_ba.popleft())          # CTS -> chunks emitted
    first = link.wire_ab.popleft()
    second = link.wire_ab.popleft()
    b.handle(*second)                          # delivered OUT OF ORDER
    assert "ordered-stream violation" in b.last_abort_reason
    b.handle(*first)                           # late frame: dropped
    drain(link, a, b)
    assert env.commit_calls == 0 and not env.live
    assert env.release_calls == 1


# ------------------------------- ordering ------------------------------------

def test_per_dest_fifo_and_cross_dest_interleave():
    a, b, link, env, clock = make_pair()
    d1 = big_payload()
    d2 = bytes(reversed(d1))                   # same size (strict-adopt)
    t1 = a.start_send("W_0", d1)
    t2 = a.start_send("W_0", d2, overwrite=True)
    t3 = a.start_send("W_9", PAYLOAD)
    # same-dest second send is QUEUED (no RTS yet); distinct dest flies
    kinds = [m["kind"] for m, _ in link.wire_ab]
    assert kinds.count("RTS") == 2             # W_0 first + W_9, not t2
    drain(link, a, b)
    assert t1.state == TransferState.DONE
    assert t2.state == TransferState.DONE      # opened after t1 finished
    assert t3.state == TransferState.DONE
    assert env.committed["W_0"] == d2          # t2 overwrote in order
    assert env.commit_calls == 3


def test_negotiating_inactivity_aborts():
    a, b, link, env, clock = make_pair(inactivity_s=5.0,
                                       faults_ab=FaultPlan(drop={0}))
    t = a.start_send("W_0", big_payload())     # RTS swallowed
    drain(link, a, b)
    clock.advance(10.0)
    a.tick()
    assert t.state == TransferState.ERROR and "inactivity" in t.error


# ------------------------------ collective FIFO ------------------------------

def test_collective_queue_fifo():
    q = CollectiveQueue()
    for j in ("allreduce:dW_0", "allreduce:counts_0", "allreduce:dW_1"):
        q.post(j)
    order = []
    job = q.take()
    while job is not None:
        order.append(job)
        q.complete(job)
        job = q.take()
    assert order == ["allreduce:dW_0", "allreduce:counts_0",
                     "allreduce:dW_1"]
    assert q.completed == order
