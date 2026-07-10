"""hostmem comm backend: device-buffer collectives staged through
pinned host scratch, exchanged over the peer link, reduced in fp32 on
the CPU — the shared-GPU loopback path and the no-GPUDirect two-box
path (spec §4).

Choreography per op, ENQUEUE-ONLY from the caller's thread (the task
launch never blocks):

    caller (on gh.stream):        worker thread (one per group):
      D2H tensor -> scratch
      record CUDA event  ------->   event.synchronize()
      cuStreamWaitValue32(flag,      exchange scratch with the peer
        GEQ, seq)                    (COLL frames on the link; the
      H2D scratch -> tensor          link READER feeds the inbox, so
      [caller returns]               both sides send-then-recv without
                                     deadlock)
                                     rank-ordered fp32 reduce
                                     flag store = seq  ==> stream
                                     resumes, H2D drains

The spec sketched a hostfn posting the job; a recorded EVENT gives the
worker the same ordering guarantee with no host-callback machinery —
the job is queued directly at call time and the worker waits on the
event before touching scratch. The stream-side park/release is exactly
the pinned decision (cuStreamWaitValue32 on a mapped pinned flag,
monotonic per-group sequence, plain host store releases; probed live
on this hardware before this file was written).

Rank-ordered accumulation: the sum is ALWAYS rank0 + rank1 (fp32),
never mine + theirs — every member computes bitwise-identical results.
World 2 only in v1 (pairwise exchange-and-add; ring topologies arrive
when a world > 2 exists to test them). Wire dtype = the tensor's own
(bf16 in practice); accumulation fp32.
"""
from __future__ import annotations

import queue
import threading

import torch
from cuda.bindings import driver as cudriver

from ...runtime.groups import GroupHandle

FLAG_WAIT_GEQ = None  # resolved at probe time


def stream_wait_value_supported() -> bool:
    """Boot probe: enqueue a GEQ wait on an already-satisfied flag."""
    global FLAG_WAIT_GEQ
    try:
        FLAG_WAIT_GEQ = cudriver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
        flag = torch.ones(1, dtype=torch.int32, pin_memory=True)
        err, dptr = cudriver.cuMemHostGetDevicePointer(flag.data_ptr(), 0)
        if int(err) != 0:
            return False
        s = torch.cuda.Stream()
        (rc,) = cudriver.cuStreamWaitValue32(int(s.cuda_stream), int(dptr),
                                             1, FLAG_WAIT_GEQ)
        s.synchronize()
        return int(rc) == 0
    except Exception:
        return False


class CollJob:
    def __init__(self, seq: int, ready_event, action, payload_view,
                 recv_view):
        self.seq = seq
        self.ready_event = ready_event    # D2H landed when this fires
        self.action = action              # worker verb
        self.payload_view = payload_view  # bytes we SEND (or None)
        self.recv_view = recv_view        # where peer bytes LAND (or None)


class HostmemComm:
    """One per (group, daemon). Requires world == 2 and a live peer
    link (the coordinator star IS the pairwise link at world 2)."""

    def __init__(self, nm, group_name: str, rank: int, world: int,
                 peer_name: str):
        if world != 2:
            raise RuntimeError(
                f"hostmem v1 is pairwise (world 2); group "
                f"{group_name!r} has world {world} — ring topologies "
                f"arrive with a real >2 fleet")
        self.nm = nm
        self.group = group_name
        self.rank = rank
        self.world = world
        self.peer_name = peer_name
        self.stream = torch.cuda.Stream()
        self.seq = 0
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()
        self.inbox: dict[int, bytes] = {}
        self.inbox_cv = threading.Condition()
        self.scratch = torch.empty(0, dtype=torch.uint8, pin_memory=True)
        self.scratch_np = self.scratch.numpy()   # wire view (buffer proto)
        self.flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        err, dptr = cudriver.cuMemHostGetDevicePointer(
            self.flag.data_ptr(), 0)
        if int(err) != 0:
            raise RuntimeError("mapped pinned flag unavailable")
        self.flag_dptr = int(dptr)
        self.wait_value_ok = stream_wait_value_supported()
        self.dead: str | None = None
        self.worker = threading.Thread(target=self.worker_loop,
                                       name=f"nm-coll-{group_name}",
                                       daemon=True)
        self.worker.start()

    # ---------------------------------------------------- link plumbing

    def deliver(self, seq: int, payload: bytes) -> None:
        """Called by the link READER thread for COLL frames."""
        with self.inbox_cv:
            self.inbox[seq] = payload
            self.inbox_cv.notify_all()

    def send_frame(self, seq: int, view) -> None:
        link = self.nm.links.get(self.peer_name)
        if link is None or not link.alive:
            raise RuntimeError(f"peer {self.peer_name} down")
        link.send_frame({"kind": "COLL", "group": self.group,
                         "seq": seq}, view)

    def await_peer(self, seq: int, timeout: float = 120.0) -> bytes:
        with self.inbox_cv:
            ok = self.inbox_cv.wait_for(
                notify_check(self.inbox, seq), timeout)
            if not ok:
                raise RuntimeError(f"collective seq {seq}: peer timeout")
            return self.inbox.pop(seq)

    def fail(self, why: str) -> None:
        self.dead = why
        self.flag[0] = 2_000_000_000       # release any parked stream
        self.nm.group_error(self.group, why, fan_out=None)

    # ---------------------------------------------------- caller side

    def ensure_scratch(self, nbytes: int) -> None:
        if self.scratch.numel() < nbytes:
            self.scratch = torch.empty(nbytes, dtype=torch.uint8,
                                       pin_memory=True)
            self.scratch_np = self.scratch.numpy()

    def enqueue(self, tensor, action: str, *, send_stage=True,
                recv_into_tensor=True, root: int | None = None,
                out=None):
        """The shared enqueue-only choreography. Returns the op seq."""
        if self.dead:
            raise RuntimeError(f"group {self.group} dead: {self.dead}")
        self.seq += 1
        seq = self.seq
        nbytes = tensor.numel() * tensor.element_size()
        self.ensure_scratch(nbytes)
        stage = self.scratch[:nbytes]
        ev = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            if send_stage:
                stage.view(tensor.dtype)[:tensor.numel()].copy_(
                    tensor.reshape(-1), non_blocking=True)
            ev.record(self.stream)
            if self.wait_value_ok:
                cudriver.cuStreamWaitValue32(
                    int(self.stream.cuda_stream), self.flag_dptr, seq,
                    FLAG_WAIT_GEQ)
                if recv_into_tensor:
                    target = tensor if out is None else out
                    tn = target.numel()
                    target.reshape(-1).copy_(
                        stage.view(target.dtype)[:tn], non_blocking=True)
        self.jobs.put(CollJob(seq, ev, (action, tensor.dtype, nbytes,
                                        root, None), stage, stage))
        if not self.wait_value_ok:
            # fallback: block the CALLER until the worker releases,
            # THEN enqueue the copy-back — correctness identical,
            # overlap lost (documented)
            spin_until(self.flag, seq, self.dead_check)
            if recv_into_tensor:
                target = tensor if out is None else out
                tn = target.numel()
                with torch.cuda.stream(self.stream):
                    target.reshape(-1).copy_(
                        stage.view(target.dtype)[:tn], non_blocking=True)
        return seq

    def dead_check(self) -> bool:
        return self.dead is not None

    # ---------------------------------------------------- worker side

    def worker_loop(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self.run_job(job)
                self.flag[0] = job.seq          # releases the stream
            except Exception as ex:
                self.fail(f"collective failed: {ex}")
                return

    def run_job(self, job: CollJob) -> None:
        action, dtype, nbytes, root, _ = job.action
        job.ready_event.synchronize()           # D2H landed
        stage = job.payload_view[:nbytes]
        np_stage = self.scratch_np[:nbytes]
        if action == "allreduce":
            self.send_frame(job.seq, np_stage)
            peer = self.await_peer(job.seq)
            mine = stage.view(dtype).float()
            theirs = torch.frombuffer(bytearray(peer),
                                      dtype=torch.uint8).view(dtype).float()
            first, second = ((mine, theirs) if self.rank == 0
                             else (theirs, mine))
            stage.view(dtype).copy_((first + second).to(dtype))
        elif action == "broadcast":
            if self.rank == root:
                self.send_frame(job.seq, np_stage)
            else:
                peer = self.await_peer(job.seq)
                stage[:] = torch.frombuffer(bytearray(peer),
                                            dtype=torch.uint8)
        elif action == "reduce":
            if self.rank != root:
                self.send_frame(job.seq, np_stage)
            else:
                peer = self.await_peer(job.seq)
                mine = stage.view(dtype).float()
                theirs = torch.frombuffer(
                    bytearray(peer), dtype=torch.uint8).view(dtype).float()
                first, second = ((mine, theirs) if self.rank == 0
                                 else (theirs, mine))
                stage.view(dtype).copy_((first + second).to(dtype))
        elif action == "reduce_scatter":
            half = nbytes // 2
            peer_lo = half if self.rank == 0 else 0
            own_lo = 0 if self.rank == 0 else half
            self.send_frame(job.seq, np_stage[peer_lo:peer_lo + half])
            peer = self.await_peer(job.seq)
            mine = stage[own_lo:own_lo + half].view(dtype).float()
            theirs = torch.frombuffer(bytearray(peer),
                                      dtype=torch.uint8).view(dtype).float()
            first, second = ((mine, theirs) if self.rank == 0
                             else (theirs, mine))
            stage[:half].view(dtype).copy_((first + second).to(dtype))
        elif action == "all_gather":
            half = nbytes // 2
            own_lo = 0 if self.rank == 0 else half
            self.send_frame(job.seq, np_stage[own_lo:own_lo + half])
            peer = self.await_peer(job.seq)
            peer_lo = half if self.rank == 0 else 0
            stage[peer_lo:peer_lo + half] = torch.frombuffer(
                bytearray(peer), dtype=torch.uint8)
        else:
            raise ValueError(action)

    # ---------------------------------------------------- GroupHandle ops

    def allreduce(self, tensor) -> None:
        self.enqueue(tensor, "allreduce")

    def broadcast(self, tensor, root: int) -> None:
        self.enqueue(tensor, "broadcast", send_stage=self.rank == root,
                     root=root)

    def reduce(self, tensor, root: int) -> None:
        self.enqueue(tensor, "reduce",
                     recv_into_tensor=self.rank == root, root=root)

    def reduce_scatter(self, full, out) -> None:
        if out.numel() * 2 != full.numel():
            raise ValueError("world-2 reduce_scatter: out must be half")
        self.enqueue_scatter(full, out, "reduce_scatter")

    def all_gather(self, own_slice, full) -> None:
        if own_slice.numel() * 2 != full.numel():
            raise ValueError("world-2 all_gather: slice must be half")
        self.enqueue_gather(own_slice, full)

    def enqueue_scatter(self, full, out, action: str):
        if self.dead:
            raise RuntimeError(f"group {self.group} dead: {self.dead}")
        self.seq += 1
        seq = self.seq
        nbytes = full.numel() * full.element_size()
        half = nbytes // 2
        self.ensure_scratch(nbytes)
        stage = self.scratch[:nbytes]
        ev = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            stage.view(full.dtype)[:full.numel()].copy_(
                full.reshape(-1), non_blocking=True)
            ev.record(self.stream)
            if self.wait_value_ok:
                cudriver.cuStreamWaitValue32(
                    int(self.stream.cuda_stream), self.flag_dptr, seq,
                    FLAG_WAIT_GEQ)
                out.reshape(-1).copy_(
                    stage[:half].view(out.dtype)[:out.numel()],
                    non_blocking=True)
        self.jobs.put(CollJob(seq, ev, (action, full.dtype, nbytes,
                                        None, None), stage, stage))
        if not self.wait_value_ok:
            spin_until(self.flag, seq, self.dead_check)
            with torch.cuda.stream(self.stream):
                out.reshape(-1).copy_(
                    stage[:half].view(out.dtype)[:out.numel()],
                    non_blocking=True)

    def enqueue_gather(self, own_slice, full):
        if self.dead:
            raise RuntimeError(f"group {self.group} dead: {self.dead}")
        self.seq += 1
        seq = self.seq
        nbytes = full.numel() * full.element_size()
        half = nbytes // 2
        own_lo = 0 if self.rank == 0 else half
        self.ensure_scratch(nbytes)
        stage = self.scratch[:nbytes]
        ev = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            stage[own_lo:own_lo + half].view(own_slice.dtype)[
                :own_slice.numel()].copy_(own_slice.reshape(-1),
                                          non_blocking=True)
            ev.record(self.stream)
            if self.wait_value_ok:
                cudriver.cuStreamWaitValue32(
                    int(self.stream.cuda_stream), self.flag_dptr, seq,
                    FLAG_WAIT_GEQ)
                full.reshape(-1).copy_(
                    stage.view(full.dtype)[:full.numel()],
                    non_blocking=True)
        self.jobs.put(CollJob(seq, ev, ("all_gather", full.dtype, nbytes,
                                        None, None), stage, stage))
        if not self.wait_value_ok:
            spin_until(self.flag, seq, self.dead_check)
            with torch.cuda.stream(self.stream):
                full.reshape(-1).copy_(
                    stage.view(full.dtype)[:full.numel()],
                    non_blocking=True)


class notify_check:
    def __init__(self, inbox: dict, seq: int):
        self.inbox = inbox
        self.seq = seq

    def __call__(self) -> bool:
        return self.seq in self.inbox


def spin_until(flag, target: int, dead_check, timeout: float = 300.0):
    import time

    t0 = time.monotonic()
    while int(flag[0]) < target:
        if dead_check():
            raise RuntimeError("group died during collective")
        if time.monotonic() - t0 > timeout:
            raise RuntimeError("collective flag timeout")
        time.sleep(0.0005)


def build_comm(nm, rec) -> HostmemComm | None:
    """Attach the backend for a READY group record; None where comms
    are impossible (fake boot / no CUDA)."""
    if not torch.cuda.is_available() or nm.store.slab is None:
        return None
    peers = [m for i, m in enumerate(rec.members) if i != rec.self_rank]
    return HostmemComm(nm, rec.name, rec.self_rank, len(rec.members),
                       peers[0])


def build_handle(nm, rec) -> GroupHandle:
    comm = None
    stream = None
    backend = rec.backend if rec.backend != "auto" else "hostmem"
    try:
        if backend == "hostmem":
            comm = build_comm(nm, rec)
            stream = comm.stream if comm is not None else None
    except Exception as ex:
        nm.state.emit("group_error", name=rec.name,
                      why=f"comm bring-up failed: {ex}")
        raise
    return GroupHandle(name=rec.name, rank=rec.self_rank,
                       world=len(rec.members), backend=backend,
                       members=tuple(rec.members),
                       coordinator=rec.coordinator, stream=stream,
                       comm=comm)
