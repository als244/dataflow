"""hostmem comm backend: device-buffer collectives staged through the
store's REGISTERED slab and exchanged peer-to-peer — RDMA_WRITE when
the link has RC QPs up, socket COLL frames otherwise (spec §4). The
shared-GPU loopback path and the no-GPUDirect two-box path.

Zero host<->host copies by construction: the staging regions are slab
extents, i.e. memory the NIC already has registered (the same MR that
object sends use) — the D2H stage lands gradient bytes DIRECTLY in
NIC-reachable memory, the peer's RDMA_WRITE lands DIRECTLY in ours,
and the reduction runs in place over the two regions. The only
device<->host copies are the unavoidable D2H/H2D (no GPUDirect on
these parts); there is no intermediate host buffer anywhere on the
rdma path. The socket fallback keeps the kernel's recv copy and
nothing else.

Choreography per op, ENQUEUE-ONLY from the caller's thread (the task
launch never blocks — async by default; sync points are the GPU
stream's own dependencies):

    caller (on gh.stream):        worker thread (one per group):
      D2H tensor -> out region
      record CUDA event  ------->   event.synchronize()
      cuStreamWaitValue32(flag,      exchange with the peer:
        GEQ, seq)                      rdma: RDY hdr (my landing addr)
      H2D out region -> tensor              -> RDMA_WRITE out region
      [caller returns]                      -> peer's landing region
                                            -> DONE hdr after CQ ack
                                       sock: COLL frame carries bytes
                                     reduce IN PLACE (native dtype by
                                       default; see below)
                                     flag store = seq  ==> stream
                                     resumes, H2D drains

Reduction is ALWAYS native dtype: the sum runs in the tensor's own
dtype — one in-place pass, no conversions; at world 2 a single add is
commutative, so replicas stay BITWISE identical (and torch/CUDA bf16
adds compute via fp32 internally, matching an fp32 reference for a
pairwise sum exactly).

World 2 only in v1 (pairwise exchange-and-add; ring topologies arrive
when a world > 2 exists to test them). broadcast / reduce_scatter /
all_gather ride socket frames in v1 — allreduce is the training hot
path and gets the rdma lane first.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque

import ctypes

import numpy as np
import torch
from cuda.bindings import driver as cudriver
from cuda.bindings import runtime as cudart

from ...runtime.groups import GroupHandle

FLAG_WAIT_GEQ = None  # resolved at probe time
WAIT_VALUE_PROBED = None  # cached probe verdict (one Stream, ever)


def stream_wait_value_supported() -> bool:
    """Boot probe: enqueue a GEQ wait on an already-satisfied flag.
    Probed once per process — comms reuse the verdict."""
    global FLAG_WAIT_GEQ, WAIT_VALUE_PROBED
    if WAIT_VALUE_PROBED is not None:
        return WAIT_VALUE_PROBED
    WAIT_VALUE_PROBED = run_wait_value_probe()
    return WAIT_VALUE_PROBED


def run_wait_value_probe() -> bool:
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
    def __init__(self, seq: int, ready_event, action, lane: str):
        self.seq = seq
        self.ready_event = ready_event    # D2H landed when this fires
        self.action = action              # (verb, dtype, nbytes, root)
        self.lane = lane                  # "rdma" | "socket"


class SlabRegion:
    """A scratch extent inside the store slab: NIC-registered memory
    with torch/numpy views over it and its absolute pointer."""

    def __init__(self, store, nbytes: int):
        self.store = store
        self.nbytes = nbytes
        self.ext = store.alloc_scratch(nbytes)
        view = store.view_extent(self.ext, nbytes)
        self.np = np.frombuffer(view, dtype=np.uint8)
        self.t = torch.from_numpy(self.np)
        self.ptr = store.slab.ptr + self.ext.offset

    def release(self) -> None:
        self.store.release_scratch(self.ext)


class HostmemComm:
    """One per (group, daemon). Requires world == 2 and a live peer
    link (the coordinator star IS the pairwise link at world 2)."""

    def __init__(self, nm, group_name: str, rank: int, world: int,
                 peer_name: str, scratch_bytes: int = 512 << 20):
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
        # recycled ready-events (one live per in-flight op; a fresh
        # Event is minted only when the pool runs dry) — steady-state
        # training reuses the same handful every step
        self.event_pool: deque = deque()
        # phase-time accumulators (seconds + op/byte counts) — the
        # breakdown behind every idle-gap question; coll_bench returns
        # deltas of this dict
        self.stats = {"ops": 0, "bytes": 0, "ev_sync_s": 0.0,
                      "rdy_s": 0.0, "write_s": 0.0, "done_s": 0.0,
                      "sock_send_s": 0.0, "sock_recv_s": 0.0,
                      "reduce_s": 0.0}
        self.inbox: dict[tuple, object] = {}   # (seq, op) -> msg|bytes
        self.inbox_cv = threading.Condition()
        # Staging = TWO slab extents (out: D2H target + reduce accum +
        # H2D source; land: the peer's RDMA_WRITEs arrive here). Slab
        # memory is cudaHostAlloc'd AND inside the NIC's MR — nothing
        # is copied to get bytes NIC-reachable, and nothing is
        # allocated after boot (cudaHostAlloc against a parked device
        # wedges the GPU — findings). Sized once, clamped to slab/8
        # per region; ensure_scratch raises LOUD past it.
        cap = getattr(nm.store.allocator, "capacity", None)
        if cap:
            scratch_bytes = min(scratch_bytes, cap // 8)
        self.out = SlabRegion(nm.store, scratch_bytes)
        self.land = SlabRegion(nm.store, scratch_bytes)
        # device landing buffer (rdma lane): the peer's bytes are
        # H2D'd here and reduced ON DEVICE into the caller's tensor —
        # zero CPU passes, and the reduce stops depending on host
        # memory bandwidth entirely (a CPU add pays 3x traffic per
        # payload byte and contends with the exchange DMA)
        # LANE is a static, SYMMETRIC decision made once at build:
        # both ends advertised rdma in the HELLO iff link.rdma_qp
        # exists on both sides, so both comms reach the same verdict —
        # per-op local guessing raced (one rank on the socket
        # protocol, the peer on rdma = cross-lane deadlock, findings).
        # When rdma is coming, WAIT for the QP pair to reach RTS.
        self.lane = "socket"
        self.dev_land = None
        link = nm.links.get(peer_name)
        qp = getattr(link, "rdma_qp", None) if link is not None else None
        if nm.rdma is not None and qp is not None:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if qp.ready and link.remote_rdma_up.is_set():
                    break
                time.sleep(0.05)
            if qp.ready and link.remote_rdma_up.is_set():
                self.lane = "rdma"
                self.dev_land = torch.empty(scratch_bytes,
                                            dtype=torch.uint8,
                                            device="cuda")
            else:
                # advertised but down (probe demotion / bring-up
                # failure): SYMMETRIC by construction — the peer sees
                # the same not-ready state — so both sides take the
                # socket lane. Loud, never fatal.
                nm.state.emit("group_lane_fallback", name=group_name,
                              peer_id=peer_name, lane="socket")
        if self.lane == "rdma":
            self.warm_device_reduce()
        # release flag: a DEDICATED cudaHostAlloc(Mapped) word — NOT a
        # torch pinned-pool tensor (pool blocks are not reliably
        # device-mappable; a second comm in one process hit exactly
        # that). If mapping is still refused, degrade to the blocking
        # spin fallback instead of failing group bring-up.
        err, host_ptr = cudart.cudaHostAlloc(
            4, cudart.cudaHostAllocMapped)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostAlloc(flag) failed: {err}")
        self.flag_host_ptr = int(host_ptr)
        raw = (ctypes.c_int32 * 1).from_address(self.flag_host_ptr)
        self.flag = np.frombuffer(raw, dtype=np.int32)
        self.flag[0] = 0
        self.flag_dptr = None
        err, dptr = cudriver.cuMemHostGetDevicePointer(
            self.flag_host_ptr, 0)
        if int(err) == 0:
            self.flag_dptr = int(dptr)
        self.wait_value_ok = (self.flag_dptr is not None
                              and stream_wait_value_supported())
        self.dead: str | None = None
        self.worker = threading.Thread(target=self.worker_loop,
                                       name=f"nm-coll-{group_name}",
                                       daemon=True)
        self.worker.start()

    def warm_device_reduce(self) -> None:
        """First-launch kernel loads need device quiescence, which a
        waitvalue-parked stream denies (the established trap, now at
        the comm layer): load every kernel the post-park device reduce
        uses — H2D copy, native add_, and the f32-path casts — while
        the device is still unparked at build time."""
        with torch.cuda.stream(self.stream):
            for dt in (torch.bfloat16, torch.float32, torch.float16):
                host = self.land.t[:64].view(dt)
                dev = self.dev_land[:64].view(dt)
                mine = self.dev_land[64:128].view(dt)
                dev.copy_(host, non_blocking=True)
                mine.add_(dev)
                mine.copy_((mine.to(torch.float32)
                            + dev.to(torch.float32)).to(dt))
        self.stream.synchronize()

    # ---------------------------------------------------- link plumbing

    def deliver(self, msg: dict, payload: bytes) -> None:
        """Called by the link READER thread for COLL frames."""
        key = (int(msg["seq"]), msg.get("op", "data"))
        with self.inbox_cv:
            self.inbox[key] = payload if key[1] == "data" else msg
            self.inbox_cv.notify_all()

    def link(self):
        link = self.nm.links.get(self.peer_name)
        if link is None or not link.alive:
            raise RuntimeError(f"peer {self.peer_name} down")
        return link

    def rdma_qp(self):
        """The link's RC QP when the rdma lane is usable, else None."""
        if self.nm.rdma is None:
            return None
        qp = getattr(self.nm.links.get(self.peer_name), "rdma_qp", None)
        return qp if qp is not None and qp.ready else None

    def send_data(self, seq: int, view) -> None:
        self.link().send_frame({"kind": "COLL", "group": self.group,
                                "seq": seq, "op": "data"}, view)

    def send_header(self, seq: int, op: str, **fields) -> None:
        self.link().send_frame({"kind": "COLL", "group": self.group,
                                "seq": seq, "op": op, **fields})

    def await_entry(self, seq: int, op: str, timeout: float = 120.0):
        key = (seq, op)
        with self.inbox_cv:
            ok = self.inbox_cv.wait_for(
                notify_check(self.inbox, key), timeout)
            if not ok:
                raise RuntimeError(
                    f"collective seq {seq}: peer {op} timeout")
            return self.inbox.pop(key)

    def fail(self, why: str) -> None:
        self.dead = why
        self.flag[0] = 2_000_000_000       # release any parked stream
        self.nm.group_error(self.group, why, fan_out=None)

    def close(self) -> None:
        """Idempotent teardown: stop the worker, release parked
        streams, hand the slab extents back."""
        self.dead = self.dead or "shutdown"
        self.flag[0] = 2_000_000_000
        self.jobs.put(None)
        if self.worker.is_alive():
            self.worker.join(timeout=10.0)
        if self.worker.is_alive():
            return   # mid-collective wedge: leak the extents rather
                     # than hand live-referenced slab memory back
        for region in (self.out, self.land):
            if region is not None:
                region.release()
        self.out = self.land = None
        if self.flag_host_ptr is not None:
            cudart.cudaFreeHost(self.flag_host_ptr)
            self.flag_host_ptr = None

    # ---------------------------------------------------- caller side

    def ensure_scratch(self, nbytes: int) -> None:
        if self.out.nbytes < nbytes:
            raise RuntimeError(
                f"collective op of {nbytes} B exceeds the group's slab "
                f"scratch ({self.out.nbytes} B) — raise "
                f"EngineConfig.peer_coll_scratch_mib (and slab size: "
                f"regions are clamped to slab/8)")

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
        stage = self.out.t[:nbytes]
        lane = self.lane if action == "allreduce" else "socket"
        ev = self.event_pool.popleft() if self.event_pool \
            else torch.cuda.Event()
        # PRODUCER CONTRACT: the tensor must already be ordered
        # against this group's stream (training records grads_ready
        # on the compute stream and gh.stream.wait_event's it; tests
        # and benches synchronize their fills). The comm must NOT
        # wait_stream(current) here: recording on the LEGACY default
        # stream captures every other blocking stream — with two
        # parked comm streams in one process that is a deadlock cycle
        # (findings).
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
                    self.enqueue_recv(target, stage, nbytes, lane)
        self.jobs.put(CollJob(seq, ev, (action, tensor.dtype, nbytes,
                                        root), lane))
        if not self.wait_value_ok:
            # fallback: block the CALLER until the worker releases,
            # THEN enqueue the copy-back — correctness identical,
            # overlap lost (documented)
            spin_until(self.flag, seq, self.dead_check)
            if recv_into_tensor:
                target = tensor if out is None else out
                with torch.cuda.stream(self.stream):
                    self.enqueue_recv(target, stage, nbytes, lane)
        return ev   # staging-ready: safe to reuse/free the INPUT once
                    # this fires; the SUM lands via the group stream

    def enqueue_recv(self, target, stage, nbytes: int, lane: str):
        """Stream ops that run AFTER the flag releases. Socket lane:
        the worker reduced on the CPU into stage — copy the sum back.
        rdma lane: the peer's raw bytes sit in the LAND region — H2D
        them and reduce ON DEVICE into the caller's tensor."""
        tn = target.numel()
        flat = target.reshape(-1)
        if lane == "socket":
            flat.copy_(stage.view(target.dtype)[:tn], non_blocking=True)
            return
        theirs_host = self.land.t[:nbytes].view(target.dtype)[:tn]
        theirs = self.dev_land[:nbytes].view(target.dtype)[:tn]
        theirs.copy_(theirs_host, non_blocking=True)
        flat.add_(theirs)

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
                self.event_pool.append(job.ready_event)  # recycle: the
                # worker just synchronized it; nothing records it again
                # until a later enqueue pops it back out
            except Exception as ex:
                self.fail(f"collective failed: {ex}")
                return

    def exchange(self, seq: int, nbytes: int):
        """Swap ``nbytes`` of the out region with the peer; returns a
        uint8 view of THEIR bytes. rdma lane: tell the peer where my
        landing region is (RDY), RDMA_WRITE into theirs, confirm after
        the CQ ack (DONE — RC completion means remote-visible, so the
        socket DONE can never pass the data). Socket lane: one COLL
        data frame each way."""
        stats = self.stats
        stats["ops"] += 1
        stats["bytes"] += nbytes
        qp = self.rdma_qp()
        if qp is not None:
            t0 = time.monotonic()
            self.send_header(seq, "rdy",
                             raddr=self.nm.rdma.slab_base
                             + self.land.ext.offset,
                             rkey=self.nm.rdma.rkey(), nbytes=nbytes)
            rdy = self.await_entry(seq, "rdy")
            t1 = time.monotonic()
            stats["rdy_s"] += t1 - t0
            qp.write(self.out.ptr, int(rdy["raddr"]), int(rdy["rkey"]),
                     nbytes)
            t2 = time.monotonic()
            stats["write_s"] += t2 - t1
            self.send_header(seq, "done")
            self.await_entry(seq, "done")
            stats["done_s"] += time.monotonic() - t2
            return self.land.t[:nbytes]
        t0 = time.monotonic()
        self.send_data(seq, self.out.np[:nbytes])
        t1 = time.monotonic()
        stats["sock_send_s"] += t1 - t0
        peer = self.await_entry(seq, "data")
        stats["sock_recv_s"] += time.monotonic() - t1
        return torch.frombuffer(peer, dtype=torch.uint8)

    def reduce_into_stage(self, stage_bytes, theirs_bytes, dtype,
                          lo: int = 0) -> None:
        """stage += theirs, native dtype: ONE in-place pass (world-2
        add commutes => replicas bitwise identical)."""
        stage_bytes.view(dtype).add_(theirs_bytes.view(dtype))

    def run_job(self, job: CollJob) -> None:
        action, dtype, nbytes, root = job.action
        t0 = time.monotonic()
        job.ready_event.synchronize()           # D2H landed
        self.stats["ev_sync_s"] += time.monotonic() - t0
        stage = self.out.t[:nbytes]
        if action == "allreduce":
            if job.lane == "rdma":
                qp = self.rdma_qp()
                if qp is None:
                    raise RuntimeError("rdma lane vanished mid-op")
                self.exchange(job.seq, nbytes)   # bytes land; the
                return                           # parked stream does
                                                 # the DEVICE reduce
            theirs = self.exchange(job.seq, nbytes)
            tr = time.monotonic()
            self.reduce_into_stage(stage, theirs, dtype)
            self.stats["reduce_s"] += time.monotonic() - tr
        elif action == "broadcast":
            if self.rank == root:
                self.send_data(job.seq, self.out.np[:nbytes])
            else:
                peer = self.await_entry(job.seq, "data")
                stage.copy_(torch.frombuffer(peer, dtype=torch.uint8))
        elif action == "reduce":
            if self.rank != root:
                self.send_data(job.seq, self.out.np[:nbytes])
            else:
                peer = self.await_entry(job.seq, "data")
                self.reduce_into_stage(
                    stage, torch.frombuffer(peer, dtype=torch.uint8),
                    dtype)
        elif action == "reduce_scatter":
            half = nbytes // 2
            peer_lo = half if self.rank == 0 else 0
            own_lo = 0 if self.rank == 0 else half
            self.send_data(job.seq, self.out.np[peer_lo:peer_lo + half])
            peer = self.await_entry(job.seq, "data")
            own = self.out.t[own_lo:own_lo + half]
            self.reduce_into_stage(
                own, torch.frombuffer(peer, dtype=torch.uint8), dtype)
            if own_lo != 0:
                stage[:half].copy_(own)
        elif action == "all_gather":
            half = nbytes // 2
            own_lo = 0 if self.rank == 0 else half
            self.send_data(job.seq, self.out.np[own_lo:own_lo + half])
            peer = self.await_entry(job.seq, "data")
            peer_lo = half if self.rank == 0 else 0
            self.out.t[peer_lo:peer_lo + half].copy_(
                torch.frombuffer(peer, dtype=torch.uint8))
        else:
            raise ValueError(action)

    # ---------------------------------------------------- GroupHandle ops

    def allreduce(self, tensor, out=None):
        """Returns the staging-ready event (input readable/reusable
        once it fires). With ``out``, the SUM lands in ``out`` instead
        of overwriting ``tensor`` — both stream-ordered on gh.stream."""
        return self.enqueue(tensor, "allreduce", out=out)

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
        stage = self.out.t[:nbytes]
        ev = self.event_pool.popleft() if self.event_pool \
            else torch.cuda.Event()
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
                                        None), "socket"))
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
        stage = self.out.t[:nbytes]
        ev = self.event_pool.popleft() if self.event_pool \
            else torch.cuda.Event()
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
        self.jobs.put(CollJob(seq, ev, ("all_gather", full.dtype,
                                        nbytes, None), "socket"))
        if not self.wait_value_ok:
            spin_until(self.flag, seq, self.dead_check)
            with torch.cuda.stream(self.stream):
                full.reshape(-1).copy_(
                    stage.view(full.dtype)[:full.numel()],
                    non_blocking=True)


class notify_check:
    def __init__(self, inbox: dict, key: tuple):
        self.inbox = inbox
        self.key = key

    def __call__(self) -> bool:
        return self.key in self.inbox


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
    scratch = getattr(nm.server.config, "peer_coll_scratch_mib", 512) << 20
    return HostmemComm(nm, rec.name, rec.self_rank, len(rec.members),
                       peers[0], scratch_bytes=scratch)


def build_handle(nm, rec) -> GroupHandle:
    comm = None
    stream = None
    backend = rec.backend if rec.backend != "auto" else "hostmem"
    try:
        if backend == "nccl":
            # built at BOOTSTRAP (collective init cannot be lazy);
            # missing here means the join-time bring-up failed
            comm = rec.comm_obj
            if comm is None:
                raise RuntimeError("nccl comm missing — group "
                                   "bootstrap did not complete")
            stream = comm.stream
        elif backend == "hostmem":
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
