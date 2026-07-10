"""The NetworkManager subsystem: PeerCore state machines pumped by real
sockets (spec §6). Shape: one NM STATE OBJECT under ONE lock, plus its
threads — a peer accept loop on the daemon's OWN port, one blocking
reader thread per connected peer (the house thread-per-socket idiom),
and one housekeeping thread (core ticks, heartbeats, liveness). P1
carries control + payload on a single ordered socket per link (the
payload-pair split arrives with the rdma work, where the planes
genuinely separate); chunks land ZERO-COPY — the reader recv_into's the
reservation's view over the FINAL store extent.

Store discipline: the NM reserves/releases extents through the store's
LOCKED reservation API (it is the store's second writer); catalog
commits are QUEUED to the dispatcher as internal calls (single-writer
catalog preserved) and the DONE_ACK waits for the commit to land
(PeerCore.commit_done). Events ride the service ring.
"""
from __future__ import annotations

import json
import socket
import threading
import time

from ..wire import Conn, ServiceError
from . import protocol as P
from .core import PeerCore, ReceiverEnv, Reservation

HELLO_SCHEMA = "dataflow-peer/s2"
PING_EVERY_S = 2.0
PEER_DOWN_AFTER_S = 6.0
HOUSEKEEP_EVERY_S = 0.25
DRAIN_SCRATCH = bytearray(1 << 20)


class InternalReply:
    """A reply_to shim for NM-originated dispatcher calls: routes
    push_call_done into a callback instead of a client socket."""

    def __init__(self, on_done):
        self.on_done = on_done

    def push_call_done(self, ticket, result=None, error=None):
        self.on_done(result, error)


class StoreReceiverEnv(ReceiverEnv):
    """ReceiverEnv over the real store + dispatcher (per peer link)."""

    def __init__(self, nm: "NetworkManager", peer_id: str):
        self.nm = nm
        self.peer_id = peer_id

    def try_reserve(self, dest_id, nbytes, overwrite):
        with self.nm.quota_lock:
            held = self.nm.inflight_bytes.get(self.peer_id, 0)
            if held + nbytes > self.nm.inflight_quota:
                return None, "CAPACITY"
            self.nm.inflight_bytes[self.peer_id] = held + nbytes
        ext, code = self.nm.store.reserve_inbound(dest_id, nbytes,
                                                  overwrite=overwrite)
        if ext is None:
            with self.nm.quota_lock:
                self.nm.inflight_bytes[self.peer_id] -= nbytes
            return None, code
        view = self.nm.store.view_extent(ext, nbytes)
        res = Reservation(dest_id=dest_id, buffer=view, extent=ext)
        link = self.nm.links.get(self.peer_id)
        qp = getattr(link, "rdma_qp", None) if link else None
        if self.nm.rdma is not None and qp is not None and qp.ready:
            res.raddr = self.nm.rdma.slab_base + ext.offset
            res.rkey = self.nm.rdma.rkey()
        return res, None

    def commit(self, res, meta, send_id):
        self.nm.queue_commit(self.peer_id, res, meta, send_id)
        return False                          # DONE_ACK on commit_done

    def release(self, res):
        self.nm.store.release_inbound(res.extent)
        with self.nm.quota_lock:
            held = self.nm.inflight_bytes.get(self.peer_id, 0)
            self.nm.inflight_bytes[self.peer_id] = \
                max(0, held - len(res.buffer))


class PeerLink:
    def __init__(self, peer_id: str, conn: Conn, core: PeerCore):
        self.peer_id = peer_id
        self.conn = conn
        self.core = core
        self.wlock = threading.Lock()
        self.last_seen = time.monotonic()
        self.last_ping = 0.0
        self.alive = True
        self.reader: threading.Thread | None = None
        self.dialer = False
        # measured peak goodput per data plane ({"socket": g, "rdma": g}
        # in Gbit/s), probed at connect time — transfer-time gates and
        # observability read this instead of hardcoding link speeds
        self.peak_gbps: dict = {}
        self.bwprobe_t0 = None
        self.bwprobe_bytes = 0
        self.bwprobe_ext = None
        self.bw_ack = threading.Event()
        self.bw_rdy = threading.Event()
        self.bw_rdy_msg: dict | None = None

    def send_frame(self, msg, payload=None):
        try:
            with self.wlock:
                self.conn.send(msg, payload)
        except OSError:
            pass                              # reader notices the death


class LinkSender:
    """The PeerCore ``send`` callable for one link (named class — no
    closures per style)."""

    def __init__(self, link: PeerLink):
        self.link = link

    def __call__(self, msg, payload=None):
        self.link.send_frame(msg, payload)


class NetworkManager:
    def __init__(self, server, *, peer_name: str, listen: str | None,
                 chunk_bytes: int = P.CHUNK_BYTES_DEFAULT,
                 inflight_quota: int = 4 << 30,
                 ping_every_s: float = PING_EVERY_S,
                 peer_down_after_s: float = PEER_DOWN_AFTER_S,
                 rdma_device: str | None = None):
        self.server = server
        self.store = server.store
        self.state = server.state
        self.peer_name = peer_name
        self.listen_addr = listen
        self.chunk_bytes = chunk_bytes
        self.rdma = None
        if rdma_device and self.store.slab is not None:
            from .rdma import RdmaEngine

            self.rdma = RdmaEngine(rdma_device)
            self.rdma.register_slab(self.store.slab.ptr,
                                    self.store.allocator.capacity)
        self.send_src_ptrs: dict[str, int] = {}   # send_id -> slab ptr
        from .groups import GroupTable

        self.groups = GroupTable()
        self.inflight_quota = inflight_quota
        self.ping_every_s = ping_every_s
        self.peer_down_after_s = peer_down_after_s
        self.lock = threading.RLock()         # cores + links + transfers
        self.quota_lock = threading.Lock()
        self.links: dict[str, PeerLink] = {}
        self.transfers: dict[str, dict] = {}  # send_id -> status row
        self.inflight_bytes: dict[str, int] = {}
        self.on_send_finish: dict[str, object] = {}   # send_id -> callable
        self.listener: socket.socket | None = None
        self.stop_flag = threading.Event()
        self.threads: list[threading.Thread] = []

    # ------------------------------------------------ lifecycle

    def start(self) -> None:
        if self.listen_addr:
            host, port = self.listen_addr.rsplit(":", 1)
            self.listener = socket.socket(socket.AF_INET,
                                          socket.SOCK_STREAM)
            self.listener.setsockopt(socket.SOL_SOCKET,
                                     socket.SO_REUSEADDR, 1)
            self.listener.bind((host, int(port)))
            self.listener.listen(8)
            self.listener.settimeout(0.2)
            t = threading.Thread(target=self.accept_loop,
                                 name="nm-accept", daemon=True)
            t.start()
            self.threads.append(t)
        t = threading.Thread(target=self.housekeeping_loop,
                             name="nm-housekeeping", daemon=True)
        t.start()
        self.threads.append(t)

    def stop(self) -> None:
        self.stop_flag.set()
        for rec in self.groups.ready_records():
            handle = rec.handle
            if handle is not None and handle.comm is not None:
                handle.comm.close()            # releases parked streams,
                                               # worker, slab scratch
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass
        with self.lock:
            links = list(self.links.values())
        for link in links:
            self.drop_link(link, why="shutdown")

    # ------------------------------------------------ connect / accept

    def accept_loop(self) -> None:
        while not self.stop_flag.is_set():
            try:
                sock, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                conn = Conn(sock)
                frame = conn.recv()
                msg = frame.msg if frame else None
                if not msg or msg.get("kind") != "HELLO" \
                        or msg.get("schema") != HELLO_SCHEMA:
                    sock.close()
                    continue
                peer_id = msg["peer_id"]
                conn.send({"kind": "HELLO", "schema": HELLO_SCHEMA,
                           "peer_id": self.peer_name,
                           "rdma": self.rdma is not None})
                self.adopt_link(peer_id, conn,
                                peer_rdma=bool(msg.get("rdma")))
            except OSError:
                continue

    def connect(self, peer_id: str, addr: str) -> dict:
        with self.lock:
            if peer_id in self.links:
                return {"peer_id": peer_id, "already": True}
        host, port = addr.rsplit(":", 1)
        sock = socket.create_connection((host, int(port)), timeout=10)
        conn = Conn(sock)
        conn.send({"kind": "HELLO", "schema": HELLO_SCHEMA,
                   "peer_id": self.peer_name,
                   "rdma": self.rdma is not None})
        frame = conn.recv()
        msg = frame.msg if frame else None
        if not msg or msg.get("kind") != "HELLO":
            sock.close()
            raise ServiceError("PEER_UNREACHABLE",
                               f"bad hello from {addr}")
        actual = msg.get("peer_id", peer_id)
        self.adopt_link(actual, conn, peer_rdma=bool(msg.get("rdma")),
                        dialer=True)
        with self.lock:
            link = self.links.get(actual)
        if link is not None and link.dialer:
            self.bw_probe_socket(link)
        return {"peer_id": actual}

    def adopt_link(self, peer_id: str, conn: Conn, *,
                   peer_rdma: bool = False, dialer: bool = False) -> None:
        with self.lock:
            old = self.links.get(peer_id)
            if old is not None:
                # connect glare: keep the smaller-name dialer's link
                if self.peer_name < peer_id:
                    conn.sock.close()
                    return
                self.drop_link(old, why="glare-replaced")
            env = StoreReceiverEnv(self, peer_id)
            link = PeerLink(peer_id, conn, core=None)
            link.dialer = dialer
            core = PeerCore(env, LinkSender(link), time.monotonic,
                            chunk_bytes=self.chunk_bytes)
            link.core = core
            self.links[peer_id] = link
            if self.rdma is not None and peer_rdma:
                link.rdma_qp = self.rdma.make_link_qp()
            reader = threading.Thread(target=self.reader_loop,
                                      args=(link,),
                                      name=f"nm-link-{peer_id}",
                                      daemon=True)
            link.reader = reader
        self.state.emit("peer_up", peer_id=peer_id)
        reader.start()
        if getattr(link, "rdma_qp", None) is not None:
            link.send_frame({"kind": "RDMA_INFO",
                             **link.rdma_qp.local_info()})

    def drop_link(self, link: PeerLink, *, why: str) -> None:
        with self.lock:
            if not link.alive:
                return
            link.alive = False
            self.links.pop(link.peer_id, None)
            core = link.core
            for machine in list(core.receivers.values()):
                core.abort_receiver(machine, f"peer_down ({why})")
            for machine in list(core.senders.values()):
                core.finish_sender(machine, P.TransferState.ERROR,
                                   f"TRANSFER_ABORTED peer_down ({why})")
        try:
            link.conn.sock.close()
        except OSError:
            pass
        # groups containing the dead peer desync (fail-stop)
        for rec in self.groups.ready_records():
            if link.peer_id in rec.members:
                self.group_error(rec.name,
                                 f"peer_down {link.peer_id} ({why})",
                                 fan_out=link.peer_id)
        self.state.emit("peer_down", peer_id=link.peer_id, why=why)

    def record_bw(self, link: PeerLink, plane: str, gbps: float) -> None:
        link.peak_gbps[plane] = round(gbps, 2)
        self.state.emit("peer_bw", peer_id=link.peer_id, plane=plane,
                        gbps=round(gbps, 2))

    def bw_probe_socket(self, link: PeerLink) -> None:
        """Measure the socket data plane at connect time: blast
        peer_bw_probe_mib through the normal frame path, the receiver
        times first-chunk to END and ACKs its goodput back. Failure
        never harms the link (peak_gbps just stays unset)."""
        mib = getattr(self.server.config, "peer_bw_probe_mib", 128)
        if mib <= 0:
            return
        chunk = memoryview(bytes(4 << 20))
        total = mib << 20
        link.bw_ack.clear()
        try:
            sent = 0
            while sent < total:
                link.send_frame({"kind": "BWPROBE"}, chunk)
                sent += len(chunk)
            link.send_frame({"kind": "BWPROBE_END", "bytes": total})
            link.bw_ack.wait(15.0)
        except Exception:
            pass

    def create_group(self, name: str, members: list, backend: str,
                     *, reduce_dtype: str = "native",
                     timeout: float = 20.0) -> dict:
        """Coordinator side: THIS daemon must be rank 0 of members and
        hold star links to every other member."""
        from .groups import GroupRecord

        if members[0] != self.peer_name:
            raise ServiceError("BAD_REQUEST",
                               "create_peer_group must land on rank 0")
        with self.lock:
            missing = [m for m in members[1:] if m not in self.links]
            if missing:
                raise ServiceError("PEER_UNREACHABLE",
                                   f"star links missing: {missing}")
        rec = GroupRecord(name=name, members=tuple(members),
                          backend=backend, self_rank=0,
                          coordinator=self.peer_name,
                          reduce_dtype=reduce_dtype)
        try:
            self.groups.create(rec)
        except ValueError:
            raise ServiceError("GROUP_EXISTS", name)
        join = {"kind": "GROUP_JOIN", "name": name,
                "members": list(members), "backend": backend,
                "reduce_dtype": reduce_dtype}
        with self.lock:
            for m in members[1:]:
                self.links[m].send_frame(join)
        if len(members) > 1 and not self.groups.wait_ready(name, timeout):
            self.groups.mark_error(name, "join barrier timeout")
            raise ServiceError("GROUP_DESYNC",
                               f"{name}: join barrier timeout")
        if len(members) == 1:
            self.groups.adopt(rec)
        self.state.emit("group_created", name=name)
        return {"ok": True, "backend": backend, "world": len(members)}

    def group_handles(self) -> dict:
        """{name -> GroupHandle} for TaskContext injection: READY,
        non-errored groups; comm backends built + cached lazily."""
        out = {}
        for rec in self.groups.ready_records():
            if rec.handle is None:
                from .comm import build_handle

                rec.handle = build_handle(self, rec)
            out[rec.name] = rec.handle
        return out

    def comm_of(self, name: str):
        with self.groups.lock:
            rec = self.groups.groups.get(name)
        if rec is None or not rec.ready or rec.error is not None:
            return None
        if rec.handle is None:
            from .comm import build_handle

            rec.handle = build_handle(self, rec)
        return rec.handle.comm

    def group_error(self, name: str, why: str, *,
                    fan_out: str | None) -> None:
        """Spec §7 two-hop fan-out: a member reports to its
        coordinator; the coordinator rebroadcasts over the star."""
        self.groups.mark_error(name, why)
        with self.groups.lock:
            rec = self.groups.groups.get(name)
            handle = rec.handle if rec is not None else None
        if handle is not None and handle.comm is not None \
                and handle.comm.dead is None:
            handle.comm.dead = why
            handle.comm.flag[0] = 2_000_000_000   # release parked streams
        self.state.emit("group_error", name=name, why=why)
        with self.groups.lock:
            rec = self.groups.groups.get(name)
        if rec is None:
            return
        if rec.self_rank == 0:                 # coordinator: fan out
            with self.lock:
                for m in rec.members[1:]:
                    if m != fan_out and m in self.links:
                        self.links[m].send_frame(
                            {"kind": "GROUP_ERROR", "name": name,
                             "why": why})
        elif fan_out is None:                  # local failure: tell coord
            with self.lock:
                link = self.links.get(rec.coordinator)
            if link is not None:
                link.send_frame({"kind": "GROUP_ERROR", "name": name,
                                 "why": why})

    def debug_sever(self, peer_id: str) -> bool:
        """Test hook: kill the socket WITHOUT cleanup — the remote side
        must discover the death itself (EOF/heartbeat)."""
        with self.lock:
            link = self.links.get(peer_id)
        if link is None:
            return False
        try:
            link.conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        return True

    # ------------------------------------------------ reader (per link)

    def reader_loop(self, link: PeerLink) -> None:
        sock = link.conn.sock
        buf = b""
        while link.alive and not self.stop_flag.is_set():
            # read one JSON header line
            while b"\n" not in buf:
                try:
                    got = sock.recv(65536)
                except OSError:
                    got = b""
                if not got:
                    self.drop_link(link, why="eof")
                    return
                buf += got
            line, buf = buf.split(b"\n", 1)
            try:
                msg = json.loads(line)
            except ValueError:
                self.drop_link(link, why="bad frame")
                return
            link.last_seen = time.monotonic()
            n = int(msg.pop("payload_bytes", 0))
            payload = None
            if msg.get("kind") == "CHUNK" and n:
                buf = self.land_chunk(link, msg, n, buf, sock)
                if buf is None:
                    return
                continue
            if n:
                # O(n) assembly (a bytes-concat loop here was O(n^2):
                # one 206 MB COLL frame stalled the reader for seconds,
                # heartbeats starved, and the peer dropped the link —
                # the 1B flagship's first failure; findings, P4)
                body = bytearray(n)
                take = min(len(buf), n)
                if take:
                    body[:take] = buf[:take]
                    buf = buf[take:]
                got_total = take
                view = memoryview(body)
                while got_total < n:
                    try:
                        got = sock.recv_into(view[got_total:],
                                             min(1 << 22, n - got_total))
                    except OSError:
                        got = 0
                    if not got:
                        self.drop_link(link, why="eof mid-payload")
                        return
                    got_total += got
                    link.last_seen = time.monotonic()
                payload = bytes(body)
            kind = msg.get("kind")
            if kind == "PING":
                link.send_frame({"kind": "PONG"})
                continue
            if kind == "PONG":
                continue
            if kind == "RDMA_INFO":
                qp = getattr(link, "rdma_qp", None)
                if qp is not None and not qp.ready:
                    qp.connect(msg)
                    with self.lock:
                        link.core.rdma_writer = RdmaWriterHook(self, link)
                    self.state.emit("peer_rdma_up", peer_id=link.peer_id)
                    if link.dialer:
                        threading.Thread(
                            target=BwProbeRdma(self, link),
                            name=f"nm-bwprobe-{link.peer_id}",
                            daemon=True).start()
                continue
            if kind == "GROUP_JOIN":
                from .groups import GroupRecord

                members = tuple(msg["members"])
                rec = GroupRecord(
                    name=msg["name"], members=members,
                    backend=msg["backend"],
                    self_rank=members.index(self.peer_name),
                    coordinator=link.peer_id,
                    reduce_dtype=msg.get("reduce_dtype", "native"))
                self.groups.adopt(rec)
                link.send_frame({"kind": "GROUP_ACK", "name": msg["name"],
                                 "member": self.peer_name})
                self.state.emit("group_joined", name=msg["name"],
                                rank=rec.self_rank)
                continue
            if kind == "GROUP_ACK":
                if self.groups.ack(msg["name"], msg["member"]):
                    self.state.emit("group_created", name=msg["name"])
                continue
            if kind == "GROUP_ERROR":
                self.group_error(msg["name"], msg.get("why", "?"),
                                 fan_out=getattr(link, "peer_id", None))
                continue
            if kind == "BWPROBE":
                now = time.monotonic()
                if link.bwprobe_t0 is None:
                    link.bwprobe_t0 = now
                link.bwprobe_bytes += len(payload or b"")
                continue
            if kind == "BWPROBE_END":
                dt = max(time.monotonic()
                         - (link.bwprobe_t0 or time.monotonic()), 1e-6)
                gbps = link.bwprobe_bytes * 8 / dt / 1e9
                link.bwprobe_t0 = None
                link.bwprobe_bytes = 0
                self.record_bw(link, "socket", gbps)
                link.send_frame({"kind": "BWPROBE_ACK",
                                 "plane": "socket",
                                 "gbps": round(gbps, 2)})
                continue
            if kind == "BWPROBE_ACK":
                self.record_bw(link, msg.get("plane", "socket"),
                               float(msg["gbps"]))
                link.bw_ack.set()
                continue
            if kind == "BWPROBE_RDMA_REQ":
                n = int(msg["bytes"])
                try:
                    ext = self.store.alloc_scratch(n)
                    link.bwprobe_ext = ext
                    link.send_frame({"kind": "BWPROBE_RDMA_RDY",
                                     "raddr": self.rdma.slab_base
                                     + ext.offset,
                                     "rkey": self.rdma.rkey(),
                                     "bytes": n})
                except Exception as ex:
                    link.send_frame({"kind": "BWPROBE_RDMA_RDY",
                                     "error": str(ex)})
                continue
            if kind == "BWPROBE_RDMA_RDY":
                link.bw_rdy_msg = msg
                link.bw_rdy.set()
                continue
            if kind == "BWPROBE_RDMA_DONE":
                if link.bwprobe_ext is not None:
                    self.store.release_scratch(link.bwprobe_ext)
                    link.bwprobe_ext = None
                self.record_bw(link, "rdma", float(msg["gbps"]))
                continue
            if kind == "COLL":
                comm = self.comm_of(msg["group"])
                if comm is not None:
                    comm.deliver(msg, payload or b"")
                continue
            with self.lock:
                link.core.handle(msg, payload)
            self.after_core_step(link)

    def land_chunk(self, link: PeerLink, msg: dict, n: int, buf: bytes,
                   sock) -> bytes | None:
        """Zero-copy landing: recv_into the reservation's view over the
        FINAL store extent. Returns the remaining stream buffer, or None
        if the link died."""
        with self.lock:
            machine = link.core.receivers.get(msg["send_id"])
            ok = (machine is not None
                  and msg.get("seq") == machine.next_seq
                  and not machine.committing
                  and machine.received + n <= machine.expect_bytes)
            view = None
            if ok:
                view = machine.res.buffer[
                    machine.received:machine.received + n]
        if view is not None:
            take = min(len(buf), n)
            if take:
                view[:take] = buf[:take]
                buf = buf[take:]
            got_total = take
            while got_total < n:
                try:
                    got = sock.recv_into(view[got_total:], n - got_total)
                except OSError:
                    got = 0
                if not got:
                    self.drop_link(link, why="eof mid-chunk")
                    return None
                got_total += got
                link.last_seen = time.monotonic()
            msg["landed"] = n
            with self.lock:
                link.core.handle(msg, None)
        else:
            # unknown/aborted/mis-sequenced transfer: drain + deliver
            # the bare header (the core aborts on the seq violation)
            remaining = n - len(buf)
            if remaining <= 0:
                buf = buf[n:]
            else:
                buf = b""
                while remaining > 0:
                    take = min(remaining, len(DRAIN_SCRATCH))
                    try:
                        got = sock.recv_into(
                            memoryview(DRAIN_SCRATCH)[:take], take)
                    except OSError:
                        got = 0
                    if not got:
                        self.drop_link(link, why="eof mid-drain")
                        return None
                    remaining -= got
            with self.lock:
                link.core.handle(msg, b"")
        self.after_core_step(link)
        return buf

    # ------------------------------------------------ housekeeping

    def housekeeping_loop(self) -> None:
        while not self.stop_flag.is_set():
            self.stop_flag.wait(HOUSEKEEP_EVERY_S)
            now = time.monotonic()
            with self.lock:
                links = list(self.links.values())
            for link in links:
                if not link.alive:
                    continue
                if now - link.last_ping >= self.ping_every_s:
                    link.last_ping = now
                    link.send_frame({"kind": "PING"})
                if now - link.last_seen >= self.peer_down_after_s:
                    self.drop_link(link, why="heartbeat timeout")
                    continue
                with self.lock:
                    link.core.tick()
                self.after_core_step(link)

    # ------------------------------------------------ outbound + status

    def start_send(self, peer_id: str, dest_id: str, payload_view, *,
                   overwrite: bool, meta: dict, on_finish,
                   src_ptr: int | None = None) -> str:
        with self.lock:
            link = self.links.get(peer_id)
            if link is None or not link.alive:
                raise ServiceError("PEER_UNREACHABLE", peer_id)
            ticket = link.core.start_send(
                dest_id, payload_view, overwrite=overwrite, meta=meta,
                on_finish=SendFinishRelay(self, peer_id, on_finish))
            if src_ptr is not None:
                self.send_src_ptrs[ticket.send_id] = src_ptr
            self.transfers[ticket.send_id] = {
                "send_id": ticket.send_id, "peer_id": peer_id,
                "dest_id": dest_id, "state": ticket.state.value,
                "bytes_total": ticket.bytes_total, "bytes_done": 0,
                "error": None,
            }
        self.state.emit("transfer_started", send_id=ticket.send_id,
                        peer_id=peer_id, dest_id=dest_id,
                        bytes=ticket.bytes_total)
        return ticket.send_id

    def transfer_status(self, send_id: str) -> dict:
        with self.lock:
            row = self.transfers.get(send_id)
            if row is None:
                raise ServiceError("UNKNOWN_OBJECT", send_id)
            return dict(row)

    def refresh_transfer(self, ticket) -> None:
        with self.lock:
            row = self.transfers.get(ticket.send_id)
            if row is not None:
                row["state"] = ticket.state.value
                row["bytes_done"] = ticket.bytes_done
                row["error"] = ticket.error

    def after_core_step(self, link: PeerLink) -> None:
        with self.lock:
            for send_id, machine in link.core.senders.items():
                row = self.transfers.get(send_id)
                if row is not None:
                    row["state"] = machine.ticket.state.value
                    row["bytes_done"] = machine.ticket.bytes_done

    # ------------------------------------------------ commit plumbing

    def queue_commit(self, peer_id: str, res, meta: dict,
                     send_id: str) -> None:
        from ..server import QueuedCall

        args = {"dest_id": res.dest_id, "extent": res.extent,
                "size_bytes": len(res.buffer), "meta": meta,
                "from_peer": peer_id,
                "landed_zero_copy": res.landed_zero_copy}
        relay = CommitRelay(self, peer_id, send_id, res)
        call = QueuedCall(ticket=f"peer-commit-{send_id}",
                          session_id="peer", op="peer_commit_inbound",
                          args=args, payload=None,
                          reply_to=InternalReply(relay))
        self.server.dispatcher.submit(call)

    def commit_landed(self, peer_id: str, send_id: str, res,
                      result, error) -> None:
        with self.lock:
            link = self.links.get(peer_id)
        if error is not None:
            # commit refused (e.g. leased since reserve): reservation
            # was already freed by the handler; abort loud
            if link is not None:
                with self.lock:
                    machine = link.core.receivers.pop(send_id, None)
            self.state.emit("transfer_error", send_id=send_id,
                            peer_id=peer_id, dest_id=res.dest_id,
                            error=str(error))
            return
        with self.quota_lock:
            held = self.inflight_bytes.get(peer_id, 0)
            self.inflight_bytes[peer_id] = max(0, held - len(res.buffer))
        if link is not None:
            with self.lock:
                link.core.commit_done(send_id)
        self.state.emit("object_received", oid=res.dest_id,
                        from_peer=peer_id, bytes=len(res.buffer),
                        zero_copy=res.landed_zero_copy)


class BwProbeRdma:
    """Dialer-side rdma-plane probe (own short thread — never the
    reader's): REQ a landing extent, one-sided RDMA_WRITE of the probe
    size from slab scratch, report the measured goodput both ways."""

    def __init__(self, nm: "NetworkManager", link: PeerLink):
        self.nm = nm
        self.link = link

    def __call__(self) -> None:
        nm, link = self.nm, self.link
        mib = getattr(nm.server.config, "peer_bw_probe_mib", 128)
        if mib <= 0 or nm.rdma is None:
            return
        n = mib << 20
        src = None
        try:
            link.bw_rdy.clear()
            link.send_frame({"kind": "BWPROBE_RDMA_REQ", "bytes": n})
            if not link.bw_rdy.wait(15.0):
                return
            rdy = link.bw_rdy_msg or {}
            if rdy.get("error") or link.rdma_qp is None:
                return
            src = nm.store.alloc_scratch(n)
            t0 = time.monotonic()
            link.rdma_qp.write(nm.store.slab.ptr + src.offset,
                               int(rdy["raddr"]), int(rdy["rkey"]), n)
            dt = max(time.monotonic() - t0, 1e-6)
            gbps = n * 8 / dt / 1e9
            nm.record_bw(link, "rdma", gbps)
            link.send_frame({"kind": "BWPROBE_RDMA_DONE",
                             "gbps": round(gbps, 2)})
        except Exception:
            pass
        finally:
            if src is not None:
                nm.store.release_scratch(src)


class RdmaWriterHook:
    """core.rdma_writer: spawns the writer thread for one transfer —
    the reader thread must never block on multi-second CQ polls."""

    def __init__(self, nm: NetworkManager, link: PeerLink):
        self.nm = nm
        self.link = link

    def __call__(self, machine, raddr: int, rkey: int) -> None:
        t = threading.Thread(
            target=self.run, args=(machine, raddr, rkey),
            name=f"nm-rdma-write-{machine.ticket.send_id}", daemon=True)
        t.start()

    def run(self, machine, raddr: int, rkey: int) -> None:
        nm, link = self.nm, self.link
        send_id = machine.ticket.send_id
        src_ptr = nm.send_src_ptrs.pop(send_id, None)
        try:
            if src_ptr is None:
                raise RuntimeError("rdma send without a slab src_ptr")
            link.rdma_qp.write(src_ptr, raddr, rkey,
                               machine.ticket.bytes_total,
                               on_progress=TicketProgress(machine.ticket))
        except Exception as ex:                # fail-stop: kill the link
            with nm.lock:
                link.core.finish_sender(
                    machine, P.TransferState.ERROR,
                    f"TRANSFER_ABORTED rdma: {ex}")
            nm.drop_link(link, why=f"rdma write failed: {ex}")
            return
        with nm.lock:
            link.core.rdma_write_finished(machine)
        nm.after_core_step(link)


class TicketProgress:
    def __init__(self, ticket):
        self.ticket = ticket

    def __call__(self, done: int) -> None:
        self.ticket.bytes_done = done


class SendFinishRelay:
    """on_finish hook: update status row, emit events, run the verb's
    completion (lease release) — named class per style."""

    def __init__(self, nm: NetworkManager, peer_id: str, after):
        self.nm = nm
        self.peer_id = peer_id
        self.after = after

    def __call__(self, ticket) -> None:
        self.nm.refresh_transfer(ticket)
        if ticket.state.value == "done":
            self.nm.state.emit("transfer_done", send_id=ticket.send_id,
                               peer_id=self.peer_id,
                               dest_id=ticket.dest_id,
                               bytes=ticket.bytes_total)
        else:
            self.nm.state.emit("transfer_error", send_id=ticket.send_id,
                               peer_id=self.peer_id,
                               dest_id=ticket.dest_id,
                               error=ticket.error)
        if self.after is not None:
            self.after(ticket)


class CommitRelay:
    def __init__(self, nm: NetworkManager, peer_id: str, send_id: str,
                 res):
        self.nm = nm
        self.peer_id = peer_id
        self.send_id = send_id
        self.res = res

    def __call__(self, result, error) -> None:
        self.nm.commit_landed(self.peer_id, self.send_id, self.res,
                              result, error)
