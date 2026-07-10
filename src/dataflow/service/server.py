"""Engine service daemon: connection threads + FIFO dispatcher.

Concurrency model (design note Part II):
- one CONNECTION THREAD per client socket: decodes frames, answers
  FAST-PATH ops inline from shared state (under a small lock), routes
  CRITICAL ops immediately, enqueues TYPICAL ops as QueuedCall;
- ONE DISPATCHER thread: pops the FIFO and executes calls strictly
  in order; a run occupies it end-to-end (S1.2);
- the EVENT RING (service events) is appended under the state lock;
  subscriber connections drain their own queues.

S1.0 implements the skeleton + lifecycle/admin/events; the object
store (S1.1) and programs/runs (S1.2) plug into `Dispatcher.execute`
and `EngineState` without changing this file's structure.
"""
from __future__ import annotations

import json
import os
import queue
import socket
import threading
from collections import defaultdict
import time
from dataclasses import dataclass, field

from .wire import SCHEMA_VERSION, Conn, ServiceError

_GIT_REV_CACHE: str | None = None


def _git_rev() -> str:
    global _GIT_REV_CACHE
    if _GIT_REV_CACHE is None:
        try:
            import subprocess

            _GIT_REV_CACHE = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(__file__),
            ).stdout.strip() or "unknown"
        except Exception:
            _GIT_REV_CACHE = "unknown"
    return _GIT_REV_CACHE


@dataclass
class EngineConfig:
    socket_path: str
    slab_backing_gib: float | str = "auto"   # pinned at store boot (S1.1)
    device: int = 0
    kernel_set: str | None = None
    fake: bool = False                       # CPU-only boot (tests)
    # peer plane (dataflow-peer/s2): enabled when peer_name is set;
    # peer_listen = "host:port" for the NM's OWN listener
    peer_name: str | None = None
    peer_listen: str | None = None
    peer_chunk_bytes: int = 128 * 1024 * 1024
    peer_rdma_device: str | None = None      # e.g. "mlx5_1" => rdma-host
    peer_coll_scratch_mib: int = 512         # per-group pinned comm scratch
    peer_bw_probe_mib: int = 128             # connect-time bw probe (0=off)
    host_bw_probe_mib: int = 256             # boot host/PCIe bw probe (0=off)

    def public(self) -> dict:
        return {
            "socket": self.socket_path,
            "slab_backing_gib": self.slab_backing_gib,
            "device": self.device, "kernel_set": self.kernel_set,
            "fake": self.fake, "schema_version": SCHEMA_VERSION,
            "git_rev": _git_rev(),
        }


@dataclass
class QueuedCall:
    ticket: str
    session_id: str
    op: str
    args: dict
    payload: bytes | None
    reply_to: "Connection"


@dataclass
class Session:
    session_id: str
    client_name: str
    connected_t: float
    calls: int = 0
    runs_submitted: int = 0
    run_seconds_total: float = 0.0
    recent_calls: list = field(default_factory=list)   # ring of dicts (cap 256)

    def note_call(self, ticket: str, op: str, status: str) -> None:
        self.calls += 1
        self.recent_calls.append(
            {"ticket": ticket, "op": op, "t": time.time(), "status": status})
        if len(self.recent_calls) > 256:
            del self.recent_calls[:len(self.recent_calls) - 256]


EVENT_RING_CAP = 4096


class EngineState:
    """Shared mutable state; every access under `lock` (small critical
    sections only — never held across I/O or execution)."""

    def __init__(self, config: EngineConfig):
        self.lock = threading.Lock()
        self.config = config
        self.boot_t = time.time()
        self.sessions: dict[str, Session] = {}
        self.counters = {
            "programs_registered": 0, "runs_total": 0, "runs_failed": 0,
            "tasks_executed": 0, "bytes_h2d": 0, "bytes_d2h": 0,
            "api_calls": 0,
        }
        self.queue_depth = 0
        self.current_run: dict | None = None
        self.snapshots_in_flight: list[str] = []
        self.events: list[dict] = []            # service-event ring
        self.event_seq = 0
        self.subscribers: list["Connection"] = []
        self.shutdown_requested = threading.Event()
        self._seq = defaultdict(int)   # c/s/r + snap + future kinds

    # ---- ids ----
    def next_id(self, kind: str) -> str:
        with self.lock:
            self._seq[kind] += 1
            return f"{kind}-{self._seq[kind]:06d}"

    # ---- events ----
    def emit(self, event: str, **payload) -> None:
        with self.lock:
            self.event_seq += 1
            ev = {"event": event, "t": time.time(),
                  "seq": self.event_seq, **payload}
            self.events.append(ev)
            if len(self.events) > EVENT_RING_CAP:
                del self.events[:len(self.events) - EVENT_RING_CAP]
            subs = list(self.subscribers)
        for conn in subs:
            conn.push_event(ev)

    def events_since(self, since_seq: int) -> list[dict]:
        with self.lock:
            return [e for e in self.events if e["seq"] > since_seq]


class Dispatcher(threading.Thread):
    """Executes queued calls strictly in FIFO order."""

    def __init__(self, state: EngineState):
        super().__init__(name="dispatcher", daemon=True)
        self.state = state
        self.fifo: "queue.Queue[QueuedCall | None]" = queue.Queue()
        self.handlers: dict[str, callable] = {}   # op -> fn(call) -> result
        self.cancel_flag = threading.Event()      # consumed by runs (S1.2)
        # LEASED calls park here (no error reply) and retry on
        # lease release. HANDLER RULE: LEASED must be raised BEFORE
        # any handler-visible mutation (parked calls re-run from
        # scratch); run() does a lease pre-pass for this reason.
        self._parked: list[QueuedCall] = []
        self._park_lock = threading.Lock()

    def submit(self, call: QueuedCall) -> None:
        with self.state.lock:
            self.state.queue_depth += 1
        self.fifo.put(call)

    def run(self) -> None:
        while True:
            call = self.fifo.get()
            if call is None:
                return
            with self.state.lock:
                self.state.queue_depth -= 1
            try:
                fn = self.handlers.get(call.op)
                if fn is None:
                    raise ServiceError("UNKNOWN_OP", call.op)
                result = fn(call)
                call.reply_to.push_call_done(call.ticket, result=result)
                status = "ok"
            except ServiceError as e:
                if e.code == "LEASED":
                    with self._park_lock:
                        self._parked.append(call)
                    continue
                call.reply_to.push_call_done(call.ticket, error=e.to_json())
                status = e.code
            except Exception as e:  # noqa: BLE001 — daemon must survive
                err = ServiceError("INTERNAL", f"{type(e).__name__}: {e}")
                call.reply_to.push_call_done(call.ticket, error=err.to_json())
                status = "INTERNAL"
            with self.state.lock:
                sess = self.state.sessions.get(call.session_id)
            if sess is not None:
                sess.note_call(call.ticket, call.op, status)

    def unpark_all(self) -> None:
        with self._park_lock:
            calls, self._parked = self._parked, []
        for c in calls:
            with self.state.lock:
                self.state.queue_depth += 1
            self.fifo.put(c)

    def stop(self) -> None:
        self.fifo.put(None)


class Connection(threading.Thread):
    """One per client socket: framing, fast-path, routing."""

    def __init__(self, server: "Server", sock: socket.socket):
        super().__init__(name="conn", daemon=True)
        self.server = server
        self.state = server.state
        self.conn = Conn(sock)
        self.wlock = threading.Lock()
        self.session_id: str | None = None
        self.subscribed = False

    # ---- pushes (any thread) ----
    def _push(self, msg: dict, payload: bytes | None = None) -> None:
        try:
            with self.wlock:
                self.conn.send(msg, payload)
        except OSError:
            pass  # client gone; reaper below cleans up

    def push_call_done(self, ticket: str, result=None, error=None) -> None:
        msg = {"event": "call_done", "ticket": ticket, "ok": error is None}
        payload = None
        if error is not None:
            msg["error"] = error
        elif isinstance(result, tuple) and len(result) == 2 \
                and isinstance(result[1], (bytes, memoryview)):
            msg["result"], payload = result
        else:
            msg["result"] = result
        self._push(msg, payload)

    def push_event(self, ev: dict) -> None:
        if self.subscribed:
            self._push(ev)

    # ---- main loop ----
    def run(self) -> None:
        try:
            self._serve()
        finally:
            with self.state.lock:
                if self in self.state.subscribers:
                    self.state.subscribers.remove(self)
                if self.session_id and self.session_id in self.state.sessions:
                    del self.state.sessions[self.session_id]
            self.conn.close()

    def _serve(self) -> None:
        # handshake first
        first = self.conn.recv()
        if first is None:
            return
        hello = first.msg
        if hello.get("op") != "hello":
            self._push({"id": hello.get("id"), "ok": False,
                        "error": ServiceError("BAD_REQUEST",
                                              "expected hello").to_json()})
            return
        if hello["args"].get("schema_version") != SCHEMA_VERSION:
            self._push({"id": hello["id"], "ok": False,
                        "error": ServiceError(
                            "SCHEMA_VERSION_SKEW",
                            f"daemon={SCHEMA_VERSION} "
                            f"client={hello['args'].get('schema_version')}",
                        ).to_json()})
            return
        self.session_id = self.state.next_id("s")
        sess = Session(self.session_id,
                       hello["args"].get("client_name", ""), time.time())
        with self.state.lock:
            self.state.sessions[self.session_id] = sess
        self._push({"id": hello["id"], "ok": True, "result": {
            "session_id": self.session_id,
            "engine": self.state.config.public(),
        }})

        while not self.state.shutdown_requested.is_set():
            frame = self.conn.recv()
            if frame is None:
                return
            self._route(frame.msg, frame.payload)

    def _route(self, msg: dict, payload: bytes | None) -> None:
        rid, op, args = msg.get("id"), msg.get("op"), msg.get("args", {})
        with self.state.lock:
            self.state.counters["api_calls"] += 1

        fast = self.server.fast_handlers.get(op)
        if fast is not None:
            try:
                result = fast(self, args)
                self._push({"id": rid, "ok": True, "result": result})
            except ServiceError as e:
                self._push({"id": rid, "ok": False, "error": e.to_json()})
            return

        crit = self.server.critical_handlers.get(op)
        if crit is not None:
            try:
                result = crit(self, args)
                self._push({"id": rid, "ok": True, "result": result})
            except ServiceError as e:
                self._push({"id": rid, "ok": False, "error": e.to_json()})
            return

        if op not in self.server.dispatcher.handlers:
            self._push({"id": rid, "ok": False,
                        "error": ServiceError("UNKNOWN_OP",
                                              str(op)).to_json()})
            return
        ticket = self.state.next_id("c")
        self._push({"id": rid, "ok": True, "ticket": ticket})
        self.server.dispatcher.submit(QueuedCall(
            ticket=ticket, session_id=self.session_id, op=op,
            args=args, payload=payload, reply_to=self))


class Server:
    """Boot, accept loop, handler registration."""

    def __init__(self, config: EngineConfig):
        import os as _os

        # match the bench tooling's allocator policy (reserved tracks
        # allocated; segment slack was the twin study's phantom
        # +1.7 GiB device offset). Must be set before torch's first
        # CUDA allocation in this process.
        _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                               "expandable_segments:True")
        self.config = config
        self.state = EngineState(config)
        self.dispatcher = Dispatcher(self.state)
        self.fast_handlers: dict[str, callable] = {}
        self.critical_handlers: dict[str, callable] = {}
        self._register_core()
        self._sock: socket.socket | None = None
        from . import (handlers_peers, handlers_runs, handlers_snapshot,
                       handlers_store)

        self.store = handlers_store.boot_store(self)
        self.host_bw = {}
        if not config.fake:
            from .hostbw import measure_host_bw

            try:
                self.host_bw = measure_host_bw(
                    getattr(config, "host_bw_probe_mib", 256))
            except Exception:
                self.host_bw = {}
        handlers_store.install(self)
        handlers_runs.install(self)
        handlers_snapshot.install(self)
        handlers_peers.install(self)

    # ---- core handlers (S1.0) ----
    def _register_core(self) -> None:
        st = self.state

        def health(conn, args):
            return {"ok": True, "time": time.time(),
                    "uptime_s": time.time() - st.boot_t}

        def engine_status(conn, args):
            with st.lock:
                return {
                    "uptime_s": time.time() - st.boot_t,
                    "boot_config": {**st.config.public(),
                                    "device_fixed_bytes": getattr(
                                        self, "device_fixed_bytes", 0)},
                    "pools": {
                        "backing": self.server_pools_backing(),
                        "fast": None,          # fast residency: S2
                    },
                    "counters": dict(st.counters),
                    "host_bw_gbs": dict(self.host_bw),
                    "current_run": st.current_run,
                    "queue_depth": st.queue_depth,
                    "sessions": [
                        {"session_id": s.session_id,
                         "client_name": s.client_name,
                         "connected_s": time.time() - s.connected_t}
                        for s in st.sessions.values()],
                    "snapshots_in_flight": list(st.snapshots_in_flight),
                }

        def session_status(conn, args):
            sid = args.get("session_id") or conn.session_id
            with st.lock:
                s = st.sessions.get(sid)
            if s is None:
                raise ServiceError("BAD_REQUEST", f"unknown session {sid}")
            return {"session_id": s.session_id,
                    "client_name": s.client_name,
                    "connected_s": time.time() - s.connected_t,
                    "calls": s.calls, "runs_submitted": s.runs_submitted,
                    "run_seconds_total": s.run_seconds_total,
                    "recent_calls": list(s.recent_calls)}

        def subscribe_events(conn, args):
            since = args.get("since_seq")
            conn.subscribed = True
            with st.lock:
                if conn not in st.subscribers:
                    st.subscribers.append(conn)
            backlog = st.events_since(since) if since is not None else []
            for ev in backlog:
                conn.push_event(ev)
            return {"subscribed": True, "replayed": len(backlog)}

        self.fast_handlers.update({
            "health": health, "engine_status": engine_status,
            "session_status": session_status,
            "subscribe_events": subscribe_events,
        })

        def shutdown(conn, args):
            # S1.3 adds the snapshot-wait; S1.2 adds run cancel
            st.emit("engine_shutdown")
            st.shutdown_requested.set()
            self.dispatcher.stop()
            # slab freed in serve_forever's finally AFTER the
            # dispatcher drains — freeing here (connection thread)
            # races a dispatcher mid-memcpy on an extent (UAF)
            return {"ok": True, "snapshot": None}

        self.critical_handlers["shutdown"] = shutdown
        # bound method used by engine_status above (store attaches in
        # __init__ right after _register_core)
        self.server_pools_backing = lambda: (
            self.store.usage() if hasattr(self, "store") else None)

        # test/diagnostic queued op: holds the dispatcher for `seconds`
        # (stands in for a run until S1.2; proves FIFO + fast-path)
        def _debug_hold(call: QueuedCall):
            time.sleep(float(call.args.get("seconds", 0.1)))
            return {"held": call.args.get("seconds", 0.1)}

        self.dispatcher.handlers["_debug_hold"] = _debug_hold

    # ---- lifecycle ----
    def serve_forever(self) -> None:
        path = self.config.socket_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            os.unlink(path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(16)
        # timeout-poll accept: closing a listening socket from another
        # thread does NOT reliably wake accept(), so shutdown is checked
        # every 200 ms instead
        self._sock.settimeout(0.2)
        self.dispatcher.start()
        self.state.emit("engine_started", config=self.config.public())
        try:
            while not self.state.shutdown_requested.is_set():
                try:
                    client, _ = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                Connection(self, client).start()
        finally:
            try:
                self._sock.close()
            except OSError:
                pass
            try:
                os.unlink(path)
            except OSError:
                pass
            self.dispatcher.join(timeout=30)
            w = getattr(self, "snapshot_writer", None)
            if w is not None:
                w.stop()
                w.join(timeout=30)
            from . import bridge

            nm = getattr(self, "nm", None)
            if nm is not None:
                nm.stop()          # abort transfers before the slab dies
            # sessions FIRST (their pools free transients through the
            # store), slab after — the reverse order dangles the pools
            bridge.close_all_sessions(getattr(self, "store", None))
            if getattr(self, "store", None) is not None \
                    and self.store.slab is not None:
                self.store.slab.free()
