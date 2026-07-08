"""Engine service client library — the only thing tools/drivers import.

Blocking by default: every queued method takes ``wait=True`` (block
until the dispatcher finishes it, return the result) or ``wait=False``
(return a ``Ticket``; redeem with ``client.wait(ticket)``). Fast-path
methods always return inline. A background reader thread routes reply
frames by request id, ``call_done`` frames by ticket, and service
events into subscription queues.
"""
from __future__ import annotations

import queue
import socket
import threading
from pathlib import Path

from .wire import SCHEMA_VERSION, Conn, ServiceError

DEFAULT_SOCKET = str(Path.home() / ".dataflow" / "dataflowd.sock")


class Ticket:
    def __init__(self, ticket_id: str):
        self.id = ticket_id
        self.done = threading.Event()
        self.result = None
        self.error: dict | None = None
        self.payload: bytes | None = None


class EngineClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET, *,
                 client_name: str = ""):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        self._conn = Conn(sock)
        self._wlock = threading.Lock()
        self._next_id = 0
        self._replies: dict[int, queue.Queue] = {}
        self._tickets: dict[str, Ticket] = {}
        self._events: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False

        # handshake happens before the reader starts (simple + ordered)
        rid = self._rid()
        self._conn.send({"id": rid, "op": "hello", "args": {
            "schema_version": SCHEMA_VERSION, "client_name": client_name}})
        frame = self._conn.recv()
        if frame is None:
            raise ConnectionError("daemon closed during handshake")
        if not frame.msg.get("ok"):
            raise ServiceError.from_json(frame.msg["error"])
        hello = frame.msg["result"]
        self.session_id = hello["session_id"]
        self.engine_info = hello["engine"]

        self._reader = threading.Thread(target=self._read_loop,
                                        name="svc-reader", daemon=True)
        self._reader.start()

    # ---- plumbing ----
    def _rid(self) -> int:
        self._next_id += 1
        return self._next_id

    def _read_loop(self) -> None:
        while True:
            try:
                frame = self._conn.recv()
            except (OSError, ConnectionError):
                frame = None
            if frame is None:
                self._closed = True
                # unblock everyone
                with self._lock:
                    for q in self._replies.values():
                        q.put(None)
                    for t in self._tickets.values():
                        if not t.done.is_set():
                            t.error = {"code": "IO_ERROR",
                                       "message": "connection closed",
                                       "detail": {}}
                            t.done.set()
                self._events.put(None)
                return
            msg, payload = frame.msg, frame.payload
            if "id" in msg:
                with self._lock:
                    q = self._replies.pop(msg["id"], None)
                if q is not None:
                    q.put((msg, payload))
            elif msg.get("event") == "call_done":
                with self._lock:
                    t = self._tickets.pop(msg["ticket"], None)
                if t is not None:
                    if msg.get("ok"):
                        t.result = msg.get("result")
                        t.payload = payload
                    else:
                        t.error = msg.get("error")
                    t.done.set()
            else:
                self._events.put(msg)

    def _call(self, op: str, args: dict, *,
              payload: bytes | memoryview | None = None,
              wait: bool = True, timeout: float | None = None):
        if self._closed:
            raise ConnectionError("client closed")
        rid = self._rid()
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._replies[rid] = q
        with self._wlock:
            self._conn.send({"id": rid, "op": op, "args": args}, payload)
        got = q.get(timeout=timeout)
        if got is None:
            raise ConnectionError("connection closed")
        msg, pay = got
        if not msg.get("ok"):
            raise ServiceError.from_json(msg["error"])
        if "result" in msg:                 # fast-path / critical inline
            return (msg["result"], pay) if pay is not None else msg["result"]
        ticket = Ticket(msg["ticket"])
        with self._lock:
            self._tickets[ticket.id] = ticket
        if not wait:
            return ticket
        return self.wait(ticket, timeout=timeout)

    def wait(self, ticket: Ticket, timeout: float | None = None):
        if not ticket.done.wait(timeout):
            raise TimeoutError(f"ticket {ticket.id}")
        if ticket.error is not None:
            raise ServiceError.from_json(ticket.error)
        if ticket.payload is not None:
            return ticket.result, ticket.payload
        return ticket.result

    # ---- lifecycle ----
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._conn.close()

    disconnect = close

    # ---- fast path ----
    def health(self) -> dict:
        return self._call("health", {})

    def engine_status(self) -> dict:
        return self._call("engine_status", {})

    def session_status(self, session_id: str | None = None) -> dict:
        return self._call("session_status", {"session_id": session_id})

    def subscribe_events(self, kinds=("run", "snapshot", "error", "engine"),
                         *, since_seq: int | None = None):
        """Returns an iterator of service events (None-terminated on
        connection close)."""
        self._call("subscribe_events",
                   {"kinds": list(kinds), "since_seq": since_seq})

        def _iter():
            while True:
                ev = self._events.get()
                if ev is None:
                    return
                yield ev

        return _iter()

    # ---- critical ----
    def shutdown(self, *, force: bool = False) -> dict:
        return self._call("shutdown", {"force": force})

    # ---- S1.0 diagnostic ----
    def _debug_hold(self, seconds: float, *, wait: bool = True,
                    timeout: float | None = None):
        return self._call("_debug_hold", {"seconds": seconds},
                          wait=wait, timeout=timeout)
