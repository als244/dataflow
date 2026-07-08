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
        # call_done frames can arrive BEFORE _call registers its Ticket
        # (the dispatcher can finish a queued op inside the window
        # between the server's acceptance reply and our registration).
        # Unmatched completions park here and are claimed on
        # registration — dropping them wedges the client forever
        # (found via faulthandler stack dump, S1.1).
        self._orphan_done: dict[str, tuple] = {}
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
                    if t is None:
                        self._orphan_done[msg["ticket"]] = (msg, payload)
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
            orphan = self._orphan_done.pop(ticket.id, None)
            if orphan is None:
                self._tickets[ticket.id] = ticket
        if orphan is not None:                # completed before registration
            done_msg, done_payload = orphan
            if done_msg.get("ok"):
                ticket.result = done_msg.get("result")
                ticket.payload = done_payload
            else:
                ticket.error = done_msg.get("error")
            ticket.done.set()
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

    # ---- object plane ----
    def put_object(self, oid: str, data=None, *, path: str | None = None,
                   meta: dict | None = None, wait: bool = True):
        if (data is None) == (path is None):
            raise ValueError("put_object: exactly one of data|path")
        if path is not None:
            return self._call("put_object",
                              {"id": oid, "path": str(path), "meta": meta},
                              wait=wait)
        if hasattr(data, "tobytes"):
            data = data.tobytes()
        return self._call("put_object", {"id": oid, "meta": meta},
                          payload=data, wait=wait)

    def get_object(self, oid: str, dest=None, *, wait: bool = True):
        if dest is not None:
            return self._call("get_object",
                              {"id": oid, "dest": str(dest)}, wait=wait)
        got = self._call("get_object", {"id": oid}, wait=wait)
        if isinstance(got, tuple):          # (info, payload)
            return got[1]
        return got

    def materialize_object(self, oid: str, fill: dict, *, wait=True):
        return self._call("materialize_object",
                          {"id": oid, "fill": fill}, wait=wait)

    def materialize_group(self, fill: dict, *, wait=True):
        return self._call("materialize_group", {"fill": fill}, wait=wait)

    def release_object(self, oid: str, *, force=False, wait=True):
        return self._call("release_object",
                          {"id": oid, "force": force}, wait=wait)

    def protect_object(self, oid: str, *, wait=True):
        return self._call("protect_object", {"id": oid}, wait=wait)

    def unprotect_object(self, oid: str, *, wait=True):
        return self._call("unprotect_object", {"id": oid}, wait=wait)

    def duplicate_object(self, src: str, dst: str, *, wait=True):
        return self._call("duplicate_object",
                          {"src": src, "dst": dst}, wait=wait)

    def duplicate_object_group(self, ogid: str, *, tag: str,
                               rename: str = "{id}@{tag}",
                               new_ogid: str | None = None, wait=True):
        return self._call("duplicate_object_group",
                          {"ogid": ogid, "tag": tag, "rename": rename,
                           "new_ogid": new_ogid}, wait=wait)

    def create_object_group(self, ogid: str, members=(), *,
                            pattern: str | None = None,
                            object_groups=(), wait=True):
        return self._call("create_object_group",
                          {"ogid": ogid, "members": list(members),
                           "pattern": pattern,
                           "object_groups": list(object_groups)}, wait=wait)

    def delete_object_group(self, ogid: str, *, wait=True):
        return self._call("delete_object_group", {"ogid": ogid}, wait=wait)

    def wipe(self, scope: str, *, force=False, wait=True):
        return self._call("wipe", {"scope": scope, "force": force},
                          wait=wait)

    def query_object(self, oid: str):
        return self._call("query_object", {"id": oid})

    def list_objects(self, pattern: str = "*", *, limit: int = 1000):
        return self._call("list_objects",
                          {"pattern": pattern, "limit": limit})

    def query_object_group(self, ogid: str):
        return self._call("query_object_group", {"ogid": ogid})

    def query_backing(self):
        return self._call("query_backing", {})

    def query_fast(self):
        return self._call("query_fast", {})

    # ---- S1.0 diagnostic ----
    def _debug_hold(self, seconds: float, *, wait: bool = True,
                    timeout: float | None = None):
        return self._call("_debug_hold", {"seconds": seconds},
                          wait=wait, timeout=timeout)
