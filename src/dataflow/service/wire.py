"""Wire protocol for the engine service: framing, envelope, errors.

One CONTROL frame = one UTF-8 JSON object terminated by ``\n``. A
control frame announcing a binary payload carries ``payload_bytes``;
exactly that many raw bytes follow the newline (no base64, either
direction). Requests are ``{"id", "op", "args"}``; replies either
answer inline (``{"id", "ok", "result"}`` — fast-path ops) or accept
into the FIFO (``{"id", "ok", "ticket"}``) with a later push frame
``{"event": "call_done", "ticket", "ok", "result"|"error"}``. Service
events (subscriptions) are push frames too: ``{"event": ..., "seq",
"t", ...}``.

Timing classes live server-side; the wire does not distinguish them
beyond inline-vs-ticket replies.
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass

SCHEMA_VERSION = "dataflow-service/s1"

ERROR_CODES = (
    "SCHEMA_VERSION_SKEW", "BAD_REQUEST", "UNKNOWN_OP", "UNKNOWN_OBJECT",
    "UNKNOWN_GROUP", "UNKNOWN_PROGRAM", "UNKNOWN_RUN", "UNKNOWN_SNAPSHOT",
    "BINDING_MISMATCH", "MISSING_INPUTS", "CAPACITY", "PROTECTED",
    "LEASED", "RUN_FAILED", "CANCELLED", "IO_ERROR", "VERSION_SKEW",
    "INTERNAL",
)


class ServiceError(Exception):
    """Raised client-side for error replies; carried as data server-side."""

    def __init__(self, code: str, message: str, detail: dict | None = None):
        assert code in ERROR_CODES, code
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message,
                "detail": self.detail}

    @classmethod
    def from_json(cls, d: dict) -> "ServiceError":
        return cls(d["code"], d.get("message", ""), d.get("detail"))


@dataclass
class Frame:
    """A decoded control frame plus its binary payload (if any)."""

    msg: dict
    payload: bytes | None = None


class Conn:
    """Blocking framed connection over a unix socket (both ends).

    Thread contract: one reader at a time, one writer at a time —
    callers hold their own locks; this class only owns framing.
    """

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._rbuf = b""

    # ---- send ----
    def send(self, msg: dict, payload: bytes | memoryview | None = None) -> None:
        if payload is not None:
            msg = dict(msg)
            msg["payload_bytes"] = len(payload)
        data = json.dumps(msg, separators=(",", ":")).encode() + b"\n"
        self.sock.sendall(data)
        if payload is not None:
            self.sock.sendall(payload)

    # ---- recv ----
    def recv(self) -> Frame | None:
        """Next frame, or None on clean EOF."""
        line = self._read_line()
        if line is None:
            return None
        msg = json.loads(line)
        payload = None
        n = msg.pop("payload_bytes", 0)
        if n:
            payload = self._read_exact(n)
        return Frame(msg, payload)

    def _read_line(self) -> bytes | None:
        while b"\n" not in self._rbuf:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                return None if not self._rbuf else self._rbuf
            self._rbuf += chunk
        line, self._rbuf = self._rbuf.split(b"\n", 1)
        return line

    def _read_exact(self, n: int) -> bytes:
        while len(self._rbuf) < n:
            chunk = self.sock.recv(min(1 << 20, n - len(self._rbuf) + (1 << 16)))
            if not chunk:
                raise ConnectionError("EOF mid-payload")
            self._rbuf += chunk
        out, self._rbuf = self._rbuf[:n], self._rbuf[n:]
        return bytes(out)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
