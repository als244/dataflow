"""Wire vocabulary for dataflow-peer/s2 rendezvous transfers.

Messages are plain dicts with a ``kind`` field (the S1 framing carries
them as the JSON line; payload bytes ride as the binary frame). The
rendezvous (design doc: distributed_plumbing_spec.md §3):

    sender                         receiver
    RTS {send_id, dest_id, bytes, meta, overwrite [, eager]}
                                   collision/lease/capacity checks;
                                   reservation = the FINAL store extent
              <- CTS {send_id}            (socket transport)
              <- ADDR {send_id, raddr, rkey}   (rdma transport)
              <- NACK {send_id, code}          (refusal, nothing held)
    CHUNK {send_id, seq, last} + bytes   (socket; rdma writes directly)
    DONE {send_id, checksum}
              <- DONE_ACK {send_id}      (after CATALOG COMMIT)

- checksum: crc32 (zlib) over the full payload — wire/DMA corruption
  detection on a trusted LAN, not cryptographic. (The spec says crc32c;
  stdlib zlib.crc32 is the pragmatic stand-in — same failure class,
  zero dependencies. Revisit only if a real corruption slips it.)
- eager: payloads <= EAGER_MAX_BYTES ride inside RTS; no CHUNK/DONE
  phase — the receiver commits and DONE_ACKs directly.
- idempotency: receivers remember recently COMPLETED send_ids and
  re-ack duplicates without re-committing; senders resend DONE if the
  ack goes missing.
- ordering: CHUNK seq must arrive strictly ascending (transports are
  ordered streams; a gap/reorder is a protocol violation => abort,
  fail-stop v1).
"""
from __future__ import annotations

import zlib
from enum import Enum

EAGER_MAX_BYTES = 64 * 1024
CHUNK_BYTES_DEFAULT = 128 * 1024 * 1024
NACK_BACKOFF_BASE_S = 0.1     # 0.1 -> 0.2 -> 0.4 -> 0.8 -> 1.6
NACK_BACKOFF_TRIES = 5        # PROVISIONAL default (spec §3 revisit flag)
INACTIVITY_TIMEOUT_S = 60.0   # no progress on a live transfer => abort
DEDUP_RING_SIZE = 256


class NackCode(str, Enum):
    BUSY = "BUSY"              # transient (lease held, admission pressure)
    CAPACITY = "CAPACITY"      # allocator/quota refusal
    COLLISION = "COLLISION"    # dest_id exists and overwrite rules forbid


class TransferState(str, Enum):
    NEGOTIATING = "negotiating"
    MOVING = "moving"
    COMMITTING = "committing"
    DONE = "done"
    ERROR = "error"


def payload_checksum(data) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF     # buffer protocol: no copy


def rts(send_id: str, dest_id: str, nbytes: int, *, meta=None,
        overwrite: bool = False, eager: bool = False) -> dict:
    return {"kind": "RTS", "send_id": send_id, "dest_id": dest_id,
            "bytes": nbytes, "meta": meta or {}, "overwrite": overwrite,
            "eager": eager}


def cts(send_id: str) -> dict:
    return {"kind": "CTS", "send_id": send_id}


def addr(send_id: str, raddr: int, rkey: int) -> dict:
    return {"kind": "ADDR", "send_id": send_id, "raddr": raddr,
            "rkey": rkey}


def nack(send_id: str, code: NackCode) -> dict:
    return {"kind": "NACK", "send_id": send_id, "code": code.value}


def chunk(send_id: str, seq: int, last: bool) -> dict:
    return {"kind": "CHUNK", "send_id": send_id, "seq": seq, "last": last}


def done(send_id: str, checksum: int) -> dict:
    return {"kind": "DONE", "send_id": send_id, "checksum": checksum}


def done_ack(send_id: str) -> dict:
    return {"kind": "DONE_ACK", "send_id": send_id}
