"""Capture files: an exact replayable record of handed-out sequences.

Format: consecutive records, each
    int32 n | int32 n_extras | n int32 tokens | n int32 targets |
    per extra: int32 name_len | name utf-8 | int32 dtype_code |
               raw array bytes (n elements)
dtype codes: 0 = int32, 1 = float32.

write_record appends one sequence; CaptureSource replays a file as a
finite DataSource (cursor = {"rec": index}).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from dataflow_training.data.sequence import Sequence, validate_sequence

DTYPES = {0: np.int32, 1: np.float32}
CODES = {np.dtype(np.int32): 0, np.dtype(np.float32): 1}


def write_record(fh, seq: Sequence) -> None:
    n = len(seq)
    head = np.asarray([n, len(seq.extras)], dtype=np.int32)
    fh.write(head.tobytes())
    fh.write(np.ascontiguousarray(seq.tokens).tobytes())
    fh.write(np.ascontiguousarray(seq.targets).tobytes())
    for name, arr in sorted(seq.extras.items()):
        blob = name.encode()
        fh.write(np.asarray([len(blob)], dtype=np.int32).tobytes())
        fh.write(blob)
        fh.write(np.asarray([CODES[arr.dtype]], dtype=np.int32).tobytes())
        fh.write(np.ascontiguousarray(arr).tobytes())


def read_capture(path: str | Path) -> list[Sequence]:
    raw = Path(path).read_bytes()
    out: list[Sequence] = []
    at = 0

    def take(nbytes: int) -> bytes:
        nonlocal at
        chunk = raw[at:at + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(f"truncated capture at byte {at}")
        at += nbytes
        return chunk

    while at < len(raw):
        n, n_extras = np.frombuffer(take(8), dtype=np.int32)
        tokens = np.frombuffer(take(4 * int(n)), dtype=np.int32).copy()
        targets = np.frombuffer(take(4 * int(n)), dtype=np.int32).copy()
        extras = {}
        for _ in range(int(n_extras)):
            (name_len,) = np.frombuffer(take(4), dtype=np.int32)
            name = take(int(name_len)).decode()
            (code,) = np.frombuffer(take(4), dtype=np.int32)
            dt = DTYPES[int(code)]
            extras[name] = np.frombuffer(
                take(int(n) * dt().itemsize), dtype=dt).copy()
        out.append(Sequence(tokens=tokens, targets=targets, extras=extras))
    return out


class CaptureSource:
    """Replays a capture file. FINITE: iteration ends at the last
    record (the one sanctioned StopIteration in the source zoo)."""

    def __init__(self, path: str | Path, *, vocab_size: int):
        self.path = str(path)
        self.records = read_capture(path)
        self.vocab_size = int(vocab_size)

    def sequences(self, cursor: dict | None) -> Iterator[tuple[Sequence, dict]]:
        at = int(cursor["rec"]) if cursor else 0
        for i in range(at, len(self.records)):
            seq = self.records[i]
            validate_sequence(seq, self.vocab_size)
            yield seq, {"rec": i + 1}

    def describe(self) -> dict:
        return {"scheme": "capture", "path": self.path,
                "records": len(self.records),
                "vocab_size": self.vocab_size, "eot_id": None,
                "tokenizer": None, "deterministic": True}
