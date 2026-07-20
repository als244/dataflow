"""DataFeed: hands the packer ready sequences; owns requeue + cursor.

CONTRACT:
- Hand-out order == source order, with requeued items FIRST (FIFO of
  deferral). This implementation is synchronous; the background
  worker arrives as a pure refactor once the semantics are gated —
  by contract the threaded feed's output is identical.
- ``requeue(seqs)`` returns packer remainders to the FRONT of the
  line in the order given.
- ``cursor()`` is the resume point BEFORE everything not yet handed
  back out: the source cursor of the next unpulled sequence plus the
  CONTENT of requeued sequences (they were consumed from the source
  already, so their bytes ride the cursor — small, bounded by the
  packer lookahead, JSON-clean for client_meta).
- Single consumer (the packer); not a general concurrent queue.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from dataflow_training.data.sequence import PackedStep, Sequence


def sequence_to_json(seq: Sequence) -> dict:
    out = {"tokens": seq.tokens.tolist(), "targets": seq.targets.tolist()}
    if seq.extras:
        out["extras"] = {k: v.tolist() for k, v in seq.extras.items()}
    return out


def sequence_from_json(d: dict) -> Sequence:
    return Sequence(
        tokens=np.asarray(d["tokens"], dtype=np.int32),
        targets=np.asarray(d["targets"], dtype=np.int32),
        extras={k: np.asarray(v) for k, v in d.get("extras", {}).items()})


class DataFeed:
    def __init__(self, source, *, start_cursor: dict | None = None,
                 capture: str | Path | None = None):
        self.source = source
        cursor = dict(start_cursor) if start_cursor else None
        self.pending: deque[Sequence] = deque()
        if cursor and cursor.get("requeued"):
            for d in cursor["requeued"]:
                self.pending.append(sequence_from_json(d))
        source_cursor = cursor.get("source") if cursor else None
        self.iter = source.sequences(source_cursor)
        self.source_cursor = source_cursor
        self.capture_fh = open(capture, "ab") if capture else None

    def next_sequence(self) -> Sequence:
        if self.pending:
            seq = self.pending.popleft()
        else:
            seq, self.source_cursor = next(self.iter)
        if self.capture_fh is not None:
            from dataflow_training.data.sources.capture import write_record

            write_record(self.capture_fh, seq)
        return seq

    def requeue(self, seqs: list[Sequence]) -> None:
        for seq in reversed(seqs):
            self.pending.appendleft(seq)

    def cursor(self) -> dict:
        return {"source": self.source_cursor,
                "requeued": [sequence_to_json(s) for s in self.pending]}

    def close(self) -> None:
        if self.capture_fh is not None:
            self.capture_fh.close()
            self.capture_fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PrepackedFeed:
    """Packer bypass: yields PackedStep objects directly (from a
    caller-provided list — e.g. hand-built rounds or capture-derived
    replays). Cursor = step index."""

    def __init__(self, steps: list[PackedStep], *, start: int = 0):
        self.steps = steps
        self.at = int(start)

    def next_step(self) -> PackedStep:
        step = self.steps[self.at]
        self.at += 1
        return step

    def cursor(self) -> dict:
        return {"step": self.at}
