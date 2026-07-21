"""DataFeed: hands the packer ready sequences; owns the ingest
worker, requeue, and the cursor.

CONTRACT:
- Hand-out order == source order, with requeued items FIRST (FIFO of
  deferral). The background worker is an implementation detail
  (prefetch): DataFeed's hand-out stream is identical to driving the
  source inline — ``prefetch_sequences=0`` runs synchronously and the
  equality is gated.
- ``requeue(seqs)`` returns packer remainders to the FRONT of the
  line in the order given.
- ``cursor()`` is the resume point BEFORE everything not yet handed
  back out: the source cursor after the LAST HANDED-OUT sequence
  (prefetched-but-unhanded items are re-derived on resume) plus the
  CONTENT of requeued sequences (already consumed from the source,
  so their bytes ride the cursor — small, bounded by the packer
  lookahead, JSON-clean for client_meta).
- Worker errors (source IO, tokenizer) surface on the next
  next_sequence() call, never silently.
- Single consumer (the packer); not a general concurrent queue.
"""
from __future__ import annotations

import queue as queue_mod
import threading
from collections import deque
from pathlib import Path

import numpy as np

from dataflow_training.data.sequence import PackedStep, Sequence

STOP = object()


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


class IngestWorker(threading.Thread):
    """Drains source.sequences(cursor) into a bounded queue of
    (sequence, cursor_after) pairs. Backpressure = the put blocks;
    errors park in ``error`` and surface on the consumer side."""

    def __init__(self, source, start_cursor, depth: int):
        super().__init__(daemon=True, name="datafeed-ingest")
        self.iter = source.sequences(start_cursor)
        self.out: queue_mod.Queue = queue_mod.Queue(maxsize=depth)
        self.stop_event = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            for item in self.iter:
                while True:
                    if self.stop_event.is_set():
                        return
                    try:
                        self.out.put(item, timeout=0.2)
                        break
                    except queue_mod.Full:
                        continue
            self.out.put(STOP)          # finite source exhausted
        except BaseException as exc:
            self.error = exc
            self.out.put(STOP)


class DataFeed:
    def __init__(self, source, *, start_cursor: dict | None = None,
                 capture: str | Path | None = None,
                 prefetch_sequences: int = 256):
        self.source = source
        cursor = dict(start_cursor) if start_cursor else None
        self.pending: deque[Sequence] = deque()
        if cursor and cursor.get("requeued"):
            for d in cursor["requeued"]:
                self.pending.append(sequence_from_json(d))
        source_cursor = cursor.get("source") if cursor else None
        self.source_cursor = source_cursor
        self.worker: IngestWorker | None = None
        self.iter = None
        if prefetch_sequences > 0:
            self.worker = IngestWorker(source, source_cursor,
                                       prefetch_sequences)
            self.worker.start()
        else:
            self.iter = source.sequences(source_cursor)
        self.exhausted = False
        self.capture_fh = open(capture, "ab") if capture else None

    def pull(self):
        """One (sequence, cursor_after) from the worker or the inline
        iterator; raises StopIteration on a finite source's end and
        re-raises worker errors."""
        if self.worker is not None:
            item = self.worker.out.get()
            if item is STOP:
                self.exhausted = True
                if self.worker.error is not None:
                    raise self.worker.error
                raise StopIteration
            return item
        return next(self.iter)

    def next_sequence(self) -> Sequence:
        if self.pending:
            seq = self.pending.popleft()
        else:
            seq, self.source_cursor = self.pull()
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
        if self.worker is not None:
            self.worker.stop_event.set()
            while True:                 # unblock a full queue
                try:
                    self.worker.out.get_nowait()
                except queue_mod.Empty:
                    break
            self.worker.join(timeout=5.0)
            self.worker = None
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
