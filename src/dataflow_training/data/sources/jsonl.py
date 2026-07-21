"""JsonlSource / TextSource: local text corpora, tokenized at ingest.

- JsonlSource: one document per JSON line (a named text field).
- TextSource: plain files split on a delimiter (default: blank line).

Targets are the source's job: within a document, target = next token;
the document-final position targets the tokenizer's end-of-text id
(the corpus-source convention — the model learns to emit it), or -1
(masked) when the tokenizer has none.

Cursors: {"file": i, "rec": j} — file index into the sorted expansion
of the path glob, record index within it. Epoch-wrapping.
"""
from __future__ import annotations

import glob as globlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from dataflow_training.data.sequence import Sequence, validate_sequence
from dataflow_training.data.tokenizers import Tokenizer, resolve_tokenizer


def doc_to_sequences(ids: list[int], eot_id, max_seqlen: int,
                     long_policy: str) -> list[Sequence]:
    """Token ids -> 0..k sequences per the long policy, with the
    shift-by-one targets and the eot (or masked) final target."""
    n = len(ids)
    if n < 1:
        return []
    if n > max_seqlen and long_policy == "exclude":
        return []
    if n > max_seqlen and long_policy == "trim":
        ids = ids[:max_seqlen]
        n = max_seqlen
    tokens = np.asarray(ids, dtype=np.int32)
    targets = np.empty(n, dtype=np.int32)
    targets[:-1] = tokens[1:]
    targets[-1] = eot_id if eot_id is not None else -1
    if n <= max_seqlen or long_policy in ("whole",):
        return [Sequence(tokens=tokens, targets=targets)]
    out = []
    for lo in range(0, n, max_seqlen):          # chunk
        hi = min(lo + max_seqlen, n)
        out.append(Sequence(tokens=np.ascontiguousarray(tokens[lo:hi]),
                            targets=np.ascontiguousarray(targets[lo:hi])))
    return out


class RecordFileSource:
    """Shared machinery: an ordered list of files, each yielding an
    ordered list of text records; subclasses define read_records."""

    def __init__(self, pattern: str, *, tokenizer, max_seqlen: int,
                 long_policy: str, vocab_size: int):
        self.pattern = pattern
        self.paths = sorted(globlib.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"no files match {pattern!r}")
        self.tokenizer: Tokenizer = (tokenizer if not isinstance(tokenizer, str)
                                     else resolve_tokenizer(tokenizer))
        self.max_seqlen = int(max_seqlen)
        self.long_policy = long_policy
        self.vocab_size = int(vocab_size)
        self.eot_id = self.tokenizer.describe().get("eot_id")

    def read_records(self, path: str) -> list[str]:
        raise NotImplementedError

    def sequences(self, cursor: dict | None) -> Iterator[tuple[Sequence, dict]]:
        fi = int(cursor["file"]) if cursor else 0
        rec0 = int(cursor.get("rec", 0)) if cursor else 0
        piece0 = int(cursor.get("piece", 0)) if cursor else 0
        while True:
            fi %= len(self.paths)
            records = self.read_records(self.paths[fi])
            for j in range(rec0, len(records)):
                ids = self.tokenizer.encode(records[j])
                pieces = doc_to_sequences(ids, self.eot_id,
                                          self.max_seqlen, self.long_policy)
                after_rec = ({"file": fi, "rec": j + 1}
                             if j + 1 < len(records)
                             else {"file": (fi + 1) % len(self.paths),
                                   "rec": 0})
                for k, seq in enumerate(pieces):
                    if k < piece0:
                        continue
                    validate_sequence(seq, self.vocab_size)
                    if k + 1 < len(pieces):
                        cur = {"file": fi, "rec": j, "piece": k + 1}
                    else:
                        cur = after_rec
                    yield seq, cur
                piece0 = 0
            rec0 = 0
            fi += 1

    def describe(self) -> dict:
        return {"scheme": self.scheme, "pattern": self.pattern,
                "files": len(self.paths),
                "vocab_size": self.vocab_size, "eot_id": self.eot_id,
                "tokenizer": self.tokenizer.describe(),
                "deterministic": True, "max_seqlen": self.max_seqlen,
                "long_policy": self.long_policy}


class JsonlSource(RecordFileSource):
    scheme = "jsonl"

    def __init__(self, pattern: str, *, field: str = "text", **kw):
        self.field = field
        super().__init__(pattern, **kw)

    def read_records(self, path: str) -> list[str]:
        out = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line)[self.field])
        return out


class TextSource(RecordFileSource):
    scheme = "txt"

    def __init__(self, pattern: str, *, delimiter: str = "\n\n", **kw):
        self.delimiter = delimiter
        super().__init__(pattern, **kw)

    def read_records(self, path: str) -> list[str]:
        raw = Path(path).read_text()
        return [d for d in raw.split(self.delimiter) if d.strip()]
