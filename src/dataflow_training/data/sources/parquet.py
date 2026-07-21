"""ParquetSource: a local parquet file/directory's text column,
tokenized at ingest.

Cursors: {"file": i, "row": j} — file index into the sorted expansion
of the path glob, absolute row within it. Epoch-wrapping. Rows read
in row-group order (pyarrow), one group resident at a time.
"""
from __future__ import annotations

import glob as globlib
from typing import Iterator

import numpy as np

from dataflow_training.data.sequence import Sequence, validate_sequence
from dataflow_training.data.sources.jsonl import doc_to_sequences
from dataflow_training.data.tokenizers import resolve_tokenizer

INSTALL_HINT = ("— install the data extra: pip install -e '.[data]'")


class ParquetSource:
    def __init__(self, pattern: str, *, column: str = "text",
                 tokenizer, max_seqlen: int, long_policy: str,
                 vocab_size: int):
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError as exc:
            raise ImportError(f"pyarrow not installed {INSTALL_HINT}") from exc
        self.pattern = pattern
        self.paths = sorted(globlib.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"no files match {pattern!r}")
        self.column = column
        self.tokenizer = (tokenizer if not isinstance(tokenizer, str)
                          else resolve_tokenizer(tokenizer))
        self.max_seqlen = int(max_seqlen)
        self.long_policy = long_policy
        self.vocab_size = int(vocab_size)
        self.eot_id = self.tokenizer.describe().get("eot_id")

    def file_rows(self, path: str) -> Iterator[tuple[int, str]]:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        row = 0
        for g in range(pf.num_row_groups):
            col = pf.read_row_group(g, columns=[self.column])
            for value in col.column(self.column).to_pylist():
                yield row, ("" if value is None else str(value))
                row += 1

    def sequences(self, cursor: dict | None) -> Iterator[tuple[Sequence, dict]]:
        fi = int(cursor["file"]) if cursor else 0
        row0 = int(cursor.get("row", 0)) if cursor else 0
        piece0 = int(cursor.get("piece", 0)) if cursor else 0
        while True:
            fi %= len(self.paths)
            n_rows = 0
            for row, text in self.file_rows(self.paths[fi]):
                n_rows = row + 1
                if row < row0:
                    continue
                text = text.strip()
                if not text:
                    continue
                ids = self.tokenizer.encode(text)
                pieces = doc_to_sequences(ids, self.eot_id,
                                          self.max_seqlen,
                                          self.long_policy)
                for k, seq in enumerate(pieces):
                    if k < piece0:
                        continue
                    validate_sequence(seq, self.vocab_size)
                    if k + 1 < len(pieces):
                        cur = {"file": fi, "row": row, "piece": k + 1}
                    else:
                        cur = {"file": fi, "row": row + 1, "piece": 0}
                    yield seq, cur
                piece0 = 0
            row0 = 0
            fi = (fi + 1) % len(self.paths)

    def describe(self) -> dict:
        return {"scheme": "parquet", "pattern": self.pattern,
                "files": len(self.paths), "column": self.column,
                "vocab_size": self.vocab_size, "eot_id": self.eot_id,
                "tokenizer": self.tokenizer.describe(),
                "deterministic": True, "max_seqlen": self.max_seqlen,
                "long_policy": self.long_policy}
