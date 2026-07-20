"""SyntheticSource: deterministic random sequences for tests/benches.

Sequence k's content is a pure function of (seed, k): lengths draw
from a clipped geometric around mean_len, ids uniform over the vocab.
No corpus, no files — the packer/feed test workhorse and a throughput
source that needs no data.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from dataflow_training.data.sequence import Sequence, validate_sequence


class SyntheticSource:
    def __init__(self, *, vocab_size: int, mean_len: int = 512,
                 seed: int = 0, max_seqlen: int,
                 long_policy: str = "exclude"):
        self.vocab_size = int(vocab_size)
        self.mean_len = int(mean_len)
        self.seed = int(seed)
        self.max_seqlen = int(max_seqlen)
        self.long_policy = long_policy

    def make(self, k: int) -> Sequence:
        rng = np.random.default_rng((self.seed << 32) ^ k)
        n = int(rng.geometric(1.0 / self.mean_len))
        n = max(1, min(n, 4 * self.mean_len))
        if n > self.max_seqlen and self.long_policy in ("trim", "chunk"):
            n = self.max_seqlen        # chunk degenerates to trim here
        tokens = rng.integers(0, self.vocab_size, size=n, dtype=np.int32)
        targets = np.roll(tokens, -1).astype(np.int32)
        return Sequence(tokens=tokens, targets=targets)

    def sequences(self, cursor: dict | None) -> Iterator[tuple[Sequence, dict]]:
        k = int(cursor["k"]) if cursor else 0
        while True:
            seq = self.make(k)
            k += 1
            if self.long_policy == "exclude" and len(seq) > self.max_seqlen:
                continue
            validate_sequence(seq, self.vocab_size)
            yield seq, {"k": k}

    def describe(self) -> dict:
        return {"scheme": "synthetic", "vocab_size": self.vocab_size,
                "eot_id": None, "tokenizer": None, "deterministic": True,
                "mean_len": self.mean_len, "seed": self.seed,
                "max_seqlen": self.max_seqlen,
                "long_policy": self.long_policy}
