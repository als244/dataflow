"""DataSource: the origin of sequences, and the spec registry.

CONTRACT — every DataSource implementation:
- ``sequences(cursor)`` is a pure function of (constructor args,
  cursor): two iterations from equal cursors yield byte-identical
  sequences forever. ``cursor=None`` starts at the beginning; each
  yield's ``cursor_after`` resumes EXACTLY after that sequence.
  Cursors are small JSON-clean dicts (they ride checkpoint
  client_meta).
- Corpus sources wrap epochs indefinitely; StopIteration is reserved
  for genuinely finite sources (e.g. a capture replay).
- Every yielded Sequence already satisfies the Sequence invariants
  and the source's max_seqlen/long_policy — enforcement lives AT THE
  SOURCE; the feed and packer never re-check.
- ``describe()`` returns JSON-clean facts (vocab_size, eot_id or
  None, tokenizer identity or None, long_policy, max_seqlen,
  deterministic flag, source specifics) — stamped into run metadata.
- Thread-safety is NOT required: exactly one DataFeed worker drives
  a source instance.

The ``--data`` SPEC grammar: ``scheme:main[,key=value...]`` — e.g.
``shards:datasets/fineweb10B``, ``shards:ROOT,window=1024``,
``synthetic:vocab=50304,mean_len=800``. ``resolve_data(spec)``
builds the source; tools never import concrete source classes.
"""
from __future__ import annotations

from typing import Iterator, Protocol

from dataflow_training.data.sequence import Sequence

LONG_POLICIES = ("exclude", "trim", "chunk", "whole")


class DataSource(Protocol):
    def sequences(self, cursor: dict | None) -> Iterator[tuple[Sequence, dict]]: ...

    def describe(self) -> dict: ...


def parse_spec(spec: str) -> tuple[str, str, dict[str, str]]:
    """'scheme:main,k=v,...' -> (scheme, main, {k: v}). The main
    argument is everything before the first ',' that contains no
    '='; key=value pairs follow."""
    scheme, sep, rest = spec.partition(":")
    if not sep:
        scheme, rest = spec, ""
    main = ""
    kv: dict[str, str] = {}
    for i, part in enumerate(rest.split(",") if rest else []):
        if "=" in part:
            k, _, v = part.partition("=")
            kv[k.strip()] = v.strip()
        elif i == 0:
            main = part.strip()
        else:
            raise ValueError(f"bad --data spec segment {part!r} in {spec!r}")
    return scheme.strip(), main, kv


def resolve_data(spec: str, *, max_seqlen: int, vocab_size: int):
    """SPEC string -> a constructed DataSource. The two geometry facts
    every source needs (max_seqlen for the long policy, vocab_size for
    the invariant check) come from the run config, not the spec."""
    scheme, main, kv = parse_spec(spec)
    long_policy = kv.pop("long_policy", "exclude")
    if long_policy not in LONG_POLICIES:
        raise ValueError(f"long_policy {long_policy!r} not in {LONG_POLICIES}")

    if scheme == "shards":
        from dataflow_training.data.sources.shards import ShardSource

        window = kv.pop("window", None)
        source = ShardSource(
            root=main, split=kv.pop("split", "train"),
            max_seqlen=max_seqlen, long_policy=long_policy,
            vocab_size=vocab_size,
            window=int(window) if window else None)
    elif scheme == "synthetic":
        from dataflow_training.data.sources.synthetic import SyntheticSource

        source = SyntheticSource(
            vocab_size=int(kv.pop("vocab", vocab_size)),
            mean_len=int(kv.pop("mean_len", "512")),
            seed=int(kv.pop("seed", "0")),
            max_seqlen=max_seqlen, long_policy=long_policy)
    elif scheme == "capture":
        from dataflow_training.data.sources.capture import CaptureSource

        source = CaptureSource(path=main, vocab_size=vocab_size)
    else:
        raise ValueError(
            f"unknown --data scheme {scheme!r} (known: shards, synthetic, "
            f"capture)")
    if kv:
        raise ValueError(f"unused --data keys {sorted(kv)} for {scheme!r}")
    return source
