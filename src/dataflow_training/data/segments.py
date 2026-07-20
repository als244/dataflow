"""Segments: the packed-round value object (per-sequence lengths,
device cu/positions materialized once per run). WORKLOAD-owned: the
engine treats run_args as fully opaque (its own contract), so the
wire seq_lens -> Segments conversion, the device materialization
(pinned + non_blocking, identity-deduped), and the dims-uniform
fallback all live here, cached in ctx.run_values by the first
consuming task (resolve_segments).
"""
from __future__ import annotations

from __future__ import annotations
import math
import torch
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Segments:
    """How one round's tokens split into sequences — the SINGLE varlen
    descriptor shared by packing, engine blocks, and reference models.

    ``lengths`` (host) are the per-sequence token counts (sum == tokens)
    and fully define the geometry. The device tensors the varlen flash
    kernels and rope need are carried as FIELDS, materialized ONCE by
    ``.on(device)``:
      - ``cu``        (n_seq + 1,) int32 cumulative segment boundaries
      - ``positions`` (tokens,)    int32 per-sequence rope indices
    ``.on`` is called once per round by the FIRST consuming task
    (resolve_segments caches the result in ctx.run_values; once per
    golden forward on the reference side); every stage/op downstream reads
    ``seg.cu`` / ``seg.positions`` as plain attributes. Nothing rebuilds a
    device tensor from host data mid-round — that would be a hidden
    host->device sync (the aten-hidden-syncs discipline). ``cu`` /
    ``positions`` are excluded from equality/hash (identity is ``lengths``).

    Replaces the old seq_spec (int | tuple) + the seq_lens_of /
    sequence_bounds / positions_for / attn_meta free-function family.
    """
    lengths: tuple[int, ...]
    cu: torch.Tensor | None = field(default=None, compare=False)
    positions: torch.Tensor | None = field(default=None, compare=False)

    @classmethod
    def uniform(cls, seq_len: int, batch: int) -> "Segments":
        return cls((int(seq_len),) * int(batch))

    @classmethod
    def from_boundaries(cls, cu) -> "Segments":
        """[0, b1, ..., tokens] cumulative boundaries -> Segments (host)."""
        cu = [int(x) for x in cu]
        if len(cu) < 2 or cu[0] != 0 or any(b < a for a, b in zip(cu, cu[1:])):
            raise ValueError(f"cumulative boundaries from 0 required, got {cu}")
        return cls(tuple(b - a for a, b in zip(cu, cu[1:])))

    @classmethod
    def from_dims(cls, d) -> "Segments":
        """The round's segmentation implied by a dims config (host):
        explicit ``seq_lens`` when ragged, else ``batch`` uniform
        ``seq_len`` sequences. Materialize with ``.on(device)``."""
        sl = getattr(d, "seq_lens", None)
        if sl is not None:
            return cls(tuple(int(n) for n in sl))
        return cls.uniform(d.seq_len, d.tokens // d.seq_len)

    @property
    def tokens(self) -> int:
        return sum(self.lengths)

    @property
    def max_len(self) -> int:
        return max(self.lengths)

    @property
    def bounds(self) -> list[tuple[int, int]]:
        out, lo = [], 0
        for n in self.lengths:
            out.append((lo, lo + n))
            lo += n
        return out

    @property
    def boundaries(self) -> list[int]:
        """[0, b1, ..., tokens] cumulative host boundaries — the inverse of
        ``from_boundaries`` and the form run_args['seq_lens'] carries."""
        out, acc = [0], 0
        for n in self.lengths:
            acc += n
            out.append(acc)
        return out

    @property
    def materialized(self) -> bool:
        return self.cu is not None

    def on(self, device) -> "Segments":
        """Materialize ``cu`` / ``positions`` on ``device`` ONCE and return a
        Segments carrying them as fields. Pinned staging + non_blocking copy
        — never a pageable H2D (the hidden-sync rule). Idempotent when the
        tensors already live on ``device``."""
        if self.cu is not None and self.cu.device == torch.device(device):
            return self
        b = [0]
        for n in self.lengths:
            b.append(b[-1] + n)
        cu_host = torch.tensor(b, dtype=torch.int32).pin_memory()
        if self.lengths:
            pos_host = torch.cat(
                [torch.arange(n, dtype=torch.int32) for n in self.lengths]
            ).pin_memory()
        else:
            pos_host = torch.empty(0, dtype=torch.int32).pin_memory()
        return replace(
            self,
            cu=cu_host.to(device, non_blocking=True),
            positions=pos_host.to(device, non_blocking=True),
        )


def uniform_segments(dims, program) -> dict:
    """The standard (unpacked / fixed-shape) path's run_args["segments"]:
    every round appearing in ``program`` maps to the SAME host ``Segments``
    implied by ``dims`` (``batch`` uniform ``seq_len`` sequences, or the
    config's fixed ``seq_lens``) — one shared object, materialized once by
    the first consuming task. Round key is the task id's ``{s}_{r}_{i}``
    middle field (matches resolve_segments), a superset of block rounds; extra
    keys are harmless."""
    seg = Segments.from_dims(dims)
    rounds = set()
    for t in program.tasks:
        parts = t.id.rsplit("_", 3)
        if len(parts) >= 3:
            rounds.add(parts[2])
    return {r: seg for r in (rounds or {"0"})}


def resolve_segments(ctx, dims, round_key) -> "Segments":
    """The round's materialized Segments, resolved from run_args by the
    FIRST consuming task and cached in ctx.run_values for the rest of
    the run. Accepts the clean internal form run_args["segments"] =
    {round: Segments}, the wire form run_args["seq_lens"] = {round:
    [0, b1, ..., t]} cumulative boundaries, or NOTHING — the uniform
    partition implied by ``dims`` (the non-packed default the service
    used to fill engine-side). Device fields build ONCE via a pinned +
    non_blocking copy (the hidden-sync rule), identity-deduped so the
    uniform case shares a single device copy across rounds; on
    non-physical backends (planning/sim) host Segments pass through
    unmaterialized."""
    rv = ctx.run_values if ctx.run_values is not None else {}
    cache = rv.setdefault("segments_materialized", {})
    if round_key in cache:
        return cache[round_key]
    ra = ctx.run_args or {}
    segs = ra.get("segments")
    host = segs.get(round_key) if segs else None
    if host is None:
        wire = ra.get("seq_lens")
        if wire and round_key in wire:
            host = Segments.from_boundaries(wire[round_key])
        else:
            host = rv.setdefault("segments_uniform_host",
                                 Segments.from_dims(dims))
    if getattr(ctx.backend, "physical", False):
        # dedup by VALUE (Segments hashes on lengths): equal partitions
        # share one device copy — id()-keyed dedup is a trap here (the
        # host intermediate dies and its id gets reused across rounds)
        by_host = rv.setdefault("segments_materialized_by_host", {})
        if host not in by_host:
            by_host[host] = host.on(f"cuda:{ctx.backend.device}")
        host = by_host[host]
    cache[round_key] = host
    return host
