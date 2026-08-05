"""Segments: the packed-round value object (per-sequence lengths,
device cu/positions materialized once per run). WORKLOAD-owned: the
engine treats run_args as fully opaque (its own contract), so the
wire seq_lens -> Segments conversion, the device materialization
(pinned + non_blocking, identity-deduped), and the dims-uniform
fallback all live here, cached in ctx.run_values by the first
consuming task (resolve_segments).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch


@dataclass(frozen=True)
class Segments:
    """How one round's tokens split into sequences — the SINGLE varlen
    descriptor shared by packing, engine blocks, and reference models.

    ``lengths`` (host) are the per-sequence token counts (sum == tokens)
    and fully define the geometry. The device tensors varlen kernels need are
    carried as FIELDS, materialized ONCE by
    ``.on(device)``:
      - ``cu``        (n_seq + 1,) int32 cumulative segment boundaries
      - ``positions`` (tokens,)    int32 per-sequence rope indices
      - ``cu_int64``  optional int64 cumulative boundaries
      - ``chunk_indices`` optional precomputed ``(sequence, chunk)`` pairs,
        keyed by chunk size
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
    cu_int64: torch.Tensor | None = field(default=None, compare=False)
    chunk_indices: dict[int, torch.Tensor] = field(
        default_factory=dict, compare=False)

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
        return cls.uniform(d.seq_len, d.max_tokens // d.seq_len)

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

    def on(self, device, *, chunk_sizes: tuple[int, ...] = ()) -> "Segments":
        """Materialize ``cu`` / ``positions`` on ``device`` ONCE and return a
        Segments carrying them as fields. Optional chunk metadata is prepared
        from host ``lengths`` here as well; a task must never derive it from a
        device ``cu`` tensor because operations such as ``repeat_interleave``
        read a CUDA count back to the host and synchronize the whole device.

        Pinned staging + non-blocking copies avoid pageable H2D transfers.
        Calls are idempotent when the base tensors and every requested chunk
        size already live on ``device``.
        """
        target = torch.device(device)
        requested = tuple(sorted({int(size) for size in chunk_sizes}))
        if any(size <= 0 for size in requested):
            raise ValueError(f"chunk sizes must be positive, got {requested}")
        base_ready = (
            self.cu is not None
            and self.positions is not None
            and self.cu.device == target
            and self.positions.device == target
        )
        chunks_ready = all(
            size in self.chunk_indices
            and self.chunk_indices[size].device == target
            for size in requested
        )
        int64_ready = not requested or (
            self.cu_int64 is not None and self.cu_int64.device == target)
        if base_ready and chunks_ready and int64_ready:
            return self

        boundaries = self.boundaries
        if base_ready:
            cu = self.cu
            positions = self.positions
        else:
            cu_host = torch.tensor(
                boundaries, dtype=torch.int32).pin_memory()
            if self.lengths:
                pos_host = torch.cat(
                    [torch.arange(n, dtype=torch.int32) for n in self.lengths]
                ).pin_memory()
            else:
                pos_host = torch.empty(0, dtype=torch.int32).pin_memory()
            cu = cu_host.to(target, non_blocking=True)
            positions = pos_host.to(target, non_blocking=True)

        cu_int64 = self.cu_int64
        if requested and not int64_ready:
            cu64_host = torch.tensor(
                boundaries, dtype=torch.int64).pin_memory()
            cu_int64 = cu64_host.to(target, non_blocking=True)

        prepared = dict(self.chunk_indices)
        for size in requested:
            existing = prepared.get(size)
            if existing is not None and existing.device == target:
                continue
            pairs = [
                (sequence, chunk)
                for sequence, length in enumerate(self.lengths)
                for chunk in range(math.ceil(length / size))
            ]
            host = torch.tensor(pairs, dtype=torch.int64)
            if not pairs:
                host = host.reshape(0, 2)
            prepared[size] = host.pin_memory().to(target, non_blocking=True)
        return replace(
            self,
            cu=cu,
            positions=positions,
            cu_int64=cu_int64,
            chunk_indices=prepared,
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
            chunk_sizes = tuple(getattr(dims, "segment_chunk_sizes", ())) \
                if dims is not None else ()
            by_host[host] = host.on(
                f"cuda:{ctx.backend.device}", chunk_sizes=chunk_sizes)
        host = by_host[host]
    cache[round_key] = host
    return host
