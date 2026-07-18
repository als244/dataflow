"""Segments: the packed-round value object (per-sequence lengths,
device cu/positions materialized once by the engine prologue). Owned
by the ENGINE side: run_args carry it, prologue tasks build it, and
the runtime's uniform_segments helper constructs it — workload code
imports it from here (an allowed runtime ABI).
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
    ``.on`` is called exactly once per round in the engine's run prologue
    (and once per golden forward); every stage/op downstream then reads
    ``seg.cu`` / ``seg.positions`` as plain attributes. Nothing rebuilds a
    device tensor from host data mid-round — that would be a hidden
    host->device sync (the aten-hidden-syncs discipline). ``cu`` /
    ``positions`` are excluded from equality/hash (identity is ``lengths``).

    Replaces the old seq_spec (int | tuple) + the seq_lens_of /
    seq_bounds_of / positions_for / attn_meta free-function family.
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
    def of_dims(cls, d) -> "Segments":
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
