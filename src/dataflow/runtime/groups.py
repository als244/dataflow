"""GroupHandle: the task-side ABI for peer-group collectives.

Tasks reach these through ``TaskContext.groups[name]`` (the daemon's
whole live-group table, injected per run). The handle is
engine-agnostic: ``rank``/``world``/``backend`` are plain data; the
collective methods delegate to a comm implementation the SERVICE
attaches (hostmem staging or nccl) — the engine never interprets them.

ABI (spec §4): every method takes DEVICE tensors and is ASYNCHRONOUS
on ``handle.stream`` — the caller owns event edges between its compute
stream and the group stream. op is SUM everywhere (the
global-denominator convention makes the plain sum exact). A handle
with no comm attached (fake boots, backend bring-up failure) raises
loudly on first use — never silently skips.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroupHandle:
    name: str
    rank: int
    world: int
    backend: str
    members: tuple = ()
    coordinator: str = ""
    stream: object = None          # torch.cuda.Stream (service-created)
    comm: object = None            # backend impl (service-attached)
    meta: dict = field(default_factory=dict)

    def require_comm(self):
        if self.comm is None:
            raise RuntimeError(
                f"group {self.name!r}: no comm backend attached "
                f"(backend={self.backend!r}) — collectives unavailable")
        return self.comm

    def allreduce(self, tensor) -> None:
        """In-place SUM across the group (rank-ordered fp32
        accumulation; result bitwise-identical on every member)."""
        self.require_comm().allreduce(tensor)

    def broadcast(self, tensor, root: int) -> None:
        self.require_comm().broadcast(tensor, root)

    def reduce(self, tensor, root: int) -> None:
        """SUM lands in-place at ROOT only; other ranks' tensors are
        left untouched."""
        self.require_comm().reduce(tensor, root)

    def reduce_scatter(self, full, out) -> None:
        """full: the whole per-group buffer (equal slices, rank-major);
        out: THIS rank's slice, receives the summed slice."""
        self.require_comm().reduce_scatter(full, out)

    def all_gather(self, own_slice, full) -> None:
        """full receives every rank's slice, rank-major."""
        self.require_comm().all_gather(own_slice, full)
