"""NcclComm: the GroupHandle backend over libnccl — the production
collective lane.

Simpler than hostmem by construction: every op enqueues DIRECTLY on
the group stream over DEVICE buffers — no D2H/H2D staging, no worker
thread, no release flag. Completion is plain stream order, so the
optimizer blocks' existing event edges work unchanged. The producer
contract is the same as hostmem's: the caller orders tensor
production against the group stream (training records grads_ready on
the compute stream and waits it on gh.stream).

Bootstrap is collective and BLOCKING (ncclCommInitRank returns only
when every rank has called it), so it happens at GROUP CREATION time
on dedicated bring-up threads — never lazily at first use — followed
immediately by a tiny warm-up allreduce that both loads NCCL's
kernels while the device is unparked (the established first-launch
trap) and proves the communicator end to end before create_peer_group
returns.
"""
from __future__ import annotations

import torch

from . import nccl

TORCH_TO_NCCL = {
    torch.bfloat16: nccl.DTYPE_BY_NAME["bf16"],
    torch.float16: nccl.DTYPE_BY_NAME["f16"],
    torch.float32: nccl.DTYPE_BY_NAME["f32"],
    torch.float64: nccl.DTYPE_BY_NAME["f64"],
    torch.int32: nccl.DTYPE_BY_NAME["int32"],
    torch.int64: nccl.DTYPE_BY_NAME["int64"],
    torch.uint8: nccl.DTYPE_BY_NAME["uint8"],
    torch.int8: nccl.DTYPE_BY_NAME["int8"],
}


class NcclComm:
    """One per (group, daemon). World-N."""

    def __init__(self, group_name: str, rank: int, world: int,
                 uid: bytes):
        self.lib = nccl.get_lib()
        self.group = group_name
        self.rank = rank
        self.world = world
        self.stream = torch.cuda.Stream()
        self.dead: str | None = None
        self.comm = self.lib.comm_init_rank(world, uid, rank)
        self.warm_up()

    def warm_up(self) -> None:
        """Collective warm-up at bootstrap: loads NCCL kernels while
        the device is unparked and proves the communicator; every
        rank runs this inside its bring-up thread concurrently."""
        probe = torch.ones(8, dtype=torch.float32, device="cuda")
        self.lib.all_reduce(probe.data_ptr(), probe.data_ptr(),
                            probe.numel(), nccl.DTYPE_BY_NAME["f32"],
                            self.comm, self.stream.cuda_stream)
        self.stream.synchronize()
        expected = float(self.world)
        if abs(float(probe[0]) - expected) > 1e-6:
            raise nccl.NcclError(
                f"group {self.group!r}: bootstrap allreduce returned "
                f"{float(probe[0])}, expected {expected}")

    def guard(self, tensor) -> int:
        if self.dead:
            raise RuntimeError(f"group {self.group} dead: {self.dead}")
        dt = TORCH_TO_NCCL.get(tensor.dtype)
        if dt is None:
            raise RuntimeError(f"nccl: unsupported dtype {tensor.dtype}")
        return dt

    def allreduce(self, tensor, out=None):
        dt = self.guard(tensor)
        dst = tensor if out is None else out
        self.lib.all_reduce(tensor.data_ptr(), dst.data_ptr(),
                            tensor.numel(), dt, self.comm,
                            self.stream.cuda_stream)
        return None   # no staging: the input is stream-ordered free

    def broadcast(self, tensor, root: int) -> None:
        dt = self.guard(tensor)
        self.lib.broadcast(tensor.data_ptr(), tensor.data_ptr(),
                           tensor.numel(), dt, root, self.comm,
                           self.stream.cuda_stream)

    def reduce(self, tensor, root: int) -> None:
        dt = self.guard(tensor)
        self.lib.reduce(tensor.data_ptr(), tensor.data_ptr(),
                        tensor.numel(), dt, root, self.comm,
                        self.stream.cuda_stream)

    def reduce_scatter(self, full, out) -> None:
        dt = self.guard(full)
        if out.numel() * self.world != full.numel():
            raise ValueError(
                f"reduce_scatter: out({out.numel()}) x world"
                f"({self.world}) != full({full.numel()})")
        self.lib.reduce_scatter(full.data_ptr(), out.data_ptr(),
                                out.numel(), dt, self.comm,
                                self.stream.cuda_stream)

    def all_gather(self, own_slice, full) -> None:
        dt = self.guard(own_slice)
        if own_slice.numel() * self.world != full.numel():
            raise ValueError(
                f"all_gather: slice({own_slice.numel()}) x world"
                f"({self.world}) != full({full.numel()})")
        self.lib.all_gather(own_slice.data_ptr(), full.data_ptr(),
                            own_slice.numel(), dt, self.comm,
                            self.stream.cuda_stream)

    def fail(self, why: str) -> None:
        self.dead = why
        if self.comm:
            self.lib.comm_abort(self.comm)
            self.comm = 0

    def close(self) -> None:
        if self.comm:
            if self.dead:
                self.lib.comm_abort(self.comm)
            else:
                self.lib.comm_destroy(self.comm)
            self.comm = 0
