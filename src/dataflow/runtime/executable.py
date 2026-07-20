"""Task-executable contract (runtime side).

The runtime hands an executable everything it may touch and nothing else:
buffers for declared inputs/outputs/mutates (+ optional workspace), the
compute stream, and the backend. The executable must only enqueue
device work on `ctx.stream` — no synchronization, no globals; scratch
allocation only through torch's caching allocator, declared via the
kernel registry's `allocates=`/`workspace=` fields (docs/task-contract.md).

The tasks layer provides real implementations resolved by
`(compute_block_key, block_params)`. `SyntheticExecutable` models a task of
known duration and is the workhorse for parity gates (fake backend) and,
later, calibrated spin kernels (cuda backend).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from dataflow.core import TaskSpec
from .device.base import Buffer, DeviceBackend, Stream


@dataclass(frozen=True)
class TaskContext:
    task: TaskSpec
    stream: Stream
    inputs: Mapping[str, Buffer]
    outputs: Mapping[str, Buffer]
    mutates: Mapping[str, Buffer]
    backend: DeviceBackend
    workspace: Buffer | None = None
    # per-run opaque parameters (engine service run(args=...)); the
    # engine never interprets them — executables read what they need
    # (e.g. a global step counter a task family keys on). IMMUTABLE
    # after the run-start prologue: tasks must never write run_args.
    run_args: Mapping[str, object] | None = None
    # small MUTABLE shared runtime state, one dict per run — written by
    # round-boundary tasks (canonical member: "current_round", published
    # by RoundPrologue), readable by every task. The object-backed
    # current_round_{s}_{r} outputs carry the ORDERING; this dict is the
    # ergonomic read.
    run_values: dict | None = None
    # live peer-group handles ({name -> {rank, world, backend, ...}}),
    # the WHOLE daemon table per run; absent name => task skips comm
    # (valid only for rank-complete lowerings — spec)
    groups: dict | None = None


class Executable(Protocol):
    def launch(self, ctx: TaskContext) -> None:
        """Enqueue this task's device work on ctx.stream. Never synchronize."""
        ...


@dataclass(frozen=True)
class SyntheticExecutable:
    """Models a task as `runtime_us` of opaque stream work (fake backend)."""

    runtime_us: float

    def launch(self, ctx: TaskContext) -> None:
        ctx.backend.advance_stream(ctx.stream, self.runtime_us)


ExecutableResolver = Callable[[TaskSpec], Executable]


def synthetic_resolver(task: TaskSpec) -> Executable:
    """Default resolver: every task modeled by its planned runtime."""
    return SyntheticExecutable(task.runtime_us)
