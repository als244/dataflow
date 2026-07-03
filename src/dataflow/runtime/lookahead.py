"""Dispatch-ahead safety analysis: which task boundaries need the host?

Strict pacing waits for task N's completion token before dispatching N+1 —
paying ~0.8 ms of exposed host time per boundary (see
docs/notes/perf-headroom.md). But most of that waiting is bookkeeping, not
a data dependency: kernels on one compute stream execute in FIFO order, so
compute-produced inputs are ordered by the stream itself, and
transfer-produced inputs can be ordered by device-side event waits.

The host is genuinely required at a boundary only when task K's ADMISSION
depends on completion effects still in the lookahead window:

- **capacity**: K's output reservation only fits if releases/offloads of
  window tasks have been credited (ledger frees happen at token time);
- **placed offsets**: K's assigned offset ranges overlap an instance whose
  death (pool.put) is inside the window — reuse before the put is applied
  would trip assigned-mode exclusion.

Both are decidable at planning time with a conservative rule: between sync
points, charges accrue at enqueue and NO frees are credited. Walking the
chain, any task whose reservation would exceed capacity — or whose output
offsets overlap an instance not dead before the previous sync point —
becomes a sync point (the dispatcher drains all outstanding tokens there,
which applies every pending credit, then proceeds).

The result deliberately over-syncs (frees are credited only at syncs), so
execution can only be MORE ordered than the analysis assumed — safety is
one-directional. At generous budgets almost no capacity syncs appear; at
tight budgets they appear where the simulator predicts stalls anyway.

Runtime-dynamic hazards that cannot be decided statically stay in the
engine as soft syncs (drain until resolvable, then device-wait):
prefetched inputs whose transfer is not yet enqueued, and mutations of
objects with an offload still in flight.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dataflow.core import Program


@dataclass(frozen=True)
class LookaheadPlan:
    """Sync-point decisions for one annotated program."""

    sync_points: frozenset[int]          # task indices that must drain tokens
    total_tasks: int

    @property
    def free_running_fraction(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return 1.0 - len(self.sync_points) / self.total_tasks


@dataclass
class _Window:
    charged: int = 0                      # bytes charged since last sync
    live_offsets: list = field(default_factory=list)  # (lo, hi) placed ranges born in window
    dying: set = field(default_factory=set)           # instance keys dying in window


def compute_lookahead(
    program: Program,
    placement=None,
    *,
    fast_capacity: int | None = None,
) -> LookaheadPlan:
    """Conservative sync-point analysis (see module docstring).

    ``placement`` enables offset-overlap checks (assigned mode); without it
    only capacity analysis applies (dynamic-slab mode is pointer-agnostic:
    the pool serves any free bytes, and stream FIFO orders reuse of a
    recycled buffer after its releasing task — but capacity credits still
    lag, so those sync points remain).
    """
    capacity = fast_capacity or program.fast_memory_capacity
    sizes = {o.id: o.size_bytes for o in program.initial_objects}
    for t in program.tasks:
        for o in t.outputs:
            sizes[o.id] = o.size_bytes

    # ledger state as of the LAST sync (all credits applied there)
    settled = sum(o.size_bytes for o in program.initial_objects if o.location == "fast")
    # per-object pending frees between syncs: releases + fast-frees from
    # offload completions (dead-everywhere handled by release accounting)
    incarnation: dict[str, int] = {}
    if placement is not None:
        deaths: dict[tuple, int] = {}   # instance key -> task index of death
        births: dict[tuple, int] = {}
        # reconstruct instance lifetimes in chain order (matches recorder
        # semantics: task outputs + h2d dst gets advance incarnations; we
        # approximate with task-order events, which is the same total order
        # the packer used)
    sync_points: set[int] = set()
    window = _Window()
    pending_free = 0                     # bytes freed inside the window (not credited)

    def placed_ranges(task_index: int, task) -> list[tuple[int, int]]:
        out = []
        if placement is None:
            return out
        for o in task.outputs:
            if o.location != "fast":
                continue
            key = (o.id, incarnation.get(o.id, 0))
            off = placement.offsets.get(key)
            if off is not None:
                out.append((off, off + o.size_bytes))
        return out

    def overlaps(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
        return any(lo < hi2 and lo2 < hi for lo, hi in a for lo2, hi2 in b)

    for i, task in enumerate(program.tasks):
        fast_out = sum(o.size_bytes for o in task.outputs if o.location == "fast")
        ranges = placed_ranges(i, task)

        need_sync = False
        if capacity is not None and settled + window.charged + fast_out > capacity:
            need_sync = True
        if not need_sync and ranges and overlaps(ranges, window.live_offsets):
            # output offset overlaps an instance born in this window whose
            # death has not been settled — conservative sync
            need_sync = True

        if need_sync:
            sync_points.add(i)
            settled += window.charged - pending_free
            window = _Window()
            pending_free = 0

        window.charged += fast_out
        window.live_offsets.extend(ranges)
        for o in task.outputs:
            if o.location == "fast":
                incarnation[o.id] = incarnation.get(o.id, 0) + 1
        # releases and offloads free fast bytes at their completion tokens —
        # credit them only at the next sync
        for oid in task.releases_after:
            pending_free += sizes.get(oid, 0)
        for d in task.offload_after:
            pending_free += sizes.get(d.object_id, 0)
        # prefetches charge fast at transfer START (token-side); treat their
        # destination bytes as window charges so capacity stays conservative
        for d in task.prefetch_after:
            window.charged += sizes.get(d.object_id, 0)
            incarnation[d.object_id] = incarnation.get(d.object_id, 0) + 1

    return LookaheadPlan(sync_points=frozenset(sync_points), total_tasks=len(program.tasks))
