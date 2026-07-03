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


def compute_lookahead(
    program: Program,
    placement=None,
    *,
    fast_capacity: int | None = None,
) -> LookaheadPlan:
    """Conservative sync-point analysis (see module docstring).

    Assigned mode (``placement`` given): the only dispatcher-side hazard is
    an output whose placed offset overlaps an instance whose death has not
    been SETTLED (its pool.put happens at a completion token the dispatcher
    may not have processed). Capacity is not a dispatcher concern there —
    physical safety is the placement proof, and the ledger stays
    token-paced so transfer admission is identical to strict mode (v1
    pre-charged at enqueue and starved prefetch admission: −20% measured).

    Dynamic mode (no placement): the ledger IS the physical guard, so
    reservations that need not-yet-credited frees become sync points
    (conservative: no frees credited between syncs).
    """
    capacity = fast_capacity or program.fast_memory_capacity
    sizes = {o.id: o.size_bytes for o in program.initial_objects}
    for t in program.tasks:
        for o in t.outputs:
            sizes[o.id] = o.size_bytes

    sync_points: set[int] = set()

    if placement is None:
        # --- dynamic mode: capacity-only analysis -------------------------------
        settled = sum(
            o.size_bytes for o in program.initial_objects if o.location == "fast"
        )
        window_charged = 0
        pending_free = 0
        for i, task in enumerate(program.tasks):
            fast_out = sum(
                o.size_bytes for o in task.outputs if o.location == "fast"
            )
            if capacity is not None and settled + window_charged + fast_out > capacity:
                sync_points.add(i)
                settled += window_charged - pending_free
                window_charged = 0
                pending_free = 0
            window_charged += fast_out
            for oid in task.releases_after:
                pending_free += sizes.get(oid, 0)
            for d in task.offload_after:
                pending_free += sizes.get(d.object_id, 0)
            for d in task.prefetch_after:
                window_charged += sizes.get(d.object_id, 0)
        return LookaheadPlan(
            sync_points=frozenset(sync_points), total_tasks=len(program.tasks)
        )

    # --- assigned mode: offset-conflict analysis --------------------------------
    incarnation: dict[str, int] = {}
    alive: dict[tuple, tuple[int, int]] = {}      # key -> (lo, hi)
    unsettled: dict[tuple, tuple[int, int]] = {}  # died since last sync

    def birth(oid: str) -> tuple | None:
        key = (oid, incarnation.get(oid, 0))
        incarnation[oid] = key[1] + 1
        off = placement.offsets.get(key)
        if off is None:
            return None
        rng = (off, off + placement.sizes[key])
        alive[key] = rng
        return rng

    def death(oid: str) -> None:
        # the currently-alive (latest) incarnation of this object dies;
        # its put settles only at a token -> unsettled until next sync
        for inc in range(incarnation.get(oid, 0) - 1, -1, -1):
            key = (oid, inc)
            rng = alive.pop(key, None)
            if rng is not None:
                unsettled[key] = rng
                return

    # initial fast objects occupy their instance-0 offsets from the start
    for o in program.initial_objects:
        if o.location == "fast":
            birth(o.id)

    def candidate_ranges(task) -> list[tuple[int, int]]:
        out = []
        for o in task.outputs:
            if o.location != "fast":
                continue
            key = (o.id, incarnation.get(o.id, 0))
            off = placement.offsets.get(key)
            if off is not None:
                out.append((off, off + placement.sizes[key]))
        return out

    for i, task in enumerate(program.tasks):
        cands = candidate_ranges(task)
        if cands and unsettled:
            hit = any(
                lo < hi2 and lo2 < hi
                for lo, hi in cands
                for lo2, hi2 in unsettled.values()
            )
            if hit:
                sync_points.add(i)
                unsettled.clear()   # drain applies every pending put

        for o in task.outputs:
            if o.location == "fast":
                birth(o.id)
        for oid in task.releases_after:
            death(oid)
        for d in task.offload_after:
            death(d.object_id)      # fast copy freed at d2h completion
        for d in task.prefetch_after:
            birth(d.object_id)      # dst occupies from transfer start

    return LookaheadPlan(sync_points=frozenset(sync_points), total_tasks=len(program.tasks))
