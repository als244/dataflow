"""Static buffer placement: offline packing of fast-memory instances.

The annotated plan fully determines every fast-memory allocation's birth and
death, so placement is an OFFLINE interval-packing problem — not an online
allocator gamble. A fake-backend dry run records the instance stream
(`(object_id, incarnation)` with ordinal lifetimes); `compute_placement`
packs instances (largest first, lowest non-conflicting offset) and FAILS AT
PLANNING TIME if the packing exceeds the budget. Execution then follows the
placement verbatim: fragmentation at runtime becomes impossible rather than
unlikely, and the headroom/overflow heuristics become unnecessary.

Real-time safety: actual completion order can differ from dry-run ordinals,
so the pool's assigned mode refuses to hand out an offset while a prior
overlapping instance is still live ("busy") — callers treat it exactly like
a capacity stall. Under strict pacing the blocker is always an in-flight
transfer or an earlier task, so progress is guaranteed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_ALIGN = 512  # vendor-grade: cuBLASLt algo choice is alignment-sensitive

InstanceKey = tuple[str, int]  # (object_id, incarnation)


class PlacementError(RuntimeError):
    pass


@dataclass
class InstanceRecord:
    key: InstanceKey
    size_bytes: int
    birth: int          # ordinal of pool.get in the dry run
    death: int          # ordinal of pool.put (2**62 if never freed)


@dataclass
class PlacementRecorder:
    """Attached to the dry run's pool: logs the fast-instance stream."""

    ordinal: int = 0
    live: dict[InstanceKey, InstanceRecord] = field(default_factory=dict)
    done: list[InstanceRecord] = field(default_factory=list)

    def on_get(self, key: InstanceKey, size_bytes: int) -> None:
        self.ordinal += 1
        self.live[key] = InstanceRecord(
            key=key, size_bytes=size_bytes, birth=self.ordinal, death=2**62,
        )

    def on_put(self, key: InstanceKey) -> None:
        self.ordinal += 1
        rec = self.live.pop(key, None)
        if rec is not None:
            rec.death = self.ordinal
            self.done.append(rec)

    def instances(self) -> list[InstanceRecord]:
        return self.done + list(self.live.values())


@dataclass(frozen=True)
class Placement:
    offsets: dict[InstanceKey, int]
    sizes: dict[InstanceKey, int]
    extent_bytes: int          # physical footprint the packing proved
    load_bytes: int            # ledger peak (lower bound; extent/load = geometry tax)
    physical_limit_bytes: int

    @property
    def overhead(self) -> float:
        return self.extent_bytes / max(self.load_bytes, 1)


def _pack(instances: list[InstanceRecord]) -> tuple[int, dict[InstanceKey, int]]:
    placed: list[tuple[int, int, InstanceRecord]] = []
    offsets: dict[InstanceKey, int] = {}
    extent = 0
    for rec in instances:
        size = (rec.size_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        conflicts = sorted(
            (off, end) for off, end, other in placed
            if not (other.death <= rec.birth or rec.death <= other.birth)
        )
        candidate = 0
        for off, end in conflicts:
            if candidate + size <= off:
                break
            candidate = max(candidate, end)
        offsets[rec.key] = candidate
        placed.append((candidate, candidate + size, rec))
        extent = max(extent, candidate + size)
    return extent, offsets


def compute_placement(
    recorder: PlacementRecorder,
    physical_limit_bytes: int,
    *,
    restarts: int = 48,
) -> Placement:
    """Offline interval packing with full lifetime knowledge.

    Tries deterministic orderings plus seeded random restarts and keeps the
    smallest extent. NOTE the honest limitation of contiguous placement: the
    optimal extent can genuinely exceed the peak concurrent load (offline
    dynamic-storage-allocation geometry) — the residual `overhead` ratio is a
    real systems tax, reported first-class. Eliminating it entirely requires
    non-contiguous backing (VMM chunk mapping) — the planned follow-up.
    Validation is against PHYSICAL VRAM, at planning time; the logical budget
    stays enforced by the ledger during execution.
    """
    import random

    base = recorder.instances()
    # peak concurrent load = lower bound for any placement
    events: list[tuple[int, int]] = []
    for r in base:
        events.append((r.birth, r.size_bytes))
        events.append((r.death, -r.size_bytes))
    load, cur = 0, 0
    for _, delta in sorted(events):
        cur += delta
        load = max(load, cur)

    orders: list[list[InstanceRecord]] = [
        sorted(base, key=lambda r: (-r.size_bytes, r.birth, r.key)),
        sorted(base, key=lambda r: (-(r.death - r.birth) * r.size_bytes, r.key)),
        sorted(base, key=lambda r: (r.birth, -r.size_bytes, r.key)),
    ]
    rng = random.Random(0)
    for _ in range(restarts):
        shuffled = list(base)
        rng.shuffle(shuffled)
        orders.append(shuffled)

    best_extent, best_offsets = None, None
    for order in orders:
        extent, offsets = _pack(order)
        if best_extent is None or extent < best_extent:
            best_extent, best_offsets = extent, offsets
    assert best_extent is not None and best_offsets is not None

    if best_extent > physical_limit_bytes:
        raise PlacementError(
            f"static placement needs {best_extent} bytes (load lower bound "
            f"{load}) but only {physical_limit_bytes} physical bytes are "
            f"available — not packable on this device; lower the budget or "
            f"re-plan (this failure is at PLANNING time by design)"
        )
    return Placement(
        offsets=best_offsets,
        sizes={r.key: r.size_bytes for r in base},
        extent_bytes=best_extent,
        load_bytes=load,
        physical_limit_bytes=physical_limit_bytes,
    )
