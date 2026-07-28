"""Source policies: WHO saves WHICH bytes, compiled from the
responsibility map into engine slice lists and record slice entries.

Every persistent object falls in exactly one of three GENERAL
classes, decided by the metadata handed in — nothing here knows any
particular parallelism scheme by name:

REPLICATED (the object appears in the responsibility plan): every
source holds the same bytes. ``simple`` (the default) has every
source save its whole copy, so each source's snapshot restores that
rank with zero cross-source shipping; the copies overlap on the
record and the equal-hash rule across them certifies replication on
every save. ``dedup`` saves one copy total instead, as the plan's
disjoint responsibility slices — trading self-sufficiency and
redundancy for disk.

ELEMENT-SHARDED (the object's root appears in ``opt_slices``):
sources hold disjoint element ranges of paired state, resident in
the slice+tail layout ``opt_state_slice_layout`` describes — aligned
per-slot fields plus, when the element count doesn't divide the
world, tail fields every source updates redundantly. The logical
object is the tight world-spanning concatenation of the slot spaces;
each source's fields map at element-derived offsets, and the
redundant tail fields overlap across sources so the drift
certificate covers them too. Alignment padding between fields is
layout, not state — it maps to no logical byte.

SOURCE-PRIVATE (everything else): state each source accumulates on
its own, with nothing synchronizing the copies — same-size copies
across sources are NOT replicas, so each source's copy becomes its
own source-qualified logical object rather than being falsely
certified. Replication and its drift certificate apply exactly to
the plan's objects.

Each source also emits its resident-object inventory ({id: bytes});
the composer records it per snapshot so the rank view can recreate
objects at their exact local sizes, padding included.
"""
from __future__ import annotations

from ..blocks.layouts import opt_state_slice_layout


def opt_root(oid: str) -> str | None:
    """The parameter root an optimizer-state object pairs with
    ("O_3" -> "W_3", "O_embed" -> "W_embed"), or None."""
    if oid.startswith("O_"):
        return "W_" + oid[2:]
    return None


def compile_source_policy(*, policy: str, world: int, source_specs: dict,
                          plan: dict, opt_slices: dict | None = None):
    """Compile per-source save sets.

    ``source_specs``: {source_key: [(id, size_bytes), ...]} — each
    source's persisted objects at ITS resident sizes (the marker
    filter applied to its program). ``plan``: the responsibility map.
    ``opt_slices``: element-shard dict {root: {n_slice, n_tail,
    opt_dtype}} when optimizer state is sharded, else None.

    Returns (logical_objects, per_source): logical_objects is the
    record's {id: {"bytes": N}}; per_source[k] carries "slices" (the
    engine wire lists), "record" (record slice entries without
    hashes — the composer fills those from snapshot statuses), and
    "objects" (the source's resident sizes, recorded per snapshot so
    the rank view recreates local geometry exactly).
    """
    if policy not in ("simple", "dedup"):
        raise ValueError(f"unknown source policy {policy!r}")
    logical: dict = {}
    per_source = {k: {"slices": [], "record": [],
                      "objects": {oid: int(size)
                                  for oid, size in source_specs[k]}}
                  for k in sorted(source_specs)}
    sizes_by_source = {w: dict(specs)
                       for w, specs in source_specs.items()}

    for source, specs in sorted(source_specs.items()):
        for oid, size in specs:
            root = opt_root(oid)
            if opt_slices and root in opt_slices:
                add_sharded(per_source[source], logical, oid, size,
                            source, world, opt_slices, root)
            elif oid in plan:
                sizes = {w: held.get(oid)
                         for w, held in sizes_by_source.items()}
                add_replicated(per_source[source], logical, oid, size,
                               source, policy, plan, sizes)
            else:
                add_private(per_source[source], logical, oid, size,
                            source)
    return logical, per_source


def add_replicated(out: dict, logical: dict, oid: str, size: int,
                   source: int, policy: str, plan: dict,
                   sizes: dict) -> None:
    others = {w: n for w, n in sizes.items() if n is not None}
    if any(n != size for n in others.values()):
        raise ValueError(
            f"{oid}: sources disagree on size ({others}) — neither "
            f"replicated nor a known sharding; the policy refuses to "
            f"guess")
    known = logical.setdefault(oid, {"bytes": int(size)})
    if known["bytes"] != size:
        raise ValueError(f"{oid}: logical size conflict "
                         f"{known['bytes']} != {size}")
    if policy == "dedup" and oid in plan:
        entries = [e for e in plan[oid]
                   if e["rank"] == source and e["role"] == "responsible"]
        for e in entries:
            lo, hi = int(e["lo"]), int(e["hi"])
            out["slices"].append({"id": oid, "src": [lo, hi]})
            out["record"].append({"logical": oid,
                                  "snapshot_range": [lo, hi],
                                  "object_range": [lo, hi]})
        return
    out["slices"].append({"id": oid})
    out["record"].append({"logical": oid,
                          "snapshot_range": [0, size],
                          "object_range": [0, size]})


def add_private(out: dict, logical: dict, oid: str, size: int,
                source: int) -> None:
    """Source-PRIVATE state: an object outside the responsibility
    plan and outside the element-shard pairing accumulates per
    source, with nothing synchronizing the copies. Same-size copies
    across sources are NOT replicas here, so certifying them under
    the drift rule would refuse legitimate saves; instead each
    source's copy becomes its own source-qualified logical object
    (``<id>@<source>``), whole. The rank view resolves the bare id
    to the source's qualified object and restores it under the bare
    local name."""
    lid = f"{oid}@{source}"
    known = logical.setdefault(lid, {"bytes": int(size)})
    if known["bytes"] != size:
        raise ValueError(f"{lid}: logical size conflict "
                         f"{known['bytes']} != {size}")
    out["slices"].append({"id": oid, "logical_id": lid,
                          "logical_bytes": int(size)})
    out["record"].append({"logical": lid,
                          "snapshot_range": [0, int(size)],
                          "object_range": [0, int(size)]})


def add_sharded(out: dict, logical: dict, oid: str, size: int,
                source: int, world: int, opt_slices: dict,
                root: str) -> None:
    """Slot structure comes entirely from the layout's fields — one
    ``<slot>_slice`` field per state slot, each optionally paired
    with a ``<slot>_tail`` — so any optimizer's slot set (one slot,
    two, more; any dtype) compiles through the same arithmetic. The
    logical object concatenates per-slot spaces in field order; a
    slot's space is the sources' slice extents in source order plus
    its tail."""
    sh = opt_slices[root]
    layout = opt_state_slice_layout(int(sh["n_slice"]),
                                    int(sh["n_tail"]),
                                    sh["opt_dtype"])
    if size != layout.total_bytes:
        raise ValueError(
            f"{oid}: source {source} holds {size} B but its element-"
            f"shard layout is {layout.total_bytes} B "
            f"({[f.name for f in layout.fields]})")
    slice_fields = [f for f in layout.fields
                    if f.name.endswith("_slice")]
    tail_fields = {f.name.removesuffix("_tail"): f
                   for f in layout.fields if f.name.endswith("_tail")}
    pieces = []
    cursor = 0
    for f in slice_fields:
        slot = f.name.removesuffix("_slice")
        tail = tail_fields.get(slot)
        pieces.append((f, [cursor + source * f.nbytes,
                           cursor + (source + 1) * f.nbytes]))
        cursor += world * f.nbytes
        if tail is not None:
            # every source updates the world-remainder tail
            # redundantly: the overlapping tail slices are replicas,
            # certified interchangeable by the drift rule
            pieces.append((tail, [cursor, cursor + tail.nbytes]))
            cursor += tail.nbytes
    total_bytes = cursor
    known = logical.setdefault(oid, {"bytes": total_bytes})
    if known["bytes"] != total_bytes:
        raise ValueError(f"{oid}: logical size conflict "
                         f"{known['bytes']} != {total_bytes}")
    for f, dst in pieces:
        src = [f.offset_bytes, f.offset_bytes + f.nbytes]
        out["slices"].append({"id": oid, "src": src,
                              "logical_id": oid, "dst": dst,
                              "logical_bytes": total_bytes})
        out["record"].append({"logical": oid,
                              "snapshot_range": list(src),
                              "object_range": list(dst)})
