"""Source policies: WHO saves WHICH bytes, compiled from the
responsibility map into engine slice lists and record slice entries.

Every persistent object falls in exactly one of three GENERAL
classes, decided by the metadata handed in — nothing here knows any
particular parallelism scheme by name:

REPLICATED (the object appears in the responsibility plan): every
writer holds the same bytes. ``simple`` (the default) has every
writer save its whole copy, so each writer's snapshot restores that
rank with zero cross-writer shipping; the copies overlap on the
record and the equal-hash rule across them certifies replication on
every save. ``dedup`` saves one copy total instead, as the plan's
disjoint responsibility slices — trading self-sufficiency and
redundancy for disk.

ELEMENT-SHARDED (the object's root appears in ``opt_slices``):
writers hold disjoint element ranges of paired state, resident in
the slice+tail layout ``opt_state_slice_layout`` describes — aligned
per-slot fields plus, when the element count doesn't divide the
world, tail fields every writer updates redundantly. The logical
object is the tight world-spanning concatenation of the slot spaces;
each writer's fields map at element-derived offsets, and the
redundant tail fields overlap across writers so the drift
certificate covers them too. Alignment padding between fields is
layout, not state — it maps to no logical byte.

WRITER-PRIVATE (everything else): state each writer accumulates on
its own, with nothing synchronizing the copies — same-size copies
across writers are NOT replicas, so each writer's copy becomes its
own writer-qualified logical object rather than being falsely
certified. Replication and its drift certificate apply exactly to
the plan's objects.

Each writer also emits its resident-object inventory ({id: bytes});
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


def compile_source_policy(*, policy: str, world: int, writer_specs: dict,
                          plan: dict, opt_slices: dict | None = None):
    """Compile per-writer save sets.

    ``writer_specs``: {writer_index: [(id, size_bytes), ...]} — each
    writer's persisted objects at ITS resident sizes (the marker
    filter applied to its program). ``plan``: the responsibility map.
    ``opt_slices``: element-shard dict {root: {n_slice, n_tail,
    opt_dtype}} when optimizer state is sharded, else None.

    Returns (logical_objects, per_writer): logical_objects is the
    record's {id: {"bytes": N}}; per_writer[k] carries "slices" (the
    engine wire lists), "record" (record slice entries without
    hashes — the composer fills those from snapshot statuses), and
    "objects" (the writer's resident sizes, recorded per snapshot so
    the rank view recreates local geometry exactly).
    """
    if policy not in ("simple", "dedup"):
        raise ValueError(f"unknown source policy {policy!r}")
    logical: dict = {}
    per_writer = {k: {"slices": [], "record": [],
                      "objects": {oid: int(size)
                                  for oid, size in writer_specs[k]}}
                  for k in sorted(writer_specs)}
    sizes_by_writer = {w: dict(specs)
                       for w, specs in writer_specs.items()}

    for writer, specs in sorted(writer_specs.items()):
        for oid, size in specs:
            root = opt_root(oid)
            if opt_slices and root in opt_slices:
                add_sharded(per_writer[writer], logical, oid, size,
                            writer, world, opt_slices, root)
            elif oid in plan:
                sizes = {w: held.get(oid)
                         for w, held in sizes_by_writer.items()}
                add_replicated(per_writer[writer], logical, oid, size,
                               writer, policy, plan, sizes)
            else:
                add_private(per_writer[writer], logical, oid, size,
                            writer)
    return logical, per_writer


def add_replicated(out: dict, logical: dict, oid: str, size: int,
                   writer: int, policy: str, plan: dict,
                   sizes: dict) -> None:
    others = {w: n for w, n in sizes.items() if n is not None}
    if any(n != size for n in others.values()):
        raise ValueError(
            f"{oid}: writers disagree on size ({others}) — neither "
            f"replicated nor a known sharding; the policy refuses to "
            f"guess")
    known = logical.setdefault(oid, {"bytes": int(size)})
    if known["bytes"] != size:
        raise ValueError(f"{oid}: logical size conflict "
                         f"{known['bytes']} != {size}")
    if policy == "dedup" and oid in plan:
        entries = [e for e in plan[oid]
                   if e["rank"] == writer and e["role"] == "responsible"]
        for e in entries:
            lo, hi = int(e["lo"]), int(e["hi"])
            out["slices"].append({"id": oid, "src": [lo, hi]})
            out["record"].append({"logical": oid,
                                  "snapshot_range": [lo, hi],
                                  "object_range": [lo, hi],
                                  "authoritative": True})
        return
    out["slices"].append({"id": oid})
    out["record"].append({"logical": oid,
                          "snapshot_range": [0, size],
                          "object_range": [0, size],
                          "authoritative": writer == 0})


def add_private(out: dict, logical: dict, oid: str, size: int,
                writer: int) -> None:
    """Writer-PRIVATE state: an object outside the responsibility
    plan and outside the element-shard pairing accumulates per
    writer, with nothing synchronizing the copies. Same-size copies
    across writers are NOT replicas here, so certifying them under
    the drift rule would refuse legitimate saves; instead each
    writer's copy becomes its own writer-qualified logical object
    (``<id>@<writer>``), whole and authoritative. The rank view
    resolves the bare id to the writer's qualified object and
    restores it under the bare local name."""
    lid = f"{oid}@{writer}"
    known = logical.setdefault(lid, {"bytes": int(size)})
    if known["bytes"] != size:
        raise ValueError(f"{lid}: logical size conflict "
                         f"{known['bytes']} != {size}")
    out["slices"].append({"id": oid, "logical_id": lid,
                          "logical_bytes": int(size)})
    out["record"].append({"logical": lid,
                          "snapshot_range": [0, int(size)],
                          "object_range": [0, int(size)],
                          "authoritative": True})


def add_sharded(out: dict, logical: dict, oid: str, size: int,
                writer: int, world: int, opt_slices: dict,
                root: str) -> None:
    """Slot structure comes entirely from the layout's fields — one
    ``<slot>_slice`` field per state slot, each optionally paired
    with a ``<slot>_tail`` — so any optimizer's slot set (one slot,
    two, more; any dtype) compiles through the same arithmetic. The
    logical object concatenates per-slot spaces in field order; a
    slot's space is the writers' slice extents in writer order plus
    its tail."""
    sh = opt_slices[root]
    layout = opt_state_slice_layout(int(sh["n_slice"]),
                                    int(sh["n_tail"]),
                                    sh["opt_dtype"])
    if size != layout.total_bytes:
        raise ValueError(
            f"{oid}: writer {writer} holds {size} B but its element-"
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
        pieces.append((f, [cursor + writer * f.nbytes,
                           cursor + (writer + 1) * f.nbytes], True))
        cursor += world * f.nbytes
        if tail is not None:
            # every writer updates the world-remainder tail
            # redundantly: the overlapping tail slices are replicas,
            # drift-certified, with writer 0 the restore winner
            pieces.append((tail, [cursor, cursor + tail.nbytes],
                           writer == 0))
            cursor += tail.nbytes
    total_bytes = cursor
    known = logical.setdefault(oid, {"bytes": total_bytes})
    if known["bytes"] != total_bytes:
        raise ValueError(f"{oid}: logical size conflict "
                         f"{known['bytes']} != {total_bytes}")
    for f, dst, authoritative in pieces:
        src = [f.offset_bytes, f.offset_bytes + f.nbytes]
        out["slices"].append({"id": oid, "src": src,
                              "logical_id": oid, "dst": dst,
                              "logical_bytes": total_bytes})
        out["record"].append({"logical": oid,
                              "snapshot_range": list(src),
                              "object_range": list(dst),
                              "authoritative": authoritative})
