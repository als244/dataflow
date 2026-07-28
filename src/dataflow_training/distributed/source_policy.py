"""Source policies: WHO saves WHICH bytes, compiled from the
responsibility map into engine slice lists and record slice entries.

``simple`` (the default): every writer saves everything it holds,
whole — its full replicated objects and its own shard-sized objects —
so each writer's snapshot restores that rank with zero cross-writer
shipping. Replicated objects overlap across writers with writer 0
authoritative; the equal-hash rule on those overlaps certifies
replication on every save. ``dedup`` saves each replicated object as
disjoint responsibility slices instead — one copy total across the
checkpoint, each slice authoritative — trading self-sufficiency and
redundancy for disk.

zero1-sharded optimizer objects are shard-sized residents laid out by
``opt_state_slice_layout`` — aligned ``[m_slice | v_slice]`` fields
plus, when the element count doesn't divide the world, ``m_tail`` /
``v_tail`` fields every rank updates redundantly. Their logical
object is the tight world-spanning ``[m_all | v_all]`` byte space:
each writer's slice fields map to element-derived offsets in it, and
the replicated tail fields overlap across writers (writer 0
authoritative) so the drift certificate covers the redundant tail
updates too. Alignment padding between fields is layout, not state —
it maps to no logical byte. Objects that are neither
replicated-by-size nor zero1-paired are refused loudly rather than
guessed at.

Each writer also emits its resident-object inventory ({id: bytes});
the composer records it per snapshot so the rank view can recreate
objects at their exact local sizes, padding included.
"""
from __future__ import annotations

from dataflow.core import DTYPE_BITS

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
    ``opt_slices``: zero1 shard dict {root: {n_slice, n_tail,
    opt_dtype}} when the optimizer is sharded, else None.

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
                add_zero1_shard(per_writer[writer], logical, oid, size,
                                writer, world, opt_slices, root)
                continue
            sizes = {w: held.get(oid)
                     for w, held in sizes_by_writer.items()}
            add_replicated(per_writer[writer], logical, oid, size,
                           writer, policy, plan, sizes)
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


def add_zero1_shard(out: dict, logical: dict, oid: str, size: int,
                    writer: int, world: int, opt_slices: dict,
                    root: str) -> None:
    sh = opt_slices[root]
    n_slice, n_tail = int(sh["n_slice"]), int(sh["n_tail"])
    esize = DTYPE_BITS[sh["opt_dtype"]] // 8
    layout = opt_state_slice_layout(n_slice, n_tail, sh["opt_dtype"])
    if size != layout.total_bytes:
        raise ValueError(
            f"{oid}: writer {writer} holds {size} B but its zero1 "
            f"slice layout is {layout.total_bytes} B "
            f"({[f.name for f in layout.fields]})")
    total = n_slice * world + n_tail
    total_bytes = 2 * total * esize
    known = logical.setdefault(oid, {"bytes": total_bytes})
    if known["bytes"] != total_bytes:
        raise ValueError(f"{oid}: logical size conflict "
                         f"{known['bytes']} != {total_bytes}")
    half_logical = total * esize
    even = world * n_slice * esize
    pieces = [
        ("m_slice", [writer * n_slice * esize,
                     (writer + 1) * n_slice * esize], True),
        ("v_slice", [half_logical + writer * n_slice * esize,
                     half_logical + (writer + 1) * n_slice * esize],
         True),
    ]
    if n_tail:
        # every rank updates the world-remainder tail redundantly:
        # the overlapping tail slices are replicas, drift-certified,
        # with writer 0 the restore winner
        pieces += [
            ("m_tail", [even, half_logical], writer == 0),
            ("v_tail", [half_logical + even, total_bytes],
             writer == 0),
        ]
    for field_name, dst, authoritative in pieces:
        f = layout.field(field_name)
        src = [f.offset_bytes, f.offset_bytes + f.nbytes]
        out["slices"].append({"id": oid, "src": src,
                              "logical_id": oid, "dst": dst,
                              "logical_bytes": total_bytes})
        out["record"].append({"logical": oid,
                              "snapshot_range": list(src),
                              "object_range": list(dst),
                              "authoritative": authoritative})
