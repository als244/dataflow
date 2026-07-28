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

zero1-sharded optimizer objects are shard-sized residents packing
their slot regions locally ([m_r | v_r]); their logical object is the
world-spanning [m_all | v_all], so each writer contributes TWO slices
per object — its m range and its v range — at element-derived
offsets. Objects that are neither replicated-by-size nor zero1-paired
are refused loudly rather than guessed at.
"""
from __future__ import annotations

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "int32": 4, "int64": 8}


def opt_root(oid: str) -> str | None:
    """The parameter root an optimizer-state object pairs with
    ("O_3" -> "W_3", "O_embed" -> "W_embed"), or None."""
    if oid.startswith("O_"):
        return "W_" + oid[2:]
    return None


def zero1_span(opt_slices: dict, root: str, rank: int, world: int):
    """(elements_this_rank, element_offset, total_elements, esize)
    for a zero1-sharded root, in optimizer elements."""
    sh = opt_slices[root]
    esize = DTYPE_BYTES[sh["opt_dtype"]]
    n_slice, n_tail = int(sh["n_slice"]), int(sh["n_tail"])
    elems = n_slice + (n_tail if rank == world - 1 else 0)
    return elems, rank * n_slice, n_slice * world + n_tail, esize


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
    engine wire lists) and "record" (record slice entries without
    hashes — the composer fills those from snapshot statuses).
    """
    if policy not in ("simple", "dedup"):
        raise ValueError(f"unknown source policy {policy!r}")
    logical: dict = {}
    per_writer = {k: {"slices": [], "record": []}
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
    elems, off, total, esize = zero1_span(opt_slices, root, writer,
                                          world)
    if size != 2 * elems * esize:
        raise ValueError(
            f"{oid}: writer {writer} holds {size} B but its zero1 "
            f"shard is {2 * elems * esize} B (m+v of {elems} "
            f"elements)")
    total_bytes = 2 * total * esize
    known = logical.setdefault(oid, {"bytes": total_bytes})
    if known["bytes"] != total_bytes:
        raise ValueError(f"{oid}: logical size conflict "
                         f"{known['bytes']} != {total_bytes}")
    half_local = elems * esize
    half_logical = total * esize
    pieces = (
        ("m", [0, half_local],
         [off * esize, off * esize + half_local]),
        ("v", [half_local, 2 * half_local],
         [half_logical + off * esize,
          half_logical + off * esize + half_local]),
    )
    for _slot, src, dst in pieces:
        out["slices"].append({"id": oid, "src": src,
                              "logical_id": oid, "dst": dst,
                              "logical_bytes": total_bytes})
        out["record"].append({"logical": oid,
                              "snapshot_range": list(src),
                              "object_range": list(dst),
                              "authoritative": True})
