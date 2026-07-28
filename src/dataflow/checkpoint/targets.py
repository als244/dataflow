"""Targets resolution: a record plus a target set becomes
per-snapshot fetch plans in restore's remap wire shape — every
requested byte sourced exactly once, total or loud.

Two precedence modes. LOGICAL-view targets ("all" or bare id lists)
assemble logical-sized objects and prefer AUTHORITATIVE slices where
coverage overlaps. KEYED targets ({writer_key: ids}) are the rank
view: only that writer's own snapshots are used — the
self-sufficiency contract, no cross-writer shipping — and bytes land
in a LOCAL object at local coordinates sized by the writer's own
ranges.

Each plan step is one restore call: {"snapshot": index,
"path": snapshot path, "remap": {logical_id: [window, ...]}} with
windows in the engine's remap shape ({"logical": [c, d),
"id": local_id, "local": [x, y), "bytes": local_total}).
"""
from __future__ import annotations

from .record import CheckpointError, coverage_gaps, validate_record


def resolve_targets(record: dict, targets, *, validate=True) -> list:
    if validate:
        validate_record(record)
    if targets == "all":
        return logical_plan(record, sorted(record["logical_objects"]))
    if isinstance(targets, dict):
        steps = []
        for key, ids in targets.items():
            steps += keyed_plan(record, key, list(ids))
        return merge_steps(record, steps)
    return logical_plan(record, list(targets))


def logical_plan(record: dict, ids: list) -> list:
    logical = record["logical_objects"]
    pieces = []
    for lid in ids:
        if lid not in logical:
            raise CheckpointError(f"unknown target {lid}")
        total = int(logical[lid]["bytes"])
        entries = record["slices"].get(lid, [])
        gaps = coverage_gaps(total, entries)
        if gaps:
            named = ", ".join(f"[{a}, {b})" for a, b in gaps)
            raise CheckpointError(
                f"{lid}: no slice covers {named}")
        for lo, hi, entry in choose_sources(entries):
            pieces.append((entry["snapshot"], lid,
                           {"logical": [lo, hi], "id": lid,
                            "local": [lo, hi], "bytes": total}))
    return merge_steps(record, pieces)


def choose_sources(entries: list) -> list:
    """One source per elementary interval: the authoritative slice
    when the interval is multiply covered (validation guarantees a
    winner exists), the sole coverer otherwise."""
    bounds = sorted({x for e in entries for x in e["object_range"]})
    chosen = []
    for lo, hi in zip(bounds, bounds[1:]):
        covering = [e for e in entries
                    if e["object_range"][0] <= lo
                    and hi <= e["object_range"][1]]
        if not covering:
            continue
        winner = covering[0]
        for e in covering:
            if e.get("authoritative"):
                winner = e
                break
        chosen.append((lo, hi, winner))
    return chosen


def keyed_plan(record: dict, key, ids: list) -> list:
    logical = record["logical_objects"]
    own = [i for i, s in enumerate(record["snapshots"])
           if s.get("writer") == key]
    if not own:
        raise CheckpointError(f"no snapshot belongs to writer {key}")
    pieces = []
    for lid in ids:
        if lid not in logical:
            raise CheckpointError(f"unknown target {lid}")
        entries = [e for e in record["slices"].get(lid, [])
                   if e["snapshot"] in own]
        if not entries:
            raise CheckpointError(
                f"writer {key} holds no bytes of {lid} — the rank "
                f"view restores only from its own snapshots")
        entries.sort(key=entry_span)
        local_total = sum(e["object_range"][1] - e["object_range"][0]
                          for e in entries)
        cursor = 0
        for e in entries:
            lo, hi = e["object_range"]
            pieces.append((e["snapshot"], lid,
                           {"logical": [lo, hi], "id": lid,
                            "local": [cursor, cursor + (hi - lo)],
                            "bytes": local_total}))
            cursor += hi - lo
    return pieces


def entry_span(entry: dict) -> tuple:
    return tuple(entry["object_range"])


def merge_steps(record: dict, pieces: list) -> list:
    """Group (snapshot, logical, window) pieces into one plan step
    per snapshot, coalescing windows that abut in both logical and
    local space."""
    by_snapshot: dict = {}
    for snap, lid, window in pieces:
        by_snapshot.setdefault(snap, {}).setdefault(lid, []).append(
            window)
    steps = []
    for snap in sorted(by_snapshot):
        remap = {}
        for lid, windows in by_snapshot[snap].items():
            windows.sort(key=window_span)
            remap[lid] = coalesce_windows(windows)
        steps.append({"snapshot": snap,
                      "path": record["snapshots"][snap]["path"],
                      "remap": remap})
    return steps


def window_span(window: dict) -> tuple:
    return tuple(window["logical"])


def coalesce_windows(windows: list) -> list:
    out = [dict(windows[0])]
    for w in windows[1:]:
        last = out[-1]
        if (w["logical"][0] == last["logical"][1]
                and w["local"][0] == last["local"][1]
                and w["bytes"] == last["bytes"]
                and w["id"] == last["id"]):
            last["logical"] = [last["logical"][0], w["logical"][1]]
            last["local"] = [last["local"][0], w["local"][1]]
        else:
            out.append(dict(w))
    return out
