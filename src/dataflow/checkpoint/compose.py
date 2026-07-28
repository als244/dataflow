"""The checkpoint composer: many writers' snapshots become one
complete, certified record — or nothing.

save_checkpoint exists to ENCODE the completeness invariant: it fans
out one snapshot per writer, waits for all of them, collects the
per-slice hashes each daemon computed while streaming, runs the
cross-writer drift check, and only then writes checkpoint_record.json
— atomically, LAST. Any failure before that point leaves snapshot
dirs on disk for forensics and NO record: the step directory is
incomplete by contract and can never be selected for resume.

The drift check is the free replication certificate: identical-span
authoritative slices from different writers (the simple policy's
overlapping replicated copies) must carry EQUAL hashes; disagreement
refuses the checkpoint, naming the object and both writers, because
certified-replicated state that diverged is a bug upstream of any
checkpoint.

Hashes travel in snapshot_status, so writers on other filesystems
need ship no bytes for the check. Writers never see each other; the
caller holding all the clients — one process, any number of daemons —
is the only joiner.
"""
from __future__ import annotations

from .record import CheckpointError, write_record


def save_checkpoint(writers: dict, dest, *, step, seed,
                    logical_objects, scheme=None, client_payload=None,
                    summary=None, launch=None, field_schemas=None,
                    timeout: float = 600.0) -> dict:
    """``writers``: {key: {"client": EngineClient, "path": subdir
    name, "slices": engine slice list, "record": record slice
    entries without hashes, "objects": the writer's resident sizes
    {id: bytes} (recorded on its snapshot entry so rank-view restores
    recreate exact local geometry), "client_meta": optional dict}} —
    the shape the source-policy compiler emits, one entry per writer.
    Returns the record dict after landing it last."""
    from pathlib import Path

    dest = Path(dest)
    receipts = {}
    for key in sorted(writers):
        w = writers[key]
        receipts[key] = w["client"].snapshot(
            str(dest / w["path"]), slices=w["slices"],
            client_meta=w.get("client_meta") or {})
    statuses = {}
    for key in sorted(writers):
        s = writers[key]["client"].wait_snapshot(
            receipts[key]["snap_id"], timeout=timeout)
        if s["state"] != "done":
            raise CheckpointError(
                f"writer {key} snapshot failed: {s.get('error')} — "
                f"no record written")
        statuses[key] = s

    snapshots = []
    slices: dict = {}
    order = {}
    for key in sorted(writers):
        order[key] = len(snapshots)
        snapshots.append({"path": writers[key]["path"],
                          "writer": str(key),
                          "objects": {oid: int(n) for oid, n in
                                      (writers[key].get("objects")
                                       or {}).items()}})
    for key in sorted(writers):
        hashes = hash_by_span(statuses[key])
        for entry in writers[key]["record"]:
            span = (entry["logical"], tuple(entry["object_range"]))
            if span not in hashes:
                raise CheckpointError(
                    f"writer {key}: snapshot status carries no hash "
                    f"for {entry['logical']} {entry['object_range']}")
            slices.setdefault(entry["logical"], []).append({
                "snapshot": order[key],
                "snapshot_range": list(entry["snapshot_range"]),
                "object_range": list(entry["object_range"]),
                "hash": hashes[span],
                "authoritative": bool(entry.get("authoritative")),
            })

    check_replication_drift(slices, snapshots)
    engine_spec = {}
    for key in sorted(writers):
        backing = writers[key]["client"].query_backing()
        engine_spec[str(key)] = {
            "backing_gib": round(
                backing.get("capacity_bytes", 0) / 1024 ** 3, 2)}
    return write_record(
        dest, step=step, seed=seed, logical_objects=logical_objects,
        slices=slices, snapshots=snapshots, engine_spec=engine_spec,
        scheme=scheme, client_payload=client_payload, summary=summary,
        launch=launch, field_schemas=field_schemas)


def hash_by_span(status: dict) -> dict:
    """{(logical_id, object_range): hash} from one snapshot status —
    the per-slice hashes the writer computed while streaming."""
    out = {}
    for s in status.get("slices", []):
        out[(s["logical_id"], tuple(s["dst"]))] = s["hash"]
    return out


def check_replication_drift(slices: dict, snapshots: list) -> None:
    """Identical-span slices are REPLICAS by construction — the
    authoritative flag picks the restore winner, hash equality
    certifies them interchangeable — so any two entries covering the
    same span of the same logical object must hash-equal, whichever
    carries the flag. Refuses naming the object and both writers.
    (validate_record re-checks this — running it here first makes the
    refusal happen BEFORE any record could exist.)"""
    for lid, entries in slices.items():
        by_span: dict = {}
        for e in entries:
            by_span.setdefault(tuple(e["object_range"]), []).append(e)
        for span, twins in by_span.items():
            first = twins[0]
            for other in twins[1:]:
                if other["hash"] != first["hash"]:
                    a = snapshots[first["snapshot"]]["writer"]
                    b = snapshots[other["snapshot"]]["writer"]
                    raise CheckpointError(
                        f"replication drift at save: {lid} "
                        f"{list(span)} from writers {a} and {b} "
                        f"carry different hashes — no record written")
