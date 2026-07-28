"""The checkpoint composer: many sources' snapshots become one
complete, certified record — or nothing.

save_checkpoint exists to ENCODE the completeness invariant: it fans
out one snapshot per source, waits for all of them, collects the
per-slice hashes each daemon computed while streaming, runs the
cross-source drift check, and only then writes checkpoint_record.json
— atomically, LAST. Any failure before that point leaves snapshot
dirs on disk for forensics and NO record: the step directory is
incomplete by contract and can never be selected for resume.

The drift check is the free replication certificate: identical-span
slices from different sources (the simple policy's overlapping
replicated copies) must carry EQUAL hashes; disagreement refuses the
checkpoint, naming the object and both sources, because
certified-replicated state that diverged is a bug upstream of any
checkpoint. Certified equal, the copies are interchangeable — any
reader may take any replica.

Hashes travel in snapshot_status, so sources on other filesystems
need ship no bytes for the check. Sources never see each other; the
caller holding all the clients — one process, any number of daemons —
is the only joiner.
"""
from __future__ import annotations

from .record import CheckpointError, write_record


def save_checkpoint(sources: dict, dest, *, step, seed,
                    logical_objects, scheme=None, client_payload=None,
                    summary=None, launch=None, field_schemas=None,
                    timeout: float = 600.0) -> dict:
    """``sources``: {key: {"client": EngineClient, "path": subdir
    name, "slices": engine slice list, "record": record slice
    entries without hashes, "objects": the source's resident sizes
    {id: bytes} (recorded on its snapshot entry so rank-view restores
    recreate exact local geometry), "client_meta": optional dict}} —
    the shape the source-policy compiler emits, one entry per source.
    Returns the record dict after landing it last."""
    from pathlib import Path

    dest = Path(dest)
    receipts = {}
    for key in sorted(sources):
        w = sources[key]
        receipts[key] = w["client"].snapshot(
            str(dest / w["path"]), slices=w["slices"],
            client_meta=w.get("client_meta") or {})
    statuses = {}
    for key in sorted(sources):
        s = sources[key]["client"].wait_snapshot(
            receipts[key]["snap_id"], timeout=timeout)
        if s["state"] != "done":
            raise CheckpointError(
                f"source {key} snapshot failed: {s.get('error')} — "
                f"no record written")
        statuses[key] = s

    snapshots = []
    slices: dict = {}
    order = {}
    for key in sorted(sources):
        order[key] = len(snapshots)
        snapshots.append({"path": sources[key]["path"],
                          "source": str(key),
                          "objects": {oid: int(n) for oid, n in
                                      (sources[key].get("objects")
                                       or {}).items()}})
    for key in sorted(sources):
        hashes = hash_by_span(statuses[key])
        for entry in sources[key]["record"]:
            span = (entry["logical"], tuple(entry["object_range"]))
            if span not in hashes:
                raise CheckpointError(
                    f"source {key}: snapshot status carries no hash "
                    f"for {entry['logical']} {entry['object_range']}")
            slices.setdefault(entry["logical"], []).append({
                "snapshot": order[key],
                "snapshot_range": list(entry["snapshot_range"]),
                "object_range": list(entry["object_range"]),
                "hash": hashes[span],
            })

    check_replication_drift(slices, snapshots)
    engine_spec = {}
    for key in sorted(sources):
        client = sources[key]["client"]
        backing = client.query_backing()
        boot = client.engine_status().get("boot_config") or {}
        engine_spec[str(key)] = {
            "backing_gib": round(
                backing.get("capacity_bytes", 0) / 1024 ** 3, 2),
            "device": boot.get("device"),
            "kernel_set": boot.get("kernel_set"),
            "fake": bool(boot.get("fake")),
        }
    return write_record(
        dest, step=step, seed=seed, logical_objects=logical_objects,
        slices=slices, snapshots=snapshots, engine_spec=engine_spec,
        scheme=scheme, client_payload=client_payload, summary=summary,
        launch=launch, field_schemas=field_schemas)


def hash_by_span(status: dict) -> dict:
    """{(logical_id, object_range): hash} from one snapshot status —
    the per-slice hashes the source engine computed while streaming."""
    out = {}
    for s in status.get("slices", []):
        out[(s["logical_id"], tuple(s["dst"]))] = s["hash"]
    return out


def check_replication_drift(slices: dict, snapshots: list) -> None:
    """Identical-span slices are REPLICAS by construction — hash
    equality certifies them interchangeable, so any two entries
    covering the same span of the same logical object must
    hash-equal. Refuses naming the object and both sources.
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
                    a = snapshots[first["snapshot"]]["source"]
                    b = snapshots[other["snapshot"]]["source"]
                    raise CheckpointError(
                        f"replication drift at save: {lid} "
                        f"{list(span)} from sources {a} and {b} "
                        f"carry different hashes — no record written")
