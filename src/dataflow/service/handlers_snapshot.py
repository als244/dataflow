"""Snapshot / restore endpoints (S1.3) — installed at boot.

Snapshot is queued->bg: FIFO admission (dispatcher) resolves the
scope, takes READ-LEASES on the id set, freezes metadata, and plans
payload offsets; the PAYLOAD copy runs on the dedicated
snapshot-writer thread (thread #5, design note §IV) reading slab
bytes directly — extents are stable while leased. Queued verbs that
hit a leased id raise LEASED and the dispatcher PARKS them until the
writer releases (no error surfaces to the client).

Dedup: an object whose lineage parent is in the same snapshot, whose
own record is clean, AND whose recorded parent_version still equals
the parent's current version references the parent's payload segment
instead of storing bytes. (Version counters exist because a clean
duplicate does NOT imply byte-equality once the parent trains
onward.) Parents must sort before children for the reference to
bind ("W" < "W@ck" lexicographically — the duplicate naming
convention guarantees it); unsortable exotics just store bytes.

Restore is fully queued (runs on the dispatcher): recreates
residents + object_groups from manifest/payload; returns client_meta
so a driver recovers its own state in the same call.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from .wire import SCHEMA_VERSION, ServiceError

ALIGN = 4096
MANIFEST_SCHEMA = "dataflow-snap/s1"
_CHUNK = 64 << 20


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


class SnapshotWriter(threading.Thread):
    """Thread #5: payload copies + manifest writes, off the dispatcher.
    Owns nothing but its job queue; catalog access is read-only views
    of LEASED records (release/wipe/put on them are parked meanwhile).
    Always releases the job's leases, success or failure."""

    def __init__(self, server):
        super().__init__(name="snapshot-writer", daemon=True)
        self.server = server
        self.jobs: "queue.Queue[dict | None]" = queue.Queue()

    def submit(self, job: dict) -> None:
        self.jobs.put(job)

    def stop(self) -> None:
        self.jobs.put(None)

    def run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            st, store = self.server.state, self.server.store
            snap_id = job["snap_id"]
            try:
                dest = Path(job["dest"])
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / "payload.bin", "wb") as f:
                    for e in job["entries"]:
                        seg = e["payload"]
                        if "ref" in seg:
                            continue
                        rec = store.objects[e["id"]]
                        mv = store.view(rec)
                        f.seek(seg["offset"])
                        f.write(mv)
                        with st.lock:
                            st.snapshots[snap_id]["bytes_done"] += \
                                rec.size_bytes
                manifest = {
                    "schema": MANIFEST_SCHEMA,
                    "service_schema": SCHEMA_VERSION,
                    "snap_id": snap_id,
                    "label": job["label"],
                    "created_t": time.time(),
                    "scope": job["scope"],
                    "client_meta": job["client_meta"],
                    "objects": job["entries"],
                    "object_groups": job["object_groups"],
                }
                tmp = dest / "manifest.json.tmp"
                tmp.write_text(json.dumps(manifest, indent=1))
                tmp.rename(dest / "manifest.json")
                with st.lock:
                    st.snapshots[snap_id]["state"] = "done"
                st.emit("snapshot_done", snap_id=snap_id,
                        path=str(dest), bytes=job["bytes_total"])
            except Exception as e:  # noqa: BLE001 — writer must survive
                with st.lock:
                    st.snapshots[snap_id]["state"] = "error"
                    st.snapshots[snap_id]["error"] = \
                        f"{type(e).__name__}: {e}"
                st.emit("snapshot_error", snap_id=snap_id,
                        error=f"{type(e).__name__}: {e}")
            finally:
                store.release_leases(job["lease_ids"])


def install(server) -> None:
    store = server.store
    st = server.state
    st.snapshots = {}
    writer = SnapshotWriter(server)
    writer.start()
    server.snapshot_writer = writer
    # parked-LEASED retry hook (design: store raises, dispatcher parks)
    store.on_lease_release = server.dispatcher.unpark_all

    # ------------------------------------------------ snapshot (queued->bg)
    def snapshot(call):
        a = call.args
        scope, dest = a["scope"], a["dest"]
        client_meta = a.get("client_meta") or {}
        label = a.get("label")
        ids = sorted(store.resolve_scope(scope))
        if not ids:
            raise ServiceError("BAD_REQUEST", f"scope '{scope}' is empty")

        entries, offset, payload_ids = [], 0, set()
        with store.catalog_lock:
            for oid in ids:
                rec = store.objects[oid]
                lin = rec.lineage
                parent = lin.parent
                seg = None
                if (not lin.dirty and parent in payload_ids
                        and lin.parent_version is not None
                        and store.objects[parent].version
                        == lin.parent_version
                        and store.objects[parent].size_bytes
                        == rec.size_bytes):
                    seg = {"ref": parent}
                else:
                    seg = {"offset": offset, "size": rec.size_bytes}
                    offset = _align(offset + rec.size_bytes)
                    payload_ids.add(oid)
                entries.append({
                    "id": oid, "size_bytes": rec.size_bytes,
                    "meta": rec.meta, "protected": rec.protected,
                    "version": rec.version,
                    "lineage": {"parent": lin.parent, "dirty": lin.dirty,
                                "parent_version": lin.parent_version},
                    "payload": seg,
                })
            idset = set(ids)
            ogroups = [
                {"ogid": g.ogid, "members": list(g.members),
                 "sub_groups": list(g.sub_groups)}
                for g in store.object_groups.values()
                if set(store.resolve_object_group(g.ogid)) <= idset
            ]
        snap_id = st.next_id("snap")
        # leases LAST + exception-safe: a failed admission must not
        # leak leases (leaked leases park every later writer forever
        # — found when a KeyError after acquire wedged the suite)
        store.acquire_leases(ids)
        with st.lock:
            st.snapshots[snap_id] = {
                "snap_id": snap_id, "state": "writing",
                "bytes_done": 0, "bytes_total": offset,
                "n_objects": len(ids), "n_deduped": len(ids) - len(
                    payload_ids),
                "path": str(dest), "label": label,
                "created_t": time.time(), "error": None,
            }
        try:
            st.emit("snapshot_started", snap_id=snap_id, path=str(dest),
                    n_objects=len(ids), bytes=offset)
        except Exception:
            store.release_leases(ids)
            raise
        writer.submit({
            "snap_id": snap_id, "dest": dest, "scope": scope,
            "label": label, "client_meta": client_meta,
            "entries": entries, "object_groups": ogroups,
            "lease_ids": ids, "bytes_total": offset,
        })
        return {"ok": True, "snap_id": snap_id, "bytes_total": offset,
                "n_objects": len(ids),
                "n_deduped": len(ids) - len(payload_ids)}

    # ------------------------------------------------ status (fast)
    def snapshot_status(conn, args):
        with st.lock:
            rec = st.snapshots.get(args["snap_id"])
            if rec is None:
                raise ServiceError("UNKNOWN_SNAPSHOT", args["snap_id"])
            return dict(rec)

    # ------------------------------------------------ restore (queued)
    def restore_snapshot(call):
        a = call.args
        path = Path(a["path"])
        placement = a.get("placement", "initial")
        duplicates = a.get("duplicates", "recreate")
        overwrite = bool(a.get("overwrite", False))
        if placement not in ("initial", "backing_only"):
            raise ServiceError("BAD_REQUEST", f"placement '{placement}'")
        if duplicates not in ("recreate", "roots_only"):
            raise ServiceError("BAD_REQUEST", f"duplicates '{duplicates}'")
        mf_path = path / "manifest.json"
        if not mf_path.is_file():
            raise ServiceError("IO_ERROR", f"no manifest at {path}")
        manifest = json.loads(mf_path.read_text())
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise ServiceError("VERSION_SKEW",
                               f"manifest schema {manifest.get('schema')}")
        by_id = {e["id"]: e for e in manifest["objects"]}

        # lease pre-pass: LEASED must precede any mutation (park rule)
        with store.catalog_lock:
            for e in manifest["objects"]:
                rec = store.objects.get(e["id"])
                if rec is not None and rec.lease_refs:
                    raise ServiceError("LEASED", e["id"])

        restored, skipped = [], []
        with open(path / "payload.bin", "rb") as f:
            for e in manifest["objects"]:
                oid = e["id"]
                if (duplicates == "roots_only"
                        and e["lineage"]["parent"] in by_id):
                    skipped.append(oid)
                    continue
                exists = store.objects.get(oid)
                if exists is not None:
                    if not overwrite:
                        raise ServiceError(
                            "COLLISION",
                            f"{oid} resident; pass overwrite=True")
                    if exists.size_bytes != e["size_bytes"]:
                        raise ServiceError(
                            "BINDING_MISMATCH",
                            f"{oid}: resident {exists.size_bytes} B != "
                            f"snapshot {e['size_bytes']} B")
                seg = e["payload"]
                src = by_id[seg["ref"]]["payload"] if "ref" in seg else seg
                rec = store.put(oid, None, size_bytes=e["size_bytes"],
                                meta=e["meta"], writer="restore")
                mv = store.view(rec)
                f.seek(src["offset"])
                off = 0
                while off < e["size_bytes"]:
                    n = f.readinto(mv[off:off + _CHUNK])
                    if not n:
                        raise ServiceError(
                            "IO_ERROR", f"short payload read for {oid}")
                    off += n
                lin = e["lineage"]
                rec.protected = bool(e.get("protected", False))
                rec.lineage.parent = lin["parent"]
                rec.lineage.dirty = lin["dirty"]
                rec.lineage.parent_version = lin["parent_version"]
                rec.version = e.get("version", 0)
                rec.last_write = {"by": f"restore:{manifest['snap_id']}",
                                  "t": time.time()}
                restored.append(oid)

        groups_recreated = []
        for g in manifest.get("object_groups", []):
            if g["ogid"] in store.object_groups:
                continue
            if not all(m in store.objects for m in g["members"]):
                continue                      # roots_only may drop members
            store.create_object_group(g["ogid"], g["members"], None,
                                      [s for s in g["sub_groups"]
                                       if s in store.object_groups])
            groups_recreated.append(g["ogid"])
        st.emit("restore_done", path=str(path), n_restored=len(restored))
        return {"ok": True, "restored": restored, "skipped": skipped,
                "object_groups_recreated": groups_recreated,
                "client_meta": manifest.get("client_meta", {})}

    server.dispatcher.handlers["snapshot"] = snapshot
    server.dispatcher.handlers["restore_snapshot"] = restore_snapshot
    server.fast_handlers["snapshot_status"] = snapshot_status
