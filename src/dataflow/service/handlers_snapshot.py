"""Snapshot / restore endpoints — installed at boot.

A snapshot saves SLICES: byte ranges of stored objects mapped into
logical objects. Each slice records where its bytes came from (``src``,
a range in the stored object) and where they belong (``logical_id`` +
``dst``, a range in the logical object's byte space, whose total size
is ``logical_bytes``); a whole-object save is the identity mapping and
needs no explicit fields. The snapshot dir is self-describing:
``payload.bin`` + ``snapshot.json`` (schema ``dataflow-snapshot/v1``,
written LAST as the completeness marker) with a streaming blake2b-16
hash per slice.

Snapshot is queued->bg: FIFO admission (dispatcher) validates the
slice list, takes READ-LEASES on the stored ids, freezes metadata, and
plans payload offsets; the payload copy + hashing run on the dedicated
payload thread reading slab bytes directly — extents are
stable while leased. Queued verbs that hit a leased id raise LEASED
and the dispatcher PARKS them until the payload thread releases.

Restore is queued->bg like snapshot, and THREE-PASS: admission (on
the dispatcher) resolves every placement, validates it (leases,
sizes, collisions), creates absent targets and takes leases on all
targets; the payload thread verifies every payload hash (default ON)
and only then places bytes; a failure rolls back the targets
admission created — under the still-held leases, so nothing can have
touched them — and any refusal leaves the store as it was. Default placement follows the
slice mapping (logical-named targets; an identity slice recreates its
stored object exactly, metadata included). An optional remap plan
EXTRACTS logical ranges into local objects instead: each slice's dst
is intersected with the plan's windows, split where a window splits
it, and skipped where no window covers it.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path

from .wire import SCHEMA_VERSION, ServiceError

ALIGN = 4096
SNAPSHOT_SCHEMA = "dataflow-snapshot/v1"
CHUNK_BYTES = 64 << 20


def align_up(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def slice_hash():
    return hashlib.blake2b(digest_size=16)


def identity_slice(entry: dict) -> bool:
    """A slice that reproduces its stored object exactly: whole-object
    src, same-named logical of the same size, unshifted dst. Identity
    slices restore the stored object's metadata; mapped slices place
    bytes only."""
    return (entry["logical_id"] == entry["id"]
            and entry["src"] == [0, entry["size_bytes"]]
            and entry["dst"] == entry["src"]
            and entry["logical_bytes"] == entry["size_bytes"])


def resolve_slices(store, specs: list) -> list:
    """Validate a slice list and normalize it into snapshot entries
    with every mapping field explicit. Refuses with BAD_REQUEST naming
    the offending slice; caller holds the catalog lock."""
    entries = []
    logical_sizes: dict = {}
    for s in specs:
        oid = s.get("id")
        rec = store.objects.get(oid)
        if rec is None:
            raise ServiceError("BAD_REQUEST", f"slice id absent: {oid}")
        size = rec.size_bytes
        src = [int(x) for x in (s.get("src") or (0, size))]
        if not (0 <= src[0] < src[1] <= size):
            raise ServiceError("BAD_REQUEST",
                               f"src {src} outside {oid} ({size} B)")
        logical_id = s.get("logical_id") or oid
        dst = [int(x) for x in (s.get("dst") or src)]
        if dst[1] - dst[0] != src[1] - src[0]:
            raise ServiceError(
                "BAD_REQUEST",
                f"{oid}: dst {dst} length != src {src} length")
        logical_bytes = int(s.get("logical_bytes") or size)
        if not (0 <= dst[0] < dst[1] <= logical_bytes):
            raise ServiceError(
                "BAD_REQUEST",
                f"{oid}: dst {dst} outside logical {logical_id} "
                f"({logical_bytes} B)")
        seen = logical_sizes.get(logical_id)
        if seen is not None and seen != logical_bytes:
            raise ServiceError(
                "BAD_REQUEST",
                f"logical {logical_id}: conflicting logical_bytes "
                f"{seen} != {logical_bytes}")
        logical_sizes[logical_id] = logical_bytes
        entries.append({
            "id": oid, "size_bytes": size, "meta": rec.meta,
            "protected": rec.protected,
            "src": src, "logical_id": logical_id, "dst": dst,
            "logical_bytes": logical_bytes,
        })
    return entries


def validate_remap(remap: dict) -> None:
    """A remap plan maps logical ranges into local objects:
    {logical_id: [{"logical": [c, d], "id": local_id,
                   "local": [x, y], "bytes": local_total}, ...]}.
    Window lengths must match and windows of one logical must not
    overlap; local targets sharing an id must agree on their size."""
    local_sizes: dict = {}
    for logical_id, windows in remap.items():
        spans = []
        for w in windows:
            llo, lhi = (int(w["logical"][0]), int(w["logical"][1]))
            xlo, xhi = (int(w["local"][0]), int(w["local"][1]))
            total = int(w["bytes"])
            if llo >= lhi or lhi - llo != xhi - xlo:
                raise ServiceError(
                    "BAD_REQUEST",
                    f"remap {logical_id}: local {w['local']} length "
                    f"!= logical {w['logical']}")
            if not (0 <= xlo < xhi <= total):
                raise ServiceError(
                    "BAD_REQUEST",
                    f"remap {logical_id}: local {w['local']} outside "
                    f"{w['id']} ({total} B)")
            seen = local_sizes.get(w["id"])
            if seen is not None and seen != total:
                raise ServiceError(
                    "BAD_REQUEST",
                    f"remap: {w['id']} sized {seen} B and {total} B")
            local_sizes[w["id"]] = total
            spans.append((llo, lhi))
        spans.sort()
        for (alo, ahi), (blo, bhi) in zip(spans, spans[1:]):
            if blo < ahi:
                raise ServiceError(
                    "BAD_REQUEST",
                    f"remap {logical_id}: overlapping windows")


def resolve_placements(entries: list, remap) -> tuple:
    """Turn slice entries into concrete byte placements. Without a
    remap plan a slice lands at dst in its logical-named object; with
    one, the dst range is intersected with the plan's windows (split
    where a window splits it, skipped where none covers it). Returns
    (placements, required) where required maps every target to its
    full size and creation metadata."""
    placements, required = [], {}
    for e in entries:
        seg = e["payload"]
        dlo, dhi = e["dst"]
        pieces = []
        if remap is None:
            meta = e["meta"] if identity_slice(e) else None
            pieces.append((e["logical_id"], e["logical_bytes"], meta,
                           dlo, dhi, seg["offset"]))
        else:
            for w in remap.get(e["logical_id"], ()):
                ilo = max(dlo, int(w["logical"][0]))
                ihi = min(dhi, int(w["logical"][1]))
                if ilo >= ihi:
                    continue
                tlo = int(w["local"][0]) + (ilo - int(w["logical"][0]))
                pieces.append((w["id"], int(w["bytes"]), None,
                               tlo, tlo + (ihi - ilo),
                               seg["offset"] + (ilo - dlo)))
        for target, total, meta, tlo, thi, file_off in pieces:
            want = required.get(target)
            if want is not None and want["size"] != total:
                raise ServiceError(
                    "BAD_REQUEST",
                    f"{target}: placements disagree on size "
                    f"({want['size']} B != {total} B)")
            if want is None:
                required[target] = {"size": total, "meta": meta}
            placements.append({"target": target, "lo": tlo, "hi": thi,
                               "file_offset": file_off,
                               "entry": e})
    return placements, required


def verify_payload(path: Path, entries: list) -> None:
    """Re-hash every stored payload segment and compare against the
    recorded slice hashes; refuses with VERIFY_FAILED before any
    placement happens."""
    with open(path / "payload.bin", "rb") as f:
        for e in entries:
            seg = e["payload"]
            h = slice_hash()
            f.seek(seg["offset"])
            left = seg["size"]
            while left:
                chunk = f.read(min(CHUNK_BYTES, left))
                if not chunk:
                    raise ServiceError("IO_ERROR",
                                       f"short payload read for {e['id']}")
                h.update(chunk)
                left -= len(chunk)
            if h.hexdigest() != e.get("hash"):
                raise ServiceError(
                    "VERIFY_FAILED",
                    f"{e['id']}: payload hash mismatch (corrupt or "
                    f"tampered slice); pass verify=False to restore "
                    f"the on-disk bytes anyway")


def snapshot_job(server, job: dict) -> None:
    """One snapshot's payload copy + snapshot.json write, on the
    payload thread. Catalog access is read-only views of LEASED
    records (release/wipe/put on them are parked meanwhile). Always
    releases the job's leases, success or failure."""
    st, store = server.state, server.store
    snap_id = job["snap_id"]
    try:
        dest = Path(job["dest"])
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / "payload.bin", "wb") as f:
            for e in job["entries"]:
                seg = e["payload"]
                rec = store.objects[e["id"]]
                mv = store.view(rec)[e["src"][0]:e["src"][1]]
                h = slice_hash()
                f.seek(seg["offset"])
                n = seg["size"]
                for off in range(0, n, CHUNK_BYTES):
                    chunk = mv[off:min(off + CHUNK_BYTES, n)]
                    f.write(chunk)
                    h.update(chunk)
                    with st.lock:
                        st.snapshots[snap_id]["bytes_done"] += len(chunk)
                e["hash"] = h.hexdigest()
        doc = {
            "schema": SNAPSHOT_SCHEMA,
            "service_schema": SCHEMA_VERSION,
            "snap_id": snap_id,
            "created_t": time.time(),
            "client_meta": job["client_meta"],
            "slices": job["entries"],
        }
        tmp = dest / "snapshot.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.rename(dest / "snapshot.json")
        with st.lock:
            st.snapshots[snap_id]["state"] = "done"
            st.snapshots[snap_id]["slices"] = [
                {"id": e["id"], "logical_id": e["logical_id"],
                 "dst": e["dst"], "hash": e["hash"]}
                for e in job["entries"]]
        st.emit("snapshot_done", snap_id=snap_id,
                path=str(dest), bytes=job["bytes_total"])
    except Exception as e:  # noqa: BLE001 — the payload thread must survive
        with st.lock:
            st.snapshots[snap_id]["state"] = "error"
            st.snapshots[snap_id]["error"] = f"{type(e).__name__}: {e}"
        st.emit("snapshot_error", snap_id=snap_id,
                error=f"{type(e).__name__}: {e}")
    finally:
        with st.lock:
            if snap_id in st.snapshots_in_flight:
                st.snapshots_in_flight.remove(snap_id)
        store.release_leases(job["lease_ids"])


def restore_job(server, job: dict) -> None:
    """One restore's verify + placement, on the payload thread. The
    targets are LEASED (admission took them) so their extents are
    stable and no dispatcher verb can mutate them — which also makes
    the failure rollback of admission-created targets race-free.
    Always releases the leases, success or failure."""
    st, store = server.state, server.store
    rid = job["restore_id"]
    error = None
    try:
        path = Path(job["path"])
        if job["verify"]:
            verify_payload(path, job["entries"])
        with open(path / "payload.bin", "rb") as f:
            for p in job["placements"]:
                target = p["target"]
                rec = store.objects[target]
                mv = store.view(rec)
                f.seek(p["file_offset"])
                off = p["lo"]
                while off < p["hi"]:
                    n = f.readinto(mv[off:min(off + CHUNK_BYTES,
                                              p["hi"])])
                    if not n:
                        raise ServiceError(
                            "IO_ERROR",
                            f"short payload read for {target}")
                    off += n
        now = time.time()
        for e in job["entries"]:
            if (job["remap"] is None and identity_slice(e)
                    and e["id"] in store.objects):
                store.objects[e["id"]].protected = \
                    bool(e.get("protected", False))
        for target in job["targets"]:
            store.objects[target].last_write = {
                "by": f"restore:{job['snap_id']}", "t": now}
        with st.lock:
            st.restores[rid].update(
                state="done", restored=job["targets"],
                client_meta=job["client_meta"])
        st.emit("restore_done", path=str(path),
                n_restored=len(job["targets"]))
    except ServiceError as e:
        error = {"code": e.code, "message": str(e)}
    except Exception as e:  # noqa: BLE001 — the payload thread must survive
        error = {"code": "IO_ERROR", "message": f"{type(e).__name__}: {e}"}
    finally:
        if error is not None:
            store.rollback_created(job["created"])
            with st.lock:
                st.restores[rid].update(state="error", error=error)
            st.emit("restore_error", restore_id=rid,
                    error=error["message"])
        with st.lock:
            if rid in st.restores_in_flight:
                st.restores_in_flight.remove(rid)
        store.release_leases(job["lease_ids"])


class PayloadThread(threading.Thread):
    """The payload-IO thread: snapshot copies + snapshot.json writes
    in one direction, restore verify + placement in the other — all
    off the dispatcher. Owns nothing but its job queue."""

    def __init__(self, server):
        super().__init__(name="snapshot-payload", daemon=True)
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
            if job.get("kind") == "restore":
                restore_job(self.server, job)
            else:
                snapshot_job(self.server, job)


def install(server) -> None:
    store = server.store
    st = server.state
    st.snapshots = {}
    st.restores = {}
    st.restores_in_flight = []
    payload = PayloadThread(server)
    payload.start()
    server.payload_thread = payload
    # parked-LEASED retry hook (design: store raises, dispatcher parks)
    store.on_lease_release = server.dispatcher.unpark_all

    # ------------------------------------------------ snapshot (queued->bg)
    def snapshot(call):
        a = call.args
        dest = a["dest"]
        client_meta = a.get("client_meta") or {}
        explicit = a.get("slices")
        if explicit is not None and not explicit:
            raise ServiceError("BAD_REQUEST", "slices is empty")

        entries, offset = [], 0
        with store.catalog_lock:
            if explicit is None:
                ids = sorted(store.objects)
                if not ids:
                    raise ServiceError("BAD_REQUEST",
                                       "store is empty; nothing to "
                                       "snapshot")
                specs = [{"id": oid} for oid in ids]
            else:
                specs = explicit
            entries = resolve_slices(store, specs)
            for e in entries:
                n = e["src"][1] - e["src"][0]
                e["payload"] = {"offset": offset, "size": n}
                offset = align_up(offset + n)
            lease_ids = sorted({e["id"] for e in entries})
        snap_id = st.next_id("snap")
        # leases LAST + exception-safe: a failed admission must not
        # leak leases (leaked leases park every later write forever
        # — found when a KeyError after acquire wedged the suite)
        store.acquire_leases(lease_ids)
        with st.lock:
            st.snapshots_in_flight.append(snap_id)
            st.snapshots[snap_id] = {
                "snap_id": snap_id, "state": "writing",
                "bytes_done": 0, "bytes_total": offset,
                "n_objects": len(entries),
                "path": str(dest), "created_t": time.time(),
                "error": None,
            }
        try:
            st.emit("snapshot_started", snap_id=snap_id, path=str(dest),
                    n_objects=len(entries), bytes=offset)
        except Exception:
            store.release_leases(lease_ids)
            raise
        payload.submit({
            "snap_id": snap_id, "dest": dest,
            "client_meta": client_meta, "entries": entries,
            "lease_ids": lease_ids, "bytes_total": offset,
        })
        return {"ok": True, "snap_id": snap_id, "bytes_total": offset,
                "n_objects": len(entries)}

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
        overwrite = bool(a.get("overwrite", False))
        verify = bool(a.get("verify", True))
        remap = a.get("remap") or None
        sj = path / "snapshot.json"
        if not sj.is_file():
            raise ServiceError("IO_ERROR", f"no snapshot.json at {path}")
        snap = json.loads(sj.read_text())
        if snap.get("schema") != SNAPSHOT_SCHEMA:
            raise ServiceError(
                "VERSION_SKEW",
                f"snapshot schema {snap.get('schema')!r} != "
                f"{SNAPSHOT_SCHEMA!r}")
        entries = snap["slices"]
        if remap is not None:
            validate_remap(remap)
        placements, required = resolve_placements(entries, remap)

        # ADMISSION PASS — validate every placement; nothing mutates
        # yet (LEASED must precede any mutation: the park rule)
        with store.catalog_lock:
            for target, want in required.items():
                rec = store.objects.get(target)
                if rec is None:
                    continue
                if rec.lease_refs:
                    raise ServiceError("LEASED", target)
                if rec.size_bytes != want["size"]:
                    raise ServiceError(
                        "BINDING_MISMATCH",
                        f"{target}: resident {rec.size_bytes} B != "
                        f"snapshot {want['size']} B")
                if not overwrite:
                    raise ServiceError(
                        "COLLISION",
                        f"{target} resident; pass overwrite=True")

        # absent targets are created HERE (the dispatcher is the
        # store's single writer); a failed restore rolls them back
        created = []
        for target, want in required.items():
            if target not in store.objects:
                store.put(target, None, size_bytes=want["size"],
                          meta=want["meta"], writer="restore")
                created.append(target)
        rid = st.next_id("restore")
        lease_ids = sorted(required)
        try:
            store.acquire_leases(lease_ids)
        except Exception:
            store.rollback_created(created)
            raise
        with st.lock:
            st.restores_in_flight.append(rid)
            st.restores[rid] = {
                "restore_id": rid, "state": "restoring",
                "path": str(path), "n_slices": len(entries),
                "created_t": time.time(), "error": None,
            }
        try:
            st.emit("restore_started", restore_id=rid, path=str(path),
                    n_targets=len(required))
        except Exception:
            store.release_leases(lease_ids)
            store.rollback_created(created)
            raise
        payload.submit({
            "kind": "restore", "restore_id": rid,
            "snap_id": snap.get("snap_id"), "path": str(path),
            "entries": entries, "placements": placements,
            "targets": sorted(required), "created": created,
            "lease_ids": lease_ids, "verify": verify, "remap": remap,
            "client_meta": snap.get("client_meta", {}),
        })
        return {"ok": True, "restore_id": rid}

    # ------------------------------------------------ status (fast)
    def restore_status(conn, args):
        with st.lock:
            rec = st.restores.get(args["restore_id"])
            if rec is None:
                raise ServiceError("UNKNOWN_RESTORE", args["restore_id"])
            return dict(rec)

    server.dispatcher.handlers["snapshot"] = snapshot
    server.dispatcher.handlers["restore_snapshot"] = restore_snapshot
    server.fast_handlers["snapshot_status"] = snapshot_status
    server.fast_handlers["restore_status"] = restore_status
