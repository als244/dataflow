"""The resident object store: boot-lifetime slab + catalog.

Memory: ONE backing slab pinned at boot (cudaHostAlloc via the
backend — the measured ~5 GiB/s cost paid once), suballocated by a
first-fit free list with coalescing. Residents have dynamic lifetimes
(put/release/duplicate at any time), which neither plan-time placement
(packs a known schedule) nor the session pool (size-class recycling of
transients) is shaped for — hence the malloc-style allocator here.

Catalog: id -> ObjectRecord {extent, meta, lineage, protected, lease
refs} plus object groups (static membership, hierarchy, reserved pool
names). The store hands zero-copy views to the engine at run time
(S1.2) exactly where per-run values dicts go today.

Fake mode (tests, --fake boot): the slab is a plain bytearray —
identical allocator/catalog behavior, no CUDA, no torch views.
"""
from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from pathlib import Path

from .wire import ServiceError

ALIGN = 4096                       # page-aligned extents
RESERVED_SCOPES = ("backing", "fast", "all")


# --------------------------------------------------------------- allocator

@dataclass
class Extent:
    offset: int
    size: int


class SlabAllocator:
    """First-fit free list with address-ordered coalescing."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.free: list[Extent] = [Extent(0, capacity)]
        self.used_bytes = 0

    def alloc(self, size: int) -> Extent:
        need = -(-size // ALIGN) * ALIGN
        for i, ext in enumerate(self.free):
            if ext.size >= need:
                got = Extent(ext.offset, need)
                if ext.size == need:
                    self.free.pop(i)
                else:
                    ext.offset += need
                    ext.size -= need
                self.used_bytes += need
                return got
        raise ServiceError(
            "CAPACITY",
            f"backing slab exhausted: need {size} B "
            f"(aligned {need}), largest free extent "
            f"{max((e.size for e in self.free), default=0)} B",
            {"largest_free": max((e.size for e in self.free), default=0),
             "free_total": self.capacity - self.used_bytes})

    def release(self, ext: Extent) -> None:
        self.used_bytes -= ext.size
        # insert address-ordered, coalesce neighbours
        lo = 0
        for lo, e in enumerate(self.free):
            if e.offset > ext.offset:
                break
        else:
            lo = len(self.free)
        self.free.insert(lo, Extent(ext.offset, ext.size))
        merged: list[Extent] = []
        for e in self.free:
            if merged and merged[-1].offset + merged[-1].size == e.offset:
                merged[-1].size += e.size
            else:
                merged.append(e)
        self.free = merged

    def stats(self) -> dict:
        return {
            "capacity_bytes": self.capacity,
            "used_bytes": self.used_bytes,
            "free_bytes": self.capacity - self.used_bytes,
            "largest_free_extent": max((e.size for e in self.free),
                                       default=0),
            "free_extents": len(self.free),
        }


# ----------------------------------------------------------------- records

@dataclass
class Lineage:
    parent: str | None = None
    dirty: bool = False


@dataclass
class ObjectRecord:
    id: str
    size_bytes: int
    meta: dict
    extent: Extent
    protected: bool = False
    lineage: Lineage = field(default_factory=Lineage)
    lease_refs: int = 0                       # snapshot read-leases (S1.3)
    last_write: dict | None = None

    def info(self) -> dict:
        return {
            "id": self.id, "size_bytes": self.size_bytes, "meta": self.meta,
            "locations": ["backing"],          # fast residency: S2
            "protected": self.protected,
            "lineage": {"parent": self.lineage.parent,
                        "dirty": self.lineage.dirty},
            "last_write": self.last_write,
        }


@dataclass
class ObjectGroup:
    ogid: str
    members: tuple[str, ...]                  # object ids (static resolve)
    sub_groups: tuple[str, ...]

    def info(self, store: "Store") -> dict:
        ids = store.resolve_object_group(self.ogid)
        return {
            "ogid": self.ogid, "n_members": len(ids),
            "bytes": sum(store.objects[i].size_bytes
                         for i in ids if i in store.objects),
            "sub_groups": list(self.sub_groups),
            "sample_ids": ids[:8],
        }


# ------------------------------------------------------------------- store

class Store:
    """Threading contract (docs/notes/engine_service_design.md §II.3):
    ALL mutations run on the ONE dispatcher thread (single writer).
    ``catalog_lock`` guards only the catalog DICTS (objects /
    object_groups / allocator bookkeeping) so fast-path readers on
    connection threads can iterate safely; byte copies into extents
    never hold it (extent ownership makes them race-free by
    construction)."""

    def __init__(self, capacity_bytes: int, *, slab=None):
        """``slab``: a hostmem.PinnedSlab (real boot) or None (fake boot
        — plain bytearray; identical allocator/catalog behavior)."""
        import threading

        self.catalog_lock = threading.Lock()
        self.allocator = SlabAllocator(capacity_bytes)
        self.objects: dict[str, ObjectRecord] = {}
        self.object_groups: dict[str, ObjectGroup] = {}
        self.slab = slab
        self._bytes = bytearray(capacity_bytes) if slab is None else None

    # ---- raw byte access ----
    def view(self, rec: ObjectRecord) -> memoryview:
        if self._bytes is not None:
            return memoryview(self._bytes)[
                rec.extent.offset:rec.extent.offset + rec.size_bytes]
        from .hostmem import bytes_view

        return bytes_view(self.slab.ptr + rec.extent.offset, rec.size_bytes)

    def ptr_of(self, rec: ObjectRecord) -> int:
        """Absolute host address of a resident (real mode) — consumed
        by the bridge (family fills, engine adoption in S1.2)."""
        if self.slab is None:
            raise ServiceError("BAD_REQUEST",
                               "ptr_of requires a real (pinned) boot")
        return self.slab.ptr + rec.extent.offset

    # ---- create/write ----
    def put(self, oid: str, data: bytes | memoryview | None, *,
            size_bytes: int | None = None, meta: dict | None = None,
            writer: str = "put") -> ObjectRecord:
        if data is not None:
            size_bytes = len(data)
        if size_bytes is None:
            raise ServiceError("BAD_REQUEST", "put needs data or size_bytes")
        rec = self.objects.get(oid)
        if rec is not None:
            if rec.lease_refs:
                raise ServiceError("LEASED", oid)
            if rec.size_bytes != size_bytes:
                raise ServiceError(
                    "BINDING_MISMATCH",
                    f"{oid}: resident {rec.size_bytes} B != {size_bytes} B")
        else:
            with self.catalog_lock:
                ext = self.allocator.alloc(size_bytes)
                rec = ObjectRecord(oid, size_bytes, meta or {}, ext)
                self.objects[oid] = rec
        if meta:
            rec.meta = meta
        if data is not None:
            self.view(rec)[:] = bytes(data)
        rec.lineage.dirty = True
        rec.last_write = {"by": writer, "t": time.time()}
        return rec

    def put_from_file(self, oid: str, path: str, *, meta=None,
                      writer: str = "put") -> ObjectRecord:
        p = Path(path)
        if not p.is_file():
            raise ServiceError("IO_ERROR", f"no such file: {path}")
        size = p.stat().st_size
        rec = self.put(oid, None, size_bytes=size, meta=meta, writer=writer)
        with open(p, "rb") as f:
            mv = self.view(rec)
            off = 0
            while off < size:
                n = f.readinto(mv[off:off + (64 << 20)])
                if not n:
                    raise ServiceError("IO_ERROR", f"short read: {path}")
                off += n
        return rec

    # ---- read ----
    def get_bytes(self, oid: str) -> bytes:
        return bytes(self.view(self._require(oid)))

    def get_to_file(self, oid: str, path: str) -> int:
        rec = self._require(oid)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(self.view(rec))
        return rec.size_bytes

    # ---- lifecycle ----
    def release(self, oid: str, *, force: bool = False) -> bool:
        rec = self.objects.get(oid)
        if rec is None:
            return False
        if rec.protected and not force:
            raise ServiceError("PROTECTED", oid)
        if rec.lease_refs:
            raise ServiceError("LEASED", oid)
        with self.catalog_lock:
            self.allocator.release(rec.extent)
            del self.objects[oid]
        return True

    def protect(self, oid: str, value: bool) -> None:
        self._require(oid).protected = value

    def wipe(self, scope: str, *, force: bool = False) -> dict:
        ids = self.resolve_scope(scope)
        freed = n = 0
        skipped: list[str] = []
        for oid in ids:
            rec = self.objects.get(oid)
            if rec is None:
                continue
            if rec.protected and not force:
                skipped.append(oid)
                continue
            if rec.lease_refs:
                raise ServiceError("LEASED", oid)
            freed += rec.size_bytes
            n += 1
            with self.catalog_lock:
                self.allocator.release(rec.extent)
                del self.objects[oid]
        return {"freed_bytes": freed, "n_objects": n, "skipped": skipped}

    # ---- duplicate (S1: eager) ----
    def duplicate(self, src: str, dst: str) -> ObjectRecord:
        s = self._require(src)
        if dst in self.objects:
            raise ServiceError("BAD_REQUEST", f"{dst} already exists")
        with self.catalog_lock:
            ext = self.allocator.alloc(s.size_bytes)
            rec = ObjectRecord(dst, s.size_bytes, dict(s.meta), ext,
                               lineage=Lineage(parent=src, dirty=False))
            self.objects[dst] = rec
        self.view(rec)[:] = self.view(s)
        rec.last_write = {"by": f"duplicate:{src}", "t": time.time()}
        return rec

    # ---- object groups ----
    def create_object_group(self, ogid: str, members: list[str],
                            pattern: str | None,
                            sub_groups: list[str]) -> ObjectGroup:
        if ogid in RESERVED_SCOPES:
            raise ServiceError("BAD_REQUEST",
                               f"'{ogid}' is a reserved pool scope")
        if ogid in self.object_groups:
            raise ServiceError("BAD_REQUEST", f"object_group {ogid} exists")
        ids = list(members)
        for m in ids:
            self._require(m)
        if pattern:
            ids += [oid for oid in sorted(self.objects)
                    if fnmatch.fnmatch(oid, pattern) and oid not in ids]
        for g in sub_groups:
            if g not in self.object_groups:
                raise ServiceError("UNKNOWN_GROUP", g)
        grp = ObjectGroup(ogid, tuple(ids), tuple(sub_groups))
        with self.catalog_lock:
            self.object_groups[ogid] = grp
        return grp

    def resolve_object_group(self, ogid: str) -> list[str]:
        grp = self.object_groups.get(ogid)
        if grp is None:
            raise ServiceError("UNKNOWN_GROUP", ogid)
        out: list[str] = []
        seen: set[str] = set()

        def walk(g: ObjectGroup):
            for oid in g.members:
                if oid not in seen:
                    seen.add(oid)
                    out.append(oid)
            for sub in g.sub_groups:
                walk(self.object_groups[sub])

        walk(grp)
        return out

    def resolve_scope(self, scope: str) -> list[str]:
        if scope in ("backing", "all"):
            return sorted(self.objects)
        if scope == "fast":
            return []                          # fast residency: S2
        return self.resolve_object_group(scope)

    # ---- queries ----
    def usage(self) -> dict:
        st = self.allocator.stats()
        st["n_objects"] = len(self.objects)
        return st

    def largest(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(((o.id, o.size_bytes) for o in self.objects.values()),
                      key=lambda kv: -kv[1])[:n]

    def _require(self, oid: str) -> ObjectRecord:
        rec = self.objects.get(oid)
        if rec is None:
            raise ServiceError("UNKNOWN_OBJECT", oid)
        return rec
