"""Store endpoint handlers — installed into the Server at boot.

Queued (dispatcher thread): put/get/materialize/release/protect/
duplicate/object-group mutations/wipe. Fast path (connection threads,
under state.lock): query_object/list_objects/query_object_group/
query_backing/query_fast. The Store itself is single-writer
(dispatcher); fast-path readers only touch plain dicts under the
lock.
"""
from __future__ import annotations

from .store import Store
from .wire import ServiceError


def boot_store(server) -> Store:
    cfg = server.config
    if cfg.fake:
        cap = cfg.slab_backing_gib
        cap_bytes = (256 << 20) if cap == "auto" else int(float(cap) * 2**30)
        return Store(cap_bytes, slab=None)
    from .hostmem import PinnedSlab, auto_cap_bytes

    cap_bytes = (auto_cap_bytes(reserve_gib=10.0)
                 if cfg.slab_backing_gib == "auto"
                 else int(float(cfg.slab_backing_gib) * 2**30))
    store = Store(cap_bytes, slab=PinnedSlab(cap_bytes, device=cfg.device))
    try:  # device fixed overhead (CUDA context) for peak decomposition
        import torch

        free_b, total_b = torch.cuda.mem_get_info(cfg.device)
        server.device_fixed_bytes = total_b - free_b
    except Exception:
        server.device_fixed_bytes = 0
    return store


def install(server) -> None:
    store: Store = server.store
    st = server.state

    # ---------------- queued ----------------
    def put_object(call):
        a = call.args
        if a.get("path"):
            rec = store.put_from_file(a["id"], a["path"],
                                      meta=a.get("meta"),
                                      writer=call.session_id)
        else:
            if call.payload is None:
                raise ServiceError("BAD_REQUEST",
                                   "put_object needs payload or path")
            rec = store.put(a["id"], call.payload, meta=a.get("meta"),
                            writer=call.session_id)
        return {"object": rec.info()}

    def get_object(call):
        a = call.args
        if a.get("dest"):
            n = store.get_to_file(a["id"], a["dest"])
            return {"path": a["dest"], "bytes": n}
        return {"object": store._require(a["id"]).info()}, \
            store.get_bytes(a["id"])

    def materialize_object(call):
        return _materialize(call.args["id"], call.args["fill"], call)

    def _materialize(oid, fill, call):
        kind = fill.get("kind")
        if kind == "zeros":
            rec = store.put(oid, None, size_bytes=int(fill["size_bytes"]),
                            writer=call.session_id)
            store.view(rec)[:] = b"\x00" * rec.size_bytes
            return {"object": rec.info()}
        if kind == "file":
            rec = store.put_from_file(oid, fill["path"],
                                      writer=call.session_id)
            return {"object": rec.info()}
        if kind == "tokens":
            import numpy as np

            rng = np.random.default_rng(int(fill["seed"]))
            ids = rng.integers(0, int(fill["vocab"]), size=int(fill["n"]),
                               dtype=np.int32)
            rec = store.put(oid, ids.tobytes(), writer=call.session_id)
            return {"object": rec.info()}
        if kind in ("family_init_all", "family_init"):
            raise ServiceError(
                "BAD_REQUEST",
                f"fill kind {kind!r} is retired: the engine no longer "
                "knows model families. Model/optimizer-state init is a "
                "program — register + run it through a registered "
                "resolver kind (dataflow_training.run.driver.init_model)")
        raise ServiceError("BAD_REQUEST", f"unknown fill kind {kind!r}")

    def release_object(call):
        a = call.args
        freed = store.release(a["id"], force=bool(a.get("force")))
        return {"freed": {"backing": freed, "fast": False}}

    def protect_object(call):
        store.protect(call.args["id"], True)
        return {"ok": True}

    def unprotect_object(call):
        store.protect(call.args["id"], False)
        return {"ok": True}

    def duplicate_object(call):
        rec = store.duplicate(call.args["src"], call.args["dst"])
        return {"object": rec.info()}

    def duplicate_object_group(call):
        a = call.args
        ids = store.resolve_object_group(a["ogid"])
        tag = a["tag"]
        rename = a.get("rename", "{id}@{tag}")
        mapping = {}
        for oid in ids:
            dst = rename.format(id=oid, tag=tag)
            store.duplicate(oid, dst)
            mapping[oid] = dst
        new_ogid = a.get("new_ogid") or f"{a['ogid']}@{tag}"
        grp = store.create_object_group(new_ogid, list(mapping.values()),
                                        None, [])
        return {"object_group": grp.info(store), "mapping": mapping}

    def create_object_group(call):
        a = call.args
        grp = store.create_object_group(
            a["ogid"], list(a.get("members", ())),
            a.get("pattern"), list(a.get("object_groups", ())))
        return {"object_group": grp.info(store)}

    def delete_object_group(call):
        with store.catalog_lock:
            if call.args["ogid"] not in store.object_groups:
                raise ServiceError("UNKNOWN_GROUP", call.args["ogid"])
            del store.object_groups[call.args["ogid"]]
        return {"ok": True}

    def wipe(call):
        return store.wipe(call.args["scope"],
                          force=bool(call.args.get("force")))

    server.dispatcher.handlers.update({
        "put_object": put_object, "get_object": get_object,
        "materialize_object": materialize_object,
        "release_object": release_object,
        "protect_object": protect_object,
        "unprotect_object": unprotect_object,
        "duplicate_object": duplicate_object,
        "duplicate_object_group": duplicate_object_group,
        "create_object_group": create_object_group,
        "delete_object_group": delete_object_group,
        "wipe": wipe,
    })

    # ---------------- fast path ----------------
    def query_object(conn, args):
        with store.catalog_lock:
            rec = store.objects.get(args["id"])
            return rec.info() if rec else None

    def list_objects(conn, args):
        import fnmatch as _fn

        pat = args.get("pattern", "*")
        limit = int(args.get("limit", 1000))
        with store.catalog_lock:
            out = [store.objects[k].info()
                   for k in sorted(store.objects)
                   if _fn.fnmatch(k, pat)][:limit]
        return out

    def query_object_group(conn, args):
        with store.catalog_lock:
            grp = store.object_groups.get(args["ogid"])
            if grp is None:
                raise ServiceError("UNKNOWN_GROUP", args["ogid"])
            info = grp.info(store)
            info["members"] = [store.objects[i].info()
                               for i in store.resolve_object_group(grp.ogid)
                               if i in store.objects]
            return info

    def query_backing(conn, args):
        with store.catalog_lock:
            u = store.usage()
            u["largest"] = store.largest()
            return u

    def query_fast(conn, args):
        return {"capacity_bytes": 0, "used_bytes": 0, "n_objects": 0,
                "note": "fast residency across runs lands in S2"}

    server.fast_handlers.update({
        "query_object": query_object, "list_objects": list_objects,
        "query_object_group": query_object_group,
        "query_backing": query_backing, "query_fast": query_fast,
    })
