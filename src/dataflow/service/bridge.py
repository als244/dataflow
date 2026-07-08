"""THE sanctioned seam between the self-contained service core and the
wider dataflow package (Shein: service/ stays self-contained; family
fills and — in S1.2 — engine execution inherently ARE the wider
package, so they cross here and only here). Nothing else under
service/ imports dataflow.* modules.
"""
from __future__ import annotations

from .wire import ServiceError


def fill_family_objects(store, fill: dict, *, writer: str) -> dict:
    """materialize_group {"kind": "family_init_all"}: allocate + fill
    every initial object of a family config, in place, via the
    family's own seeded init (the refill-identity property makes this
    byte-equivalent to initial_values)."""
    if store.slab is None:
        raise ServiceError("BAD_REQUEST",
                           "family_init requires a real (pinned) boot")
    if fill.get("pattern"):
        raise ServiceError(
            "BAD_REQUEST",
            "family_init_all pattern= is deferred: the fill consumes one "
            "sequential RNG stream over ALL initial objects")

    from dataflow.runtime.device.cuda import Buffer
    from dataflow.training.families import family as _family

    fam = _family(fill["family"])
    cfg_obj = fam.config_type(**fill["cfg"])
    prog = fam.lower(cfg_obj)
    specs = list(prog.initial_objects)
    into = {}
    for s in specs:
        rec = store.put(s.id, None, size_bytes=s.size_bytes, writer=writer)
        into[s.id] = Buffer(id=f"store:{s.id}", location="backing",
                            size_bytes=s.size_bytes,
                            ptr=store.ptr_of(rec), raw=None)
    fam.initial_values(prog, cfg_obj, None,
                       seed=int(fill.get("seed", 0)), into=into)
    return {"created": [s.id for s in specs],
            "bytes": sum(s.size_bytes for s in specs)}
