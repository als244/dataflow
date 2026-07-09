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


# ===================== S1.2: execution bridge =========================
# Resolver/program/placement caches + the run path. Everything here
# imports the wider package; nothing outside bridge.py does.

_BACKEND = None
_STREAMS = None               # ONE stream trio shared by all execution
                              # contexts: torch's caching allocator is
                              # STREAM-AWARE — per-program streams made
                              # each program's cached scratch dead to the
                              # next (+4-7 GiB reserved PER PROGRAM,
                              # Shein's 29-GiB dev-20 observation).
                              # Streams are program-agnostic (one run at
                              # a time); sharing makes the cache fully
                              # reusable at zero recurring cost.
_SESSIONS: dict = {}          # prog_id -> Session (adoption is
                              # program-scoped: the engine's placement
                              # adoption records assume ONE shape-stable
                              # program per session — sharing a session
                              # across programs with overlapping object
                              # ids of different sizes corrupts it;
                              # found by the cancel gate in battery)
_RESOLVERS: dict = {}


def get_backend():
    global _BACKEND
    if _BACKEND is None:
        from dataflow.runtime.device.cuda import CudaBackend

        _BACKEND = CudaBackend()
    return _BACKEND


def get_session(prog_id: str, store=None):
    """One Session PER REGISTERED PROGRAM: streams + BufferPool +
    placement-adoption records reused across that program's runs,
    never across programs. With a real store, the pool's BACKING
    transients draw lazily from the STORE SLAB (one pinned budget;
    only the real high-water is carved — the conservative demand
    bound never allocates)."""
    if prog_id not in _SESSIONS:
        from dataflow.runtime.device.cuda import Buffer
        from dataflow.runtime.engine import Session

        global _STREAMS
        if _STREAMS is None:
            b = get_backend()
            _STREAMS = (b.create_stream("compute"),
                        b.create_stream("h2d"),
                        b.create_stream("d2h"))
        ext_pair = None
        if store is not None and store.slab is not None:
            def _alloc(size, _store=store, _owner=prog_id):
                ptr, ext, token = _store.alloc_transient(_owner, size)
                return Buffer(id=f"ext:{token}", location="backing",
                              size_bytes=size, ptr=ptr,
                              raw=("external", token))

            def _free(buf, _store=store):
                _store.free_transient(buf.raw[1])

            ext_pair = (_alloc, _free)
        _SESSIONS[prog_id] = Session(backend=get_backend(),
                                     streams=_STREAMS,
                                     external_backing=ext_pair)
    return _SESSIONS[prog_id]


def close_session(prog_id: str) -> bool:
    s = _SESSIONS.pop(prog_id, None)
    if s is not None:
        s.close()
        # NO empty_cache here (Shein: many short programs — the cost is
        # too high, and same-shape programs reuse the retained cache).
        # Only the peak COUNTER resets, so each program's device-peak
        # report is row-scoped; retained cache still counts as reserved
        # (honest under the no-drop regime).
        import torch

        torch.cuda.reset_peak_memory_stats()
        return True
    return False


def parse_program(program_dict: dict):
    from dataflow.core.jsonio import program_from_dict

    return program_from_dict(program_dict)


def _hyper_from_spec(h: dict | None):
    """Rebuild an ``AdamWHyper`` (+ optional ``LRSchedule``) from the wire
    ``hyper`` dict. Lets a client set LR / weight decay / a cosine schedule
    via ``register_program(resolver={..., 'hyper': {...}})``; without it the
    service is stuck at the built-in default (lr=1e-4, no schedule). ``None``
    -> the family default (unchanged historical behavior)."""
    from dataflow.tasks.base_blocks import AdamWHyper

    h = dict(h or {})
    sched = h.pop("schedule", None)
    if sched is not None:
        from dataflow.tasks.optim import LRSchedule

        h["schedule"] = LRSchedule(**sched)
    return AdamWHyper(**h)


def resolver_for(spec: dict):
    """(family, cfg[, hyper]) -> cached (fam, cfg_obj, dims, resolver).

    The optional ``hyper`` (lr/betas/eps/weight_decay/schedule) overrides the
    family default so LR schedules ride the resolver channel; it joins the
    cache key so different hypers don't collide. When absent, the resolver is
    built exactly as before (byte-identical to plain ``build_resolver(dims)``
    — the losses-bit-equal path is untouched)."""
    import json

    from dataflow.training.families import family as _family

    key = (spec["family"], json.dumps(spec["cfg"], sort_keys=True),
           json.dumps(spec.get("hyper"), sort_keys=True))
    hit = _RESOLVERS.get(key)
    if hit is None:
        fam = _family(spec["family"])
        cfg_obj = fam.config_type(**spec["cfg"])
        dims = fam.dims_of(cfg_obj)
        hyper = spec.get("hyper")
        resolver = (fam.build_resolver(dims, _hyper_from_spec(hyper))
                    if hyper else fam.build_resolver(dims))
        hit = (fam, cfg_obj, dims, resolver)
        _RESOLVERS[key] = hit
    return hit


def store_buffer(store, rec):
    from dataflow.runtime.device.cuda import Buffer

    return Buffer(id=f"store:{rec.id}", location="backing",
                  size_bytes=rec.size_bytes, ptr=store.ptr_of(rec),
                  raw=None)


def prepare_placement(program, values):
    """Placement + pool demand, computed once per registered program
    (train_loop does this once per train(); we cache per prog_id)."""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.placement import PlacementRecorder, compute_placement

    recorder = PlacementRecorder()
    dry = Engine(FakeBackend()).execute(program, initial_buffers=values,
                                        record_placement=recorder)
    placement = compute_placement(recorder, physical_limit_bytes=2**62)
    demand = dict(dry.pool_demand)
    dry.close()
    return placement, demand


def check_pool_headroom(pool_demand: dict) -> None:
    """A run's transient backing (session pool) is cudaHostAlloc'd
    OUTSIDE the store slab in S1. Refuse (CAPACITY) instead of letting
    the host OOM: projected pool bytes must fit in MemAvailable minus
    the system reserve. Removed when pools draw from the slab."""
    from .hostmem import GIB, PinnedSlab, meminfo_available_bytes
    from .wire import ServiceError

    projected = sum(int(size) * int(count)
                    for (loc, size), count in (pool_demand or {}).items()
                    if loc == "backing")
    avail = meminfo_available_bytes()
    reserve = int(PinnedSlab.SYSTEM_RESERVE_GIB * GIB)
    if projected > max(0, avail - reserve):
        raise ServiceError(
            "CAPACITY",
            f"run refused: session pool needs {projected / GIB:.1f} GiB "
            f"pinned but only {avail / GIB:.1f} GiB available "
            f"({reserve / GIB:.0f} GiB system reserve). Unregister idle "
            f"programs or boot with a smaller slab.",
            {"projected_pool_bytes": projected, "available": avail})


def execute_run(program, resolver, values, *, prog_id, store=None,
                placement, pool_demand, run_args, cancel_event):
    """One engine run over store-backed buffers. Returns (result,
    error_kind, error_msg); the caller owns result.close()."""
    from dataflow.runtime import Engine
    from dataflow.runtime.engine import CancelledRun, ExecutionError

    try:
        result = Engine(get_backend(),
                        session=get_session(prog_id, store=store)).execute(
            program, resolver=resolver, initial_buffers=values,
            pool_prewarm=pool_demand, placement=placement,
            run_args=run_args, cancel_event=cancel_event,
        )
        return result, None, None
    except CancelledRun as e:
        _abort_drain()
        return None, "CANCELLED", str(e)
    except ExecutionError as e:
        _abort_drain()
        return None, "RUN_FAILED", str(e)
    except Exception as e:  # noqa: BLE001 — daemon survives anything
        _abort_drain()
        return None, "RUN_FAILED", f"{type(e).__name__}: {e}"


def _abort_drain():
    """After a cancelled/failed run: the daemon's Session lives on, so
    the dead run's pending completions must not leak into the next
    run (bug found by the cancel gate: "completion for a job that is
    not in flight" on the follow-up run)."""
    n = get_backend().drain_aborted()
    return n


def capture_finals(store, program, values, result, *, writer):
    """Persist engine-produced final_locations objects into the store.
    Initial objects were store extents already (mutations landed in
    place); only NEW objects (losses etc.) need copying out."""
    import ctypes

    captured = []
    for oid, loc in (program.final_locations or {}).items():
        if oid in values:
            continue                      # store-resident all along
        rec_slot = result.objects.get(oid)
        if rec_slot is None:
            continue
        slot = rec_slot.backing or rec_slot.fast
        if slot is None:
            continue
        size = slot.buffer.size_bytes
        raw = bytes((ctypes.c_char * size).from_address(slot.buffer.ptr))
        store.put(oid, raw, writer=writer)
        captured.append(oid)
    return captured


def profile_program(program_dict: dict, spec: dict, *, refresh: bool):
    from dataflow.training.profiling import load_or_profile

    program = parse_program(program_dict)
    fam, cfg_obj, dims, resolver = resolver_for(spec)
    profiles = load_or_profile(program, resolver, get_backend(),
                               refresh=refresh)
    return {
        "profiles": {repr(k): {"runtime_us": p.runtime_us,
                               "workspace_bytes": p.workspace_bytes}
                     for k, p in profiles.items()},
        "n_signatures": len(profiles),
        "cache_path": None,
    }


def load_plugin(spec: dict):
    import importlib
    import importlib.util

    from dataflow.training import families as fam_mod

    before = set(fam_mod._FAMILIES)
    if "module" in spec:
        importlib.import_module(spec["module"])
    elif "path" in spec:
        p = spec["path"]
        mspec = importlib.util.spec_from_file_location("dataflow_plugin", p)
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
    else:
        raise ServiceError("BAD_REQUEST", "load_plugin needs module|path")
    return {"families_registered": sorted(set(fam_mod._FAMILIES) - before)}


def list_families():
    from dataflow.training import families as fam_mod

    return [{"family": name, "source": "builtin"}
            for name in sorted(fam_mod._FAMILIES)]
