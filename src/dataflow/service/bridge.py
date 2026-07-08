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


def get_session(prog_id: str):
    """One Session PER REGISTERED PROGRAM: streams + BufferPool +
    placement-adoption records reused across that program's runs
    (steady-state steps stay zero-vendor-call), never across
    programs."""
    if prog_id not in _SESSIONS:
        from dataflow.runtime.engine import Session

        _SESSIONS[prog_id] = Session(backend=get_backend())
    return _SESSIONS[prog_id]


def close_session(prog_id: str) -> bool:
    s = _SESSIONS.pop(prog_id, None)
    if s is not None:
        s.close()
        return True
    return False


def parse_program(program_dict: dict):
    from dataflow.core.jsonio import program_from_dict

    return program_from_dict(program_dict)


def resolver_for(spec: dict):
    """(family, cfg) -> cached (fam, cfg_obj, dims, resolver)."""
    import json

    from dataflow.training.families import family as _family

    key = (spec["family"], json.dumps(spec["cfg"], sort_keys=True))
    hit = _RESOLVERS.get(key)
    if hit is None:
        fam = _family(spec["family"])
        cfg_obj = fam.config_type(**spec["cfg"])
        dims = fam.dims_of(cfg_obj)
        hit = (fam, cfg_obj, dims, fam.build_resolver(dims))
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


def execute_run(program, resolver, values, *, prog_id, placement,
                pool_demand, run_args, cancel_event):
    """One engine run over store-backed buffers. Returns (result,
    error_kind, error_msg); the caller owns result.close()."""
    from dataflow.runtime import Engine
    from dataflow.runtime.engine import CancelledRun, ExecutionError

    try:
        result = Engine(get_backend(),
                        session=get_session(prog_id)).execute(
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
