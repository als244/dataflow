"""Engine execution machinery for the service: backend/stream/session
caches, the run path, placement prep, and final-object capture. All of
it is workload-AGNOSTIC — programs arrive parsed, resolvers arrive from
the registry (service/registry.py), buffers are store extents.
"""
from __future__ import annotations

import re

from .wire import ServiceError

# Resolver/program/placement caches + the run path. Everything here
# imports the wider package; nothing outside bridge.py does.

_BACKEND: dict = {}
_STREAMS: dict = {}           # store-id -> ONE stream trio shared by
                              # that DAEMON's execution contexts:
                              # torch's caching allocator is
                              # STREAM-AWARE — per-program streams made
                              # each program's cached scratch dead to the
                              # next (+4-7 GiB reserved PER PROGRAM,
                              # the observed 29-GiB dev case).
                              # Streams are program-agnostic WITHIN a
                              # daemon (one run at a time per daemon);
                              # scoping by store exists for in-process
                              # multi-daemon rigs, where two CONCURRENT
                              # engines on one trio corrupted transfer
                              # accounting (completion-for-unknown-job +
                              # phantom deadlocks — findings, P4a).
_SESSIONS: dict = {}          # prog_id -> Session (adoption is
                              # program-scoped: the engine's placement
                              # adoption records assume ONE shape-stable
                              # program per session — sharing a session
                              # across programs with overlapping object
                              # ids of different sizes corrupts it;
                              # found by the cancel gate in battery)
_RESOLVERS: dict = {}


def get_backend(store=None):
    """One CudaBackend PER DAEMON (store identity): the backend
    tracks in-flight transfer completions — two concurrent engines
    sharing one instance cross-contaminated tokens ("completion for a
    job that is not in flight" + phantom deadlocks; findings, P4a).
    Single-daemon processes see exactly one instance, as before."""
    key = id(store) if store is not None else 0
    if key not in _BACKEND:
        from dataflow.runtime.device.cuda import CudaBackend

        _BACKEND[key] = CudaBackend()
    return _BACKEND[key]


def session_key(prog_id: str, store) -> tuple:
    """Sessions are STORE-scoped: prog_id is a content hash, so two
    in-process daemons (or one daemon relaunched) collide on it —
    the store object is the daemon identity. (Found twice: the
    relaunch UAF, then its concurrent sibling when two live daemons
    first RAN the same program in one test process.)"""
    return (id(store) if store is not None else 0, prog_id)


def get_session(prog_id: str, store=None):
    """One Session PER (STORE, REGISTERED PROGRAM): streams +
    BufferPool + placement-adoption records reused across that
    program's runs, never across programs or daemons. With a real
    store, the pool's BACKING transients draw lazily from the STORE
    SLAB (one pinned budget; only the real high-water is carved — the
    conservative demand bound never allocates)."""
    key = session_key(prog_id, store)
    if key not in _SESSIONS:
        from dataflow.runtime.device.cuda import Buffer
        from dataflow.runtime.engine import Session

        skey = id(store) if store is not None else 0
        if skey not in _STREAMS:
            from dataflow.runtime.streams import shared_streams

            _STREAMS[skey] = shared_streams(get_backend(store),
                                            ("service", skey))
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
        _SESSIONS[key] = Session(backend=get_backend(store),
                                 streams=_STREAMS[skey],
                                 external_backing=ext_pair)
    return _SESSIONS[key]


def engine_unrecoverable() -> bool:
    """True if a prior run left this process's device context corrupted (a kernel
    illegal-access): every later run then refuses to start, so the process must
    be replaced. The context is process-wide, so one corrupted execution
    context dooms the whole process; health() surfaces this so a client can
    spawn a fresh process instead of reusing a dead one."""
    return any(s.unrecoverable for s in _SESSIONS.values())


def active_pools(store=None) -> list:
    """The engine device pools currently resident — one entry per registered
    program that has RUN (get_session creates a pool lazily on first run;
    close_session / unregister_program frees it). An observability aid for pool
    leaks: this list should stay small (the running program, plus any not yet
    unregistered). A steadily GROWING list is programs that ran but were never
    unregistered — each still pinning its fast device residency (the placement
    base). Pass a ``store`` to report only that daemon's pools. Snapshots
    _SESSIONS (the dispatcher may add/remove during a run on another thread)."""
    rows = []
    for key, s in list(_SESSIONS.items()):
        if store is not None and key[0] != id(store):
            continue
        base = getattr(s.pool, "_placement_base", None) if s.pool is not None else None
        rows.append({
            "prog_id": key[1],
            "has_pool": s.pool is not None,
            "fast_extent_bytes": getattr(base, "size_bytes", 0) if base is not None else 0,
            "unrecoverable": s.unrecoverable,
        })
    return rows


def close_session(prog_id: str, store=None) -> bool:
    s = _SESSIONS.pop(session_key(prog_id, store), None)
    if s is not None:
        s.close()
        # NO empty_cache here (many short programs — the cost is
        # too high, and same-shape programs reuse the retained cache).
        # Only the peak COUNTER resets, so each program's device-peak
        # report is row-scoped; retained cache still counts as reserved
        # (honest under the no-drop regime).
        import torch

        torch.cuda.reset_peak_memory_stats()
        return True
    return False


def close_all_sessions(store=None) -> int:
    """Drain and drop EVERY cached session. Server shutdown must call
    this BEFORE freeing the store slab: sessions hold BufferPools whose
    backing transients live in that slab (and free through the store),
    so a session that outlives its daemon dangles — the next in-process
    daemon boot re-hashes an identical program to the SAME prog_id and
    would inherit freed pointers (segfault in the first backing copy).
    Pass the daemon's ``store`` to close ONLY its sessions — required
    when multiple in-process daemons are live (fleet tests); backend
    and streams remain process-shared by design."""
    n = 0
    for key in list(_SESSIONS):
        if store is not None and key[0] != id(store):
            continue                      # another live daemon's session
        s = _SESSIONS.pop(key, None)
        if s is None:
            continue
        try:
            s.close()
            n += 1
        except Exception:
            pass
    if n:
        import torch

        torch.cuda.reset_peak_memory_stats()
    return n


def parse_program(program_dict: dict):
    from dataflow.core.jsonio import program_from_dict

    return program_from_dict(program_dict)


def store_buffer(store, rec):
    from dataflow.runtime.device.cuda import Buffer

    return Buffer(id=f"store:{rec.id}", location="backing",
                  size_bytes=rec.size_bytes, ptr=store.host_ptr(rec),
                  raw=None)


def host_task_buffer(store, rec):
    """BACKING buffer over an object's store extent for a HOST task.
    Unlike store_buffer (whose pointer feeds engine adoption/DMA and is
    therefore real-boot-only), host tasks do plain CPU writes — valid
    against the pinned slab AND the fake boot's bytearray."""
    from dataflow.runtime.device.cuda import Buffer

    return Buffer(id=f"store:{rec.id}", location="backing",
                  size_bytes=rec.size_bytes, ptr=store.host_address(rec),
                  raw=None)


def prepare_placement(program, values):
    """Placement + pool demand + chain index, computed once per
    registered program and cached per prog_id. The dry run executes
    with full validation, so a program with cached artifacts has
    passed validate_program — real runs skip re-validating it."""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.engine import build_chain_index
    from dataflow.runtime.placement import PlacementRecorder, compute_placement

    recorder = PlacementRecorder()
    dry = Engine(FakeBackend()).execute(program, initial_buffers=values,
                                        record_placement=recorder)
    if dry.pool_demand is None:
        # the dry run FAILED — surface its real error, never a
        # downstream TypeError on the missing demand dict
        msg = "placement dry-run failed"
        if dry.outcome is not None:
            msg = (f"placement dry-run failed: {dry.outcome.message}\n"
                   f"{dry.outcome.traceback_text or ''}")
        dry.close()
        raise RuntimeError(msg)
    placement = compute_placement(recorder, physical_limit_bytes=2**62)
    demand = dict(dry.pool_demand)
    dry.close()
    return placement, demand, build_chain_index(program)


class DisplayPrefix:
    """NVTX display renamer built from the run's caller-provided
    string label: every display name gets the label as a prefix. The
    service stays blind to what the label MEANS. Display-only; trace
    and plan ids are untouched."""

    def __init__(self, label: str):
        self.label = label

    def __call__(self, name: str, kind: str = "task") -> str:
        return f"{self.label} {name}"


class DisplaySub:
    """NVTX display renamer built from a caller-provided
    [pattern, replacement] pair: task display names are rewritten by
    that substitution (transfer/object names pass through untouched —
    object identity is stable across a program's replays by design).
    The service applies the pair as opaque data; what it rewrites is
    entirely the caller's vocabulary."""

    def __init__(self, pattern: str, repl: str):
        self.pattern = re.compile(pattern)
        self.repl = repl

    def __call__(self, name: str, kind: str = "task") -> str:
        if kind != "task":
            return name
        return self.pattern.sub(self.repl, name, count=1)


def display_renamer(label):
    """The run verb's label, resolved to a display renamer: a string
    is a prefix; a [pattern, replacement] pair is a task-name
    substitution; anything else (None) disables renaming."""
    if isinstance(label, str) and label:
        return DisplayPrefix(label)
    if isinstance(label, (list, tuple)) and len(label) == 2:
        return DisplaySub(str(label[0]), str(label[1]))
    return None


def execute_run(program, resolver, values, *, prog_id, store=None,
                placement, pool_demand, run_args, cancel_event,
                groups=None, poison_on_free=False, annotate_label=None,
                chain=None):
    """One engine run over store-backed buffers. Returns (result,
    error_kind, outcome); the caller owns result.close().

    The engine boundary owns the failure contract: a run-level failure or a
    cancel comes back as a RunResult carrying a FAILED/CANCELLED RunOutcome (the
    drain already ran inside execute()), so there is no post-hoc abort_drain and
    no raw exception carrying a live view. Only the engine's OWN invariant
    violations still raise (scrubbed) — those are bugs; the daemon logs them
    loud and survives, and we synthesize the SAME-shaped RunOutcome for them.

    On failure the returned ``outcome`` is THE canonical diagnostic — the very
    object an in-process caller reads off ``result.outcome`` (kind + message +
    task_id + traceback). The handler serializes it verbatim; it does not
    re-pick fields, and the client sees exactly what an in-process caller does.
    ``error_kind`` is only the wire code (RUN_FAILED / CANCELLED) the outcome
    kind maps to."""
    import traceback as tb_mod

    from dataflow.runtime import Engine
    from dataflow.runtime.engine import (
        DeadlockError, ExecutionError, RunOutcome, RunOutcomeKind)

    try:
        # validate=False: every program reaching here carried cached
        # placement artifacts, and prepare_placement's dry run already
        # validated it once — re-checking per run is pure boundary cost
        result = Engine(get_backend(store),
                        validate=False,
                        session=get_session(prog_id, store=store),
                        poison_on_free=poison_on_free).execute(
            program, resolver=resolver, initial_buffers=values,
            pool_prewarm=pool_demand, placement=placement,
            run_args=run_args, cancel_event=cancel_event, groups=groups,
            annotate_rename=display_renamer(annotate_label),
            chain=chain,
        )
    except (ExecutionError, DeadlockError) as e:
        # engine-invariant violation (a bug): the ordered abort-cleanup already
        # ran inside execute() and the raise is scrubbed of frames. Present it
        # as the same RunOutcome shape as any other failure so the client path
        # is uniform.
        outcome = RunOutcome(kind=RunOutcomeKind.FAILED,
                             message=f"{type(e).__name__}: {e}",
                             traceback_text=tb_mod.format_exc())
        print(f"[engine-invariant] {outcome.message}", flush=True)
        return None, "RUN_FAILED", outcome
    if result.outcome.is_success:
        return result, None, None
    outcome = result.outcome
    print(f"[run-failed] {outcome.message}", flush=True)
    if outcome.traceback_text:
        print(outcome.traceback_text, flush=True)
    kind = "CANCELLED" if outcome.is_cancelled else "RUN_FAILED"
    return None, kind, outcome


class HostRunResult:
    """Result shape for an all-HOST program run (execute_host_run): the
    run handler reads the same fields an engine RunResult carries;
    every device-side counter is structurally zero — there was no
    engine, no pool, no placement."""

    def __init__(self, makespan_us: float):
        self.makespan_us = makespan_us
        self.peak_fast_bytes = 0
        self.slab_overflows = 0
        self.pressure_evictions = 0
        self.placement_escapes = 0
        self.objects: dict = {}

    def close(self) -> None:
        return None


def execute_host_run(program, resolver, values, *, run_args, cancel_event):
    """Run an all-host program: every task binds its declared objects'
    STORE EXTENTS directly (the ``values`` binding the run handler
    built) and executes synchronously on the dispatcher thread — the
    store's single-writer contract, the same thread put_object mutates
    on. No engine, no placement, no transients: a host task's writes
    land in the objects' catalogued extents, which is the point —
    init-scale fills would otherwise exist twice in the slab (task
    output transient + capture copy).

    Failure contract mirrors execute_run: a raise inside a task comes
    back as a FAILED RunOutcome (kind + message + task_id + traceback)
    and the daemon survives. Cancel is honored BETWEEN tasks only —
    host tasks are synchronous CPU work with no mid-task preemption."""
    import time as time_mod
    import traceback as tb_mod

    from dataflow.runtime.engine import RunOutcome, RunOutcomeKind
    from dataflow.runtime.executable import TaskContext

    t0 = time_mod.perf_counter()
    for task in program.tasks:
        if cancel_event is not None and cancel_event.is_set():
            outcome = RunOutcome(kind=RunOutcomeKind.CANCELLED,
                                 message=f"cancelled before {task.id}",
                                 task_id=task.id)
            return None, "CANCELLED", outcome
        try:
            resolver(task).launch(TaskContext(
                task=task, stream=None,
                inputs={o: values[o] for o in task.inputs},
                outputs={},
                mutates={o: values[o] for o in task.mutates},
                backend=None, run_args=run_args))
        except Exception as e:
            outcome = RunOutcome(kind=RunOutcomeKind.FAILED,
                                 message=f"{type(e).__name__}: {e}",
                                 task_id=task.id,
                                 traceback_text=tb_mod.format_exc())
            print(f"[host-run-failed] {task.id}: {outcome.message}",
                  flush=True)
            return None, "RUN_FAILED", outcome
    return HostRunResult((time_mod.perf_counter() - t0) * 1e6), None, None


def abort_drain(store=None):
    """After a cancelled/failed run: the daemon's Session lives on, so
    the dead run's pending completions must not leak into the next
    run (bug found by the cancel gate: "completion for a job that is
    not in flight" on the follow-up run)."""
    n = get_backend(store).drain_aborted()
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


