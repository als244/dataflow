"""The generic dataflow engine.

One control thread executes an annotated program over a DeviceBackend:

- **Dispatcher** walks the chain in order. For each task it (1) waits until
  every input's fast copy is host-observed live, (2) waits until the ledger
  can reserve the task's fast outputs, then (3) reserves, launches the task's
  executable on the compute stream, and registers a task-done token. Waiting
  means processing completion tokens — never sleeping, never polling device
  state.
- **Directives** fire inside the task-done token handler (release / offload /
  prefetch, anchored on the task-end event), mirroring the simulator's
  task-end processing order exactly.
- **Transfer engines** run fully asynchronously: a blocked transfer head
  never blocks the dispatcher.

Strict pacing: the dispatcher launches a task only after its inputs'
ready events have *completed* (host-observed), so every ledger charge lands
at the same virtual time the simulator charges it — parity by construction.
The cost on real hardware is one host wake-up per task; an aggressive
dispatch-ahead mode (stream-wait on input events + committed-ahead
accounting) is a deferred experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from dataflow.core import ObjectSpec, Program, TaskSpec, validate_program

from .device.base import PRIORITY_TASK_DONE, Buffer, DeviceBackend, Event
from .executable import ExecutableResolver, TaskContext, synthetic_resolver
from .ledger import Ledger
from .objects import ObjectRecord, ObjectTable, Slot
from .pool import BufferPool
from .device.annotate import NoopAnnotator
from .trace import Interval, RunTrace, TraceEvent
from .transfers import TransferDone, TransferEngine, TransferJob


class CancelledRun(RuntimeError):
    """Boundary-cancel (engine service): raised between task
    dispatches when the caller's cancel_event is set; unwinds through
    the same cleanup path as ExecutionError."""


class ExecutionError(RuntimeError):
    """A directive or task hit an invalid object state (plan/runtime bug)."""


_NOOP_ANNOTATOR = NoopAnnotator()


class DeadlockError(RuntimeError):
    """The engine is blocked and no in-flight work can unblock it."""


@dataclass(frozen=True)
class _TaskDone:
    task: TaskSpec
    start_event: Event
    done_event: Event


@dataclass(frozen=True)
class RunResult:
    trace: RunTrace
    makespan_us: float
    peak_fast_bytes: int
    final_location_violations: tuple[str, ...]
    buffers_allocated: int
    buffers_reused: int
    slab_overflows: int
    peak_backing_bytes: int = 0
    placement_escapes: int = 0  # cumulative pool counter (see slab_overflows)
    pressure_evictions: int = 0  # quiescent-deadlock ledger evictions (this run)
    # exact (location, size) -> count buffer demand of this run; feed it to a
    # subsequent run's `pool_prewarm` (e.g. fake dry run -> real run)
    pool_demand: dict[tuple[str, int], int] = None  # type: ignore[assignment]
    # final object table: records with live slots/buffers so callers can read
    # results (losses, updated params) after the run
    objects: ObjectTable = None  # type: ignore[assignment]
    # engine-local pool (None when a Session owns it): call close() after
    # readback to release the slab/arena — tests and one-shot runs that skip
    # this leak the reservations for the process lifetime
    _owned_pool: BufferPool = None  # type: ignore[assignment]

    def close(self) -> None:
        if self._owned_pool is not None:
            self._owned_pool.drain()


@dataclass
class Session:
    """Long-lived device state reused across Engine.execute() calls.

    Multi-step training replays the same annotated chain once per optimizer
    step; re-creating the pool each step would re-allocate the device slab
    and re-pin tens of GB of host memory. A Session owns the backend and the
    BufferPool (with its slabs) so steady-state steps perform zero vendor
    allocations. Slabs are keyed to the first program's capacities; reusing a
    session across programs with different budgets raises.
    """

    backend: DeviceBackend
    pool: "BufferPool | None" = None
    # (alloc_fn, free_fn) for backing transients from an external owner
    # (engine service store slab); see BufferPool.external_alloc
    external_backing: tuple | None = None
    slab_caps: dict[str, int] | None = None
    # streams created once and reused every execute(): torch's caching
    # allocator keys cached blocks per-stream, so per-execute stream churn
    # multiplies the scratch cache by the number of steps
    streams: tuple | None = None

    def streams_for(self) -> tuple:
        if self.streams is None:
            self.streams = (
                self.backend.create_stream("compute"),
                self.backend.create_stream("h2d"),
                self.backend.create_stream("d2h"),
            )
        return self.streams

    def pool_for(
        self, program: Program, *, placed: bool = False, vmm: bool = False
    ) -> "BufferPool":
        caps = {}
        if program.fast_memory_capacity is not None and not (placed or vmm):
            # static placement / vmm replace the fast slab + overflow arena
            caps["fast"] = program.fast_memory_capacity
        if program.backing_memory_capacity is not None:
            caps["backing"] = program.backing_memory_capacity
        if self.pool is None:
            self.pool = BufferPool(self.backend)
            if self.external_backing is not None:
                self.pool.external_alloc["backing"] = self.external_backing
            if getattr(self.backend, "physical", False):
                for location, cap in caps.items():
                    self.pool.add_slab(
                        location, cap,
                        overflow_bytes=(3 * 1024**3 if location == "fast" else 0),
                    )
                if vmm:
                    if program.fast_memory_capacity is None:
                        raise ValueError("vmm fast mode requires a fast capacity")
                    from .device.vmm import VmmArena

                    self.pool.enable_vmm(VmmArena(
                        device_index=getattr(self.backend, "device", 0),
                        capacity_bytes=program.fast_memory_capacity,
                        event_complete=self.backend.event_complete,
                    ))
            elif vmm:
                raise ValueError("vmm fast mode requires a physical backend")
            self.slab_caps = caps
        elif caps != self.slab_caps:
            raise ValueError(
                f"session slabs sized for {self.slab_caps}; program needs {caps} — "
                f"use a fresh Session per budget"
            )
        return self.pool

    def close(self) -> None:
        if self.pool is not None:
            self.pool.drain()
            self.pool = None


@dataclass
class Engine:
    backend: DeviceBackend
    validate: bool = True
    strict_final_locations: bool = True
    # debug: fill freed buffers with 0xFF (NaN in bf16/fp32) at retirement so
    # any use-after-release surfaces as a NaN explosion instead of silent
    # corruption. The memset enqueues on the compute stream; h2d reuse waits
    # on the recorded guard event.
    poison_on_free: bool = False
    # long-lived pool/slab state for repeated execute() calls (multi-step)
    session: Session | None = None

    def execute(
        self,
        program: Program,
        resolver: ExecutableResolver | None = None,
        *,
        initial_buffers: Mapping[str, Buffer] | None = None,
        pool_prewarm: Mapping[tuple[str, int], int] | None = None,
        placement=None,           # runtime.placement.Placement: assigned mode
        record_placement=None,    # runtime.placement.PlacementRecorder: dry runs
        vmm: bool = False,        # fast buffers from a VMM arena (device/vmm.py)
        run_args: Mapping[str, object] | None = None,   # -> TaskContext.run_args
        cancel_event=None,        # threading.Event: boundary-cancel (service)
        annotate_rename=None,     # Callable[[str], str]: NVTX display names only
                                  # (a replayed 1-step plan bakes step 0 into every
                                  # id; the caller knows the GLOBAL step and rewrites
                                  # names for the profiler — trace/plan ids untouched)
    ) -> RunResult:
        if self.validate:
            validate_program(program)
        resolver = resolver or synthetic_resolver
        initial_buffers = dict(initial_buffers or {})

        if self.session is not None:
            compute, h2d_stream, d2h_stream = self.session.streams_for()
        else:
            compute = self.backend.create_stream("compute")
            h2d_stream = self.backend.create_stream("h2d")
            d2h_stream = self.backend.create_stream("d2h")

        trace = RunTrace()
        ledger = Ledger(
            fast_capacity=program.fast_memory_capacity,
            backing_capacity=program.backing_memory_capacity,
        )

        def _on_ledger_change(location: str, used: int) -> None:
            if location == "fast":
                trace.memory_trace.append((self.backend.host_now_us(), used))

        ledger.on_change = _on_ledger_change
        if self.session is not None:
            if self.session.backend is not self.backend:
                raise ValueError("session backend differs from engine backend")
            pool = self.session.pool_for(
                program, placed=placement is not None, vmm=vmm,
            )
        else:
            pool = BufferPool(self.backend)
            if getattr(self.backend, "physical", False):
                # one upfront device allocation sized to the budget (+headroom
                # +overflow arena): physical usage then tracks the ledger
                # instead of summing per-size-class maxima over time. Static
                # placement / vmm replace the fast slab/arena wholesale.
                if (
                    program.fast_memory_capacity is not None
                    and placement is None
                    and not vmm
                ):
                    pool.add_slab(
                        "fast", program.fast_memory_capacity,
                        overflow_bytes=3 * 1024**3,
                    )
                if vmm:
                    if program.fast_memory_capacity is None:
                        raise ValueError("vmm fast mode requires a fast capacity")
                    from .device.vmm import VmmArena

                    pool.enable_vmm(VmmArena(
                        device_index=getattr(self.backend, "device", 0),
                        capacity_bytes=program.fast_memory_capacity,
                        event_complete=self.backend.event_complete,
                    ))
                if program.backing_memory_capacity is not None:
                    pool.add_slab("backing", program.backing_memory_capacity)
            elif vmm:
                raise ValueError("vmm fast mode requires a physical backend")
        pool.recorder = record_placement
        if placement is not None and pool.placement is None:
            pool.enable_placement(placement)
        elif pool.placement is not None:
            pool.reset_placement_epoch()
        table = ObjectTable()

        poison = None
        if self.poison_on_free:
            def poison(buffer: Buffer) -> None:  # noqa: F811
                if buffer.location != "fast":
                    return
                self.backend.memset_async(buffer, 0xFF, compute)
                buffer.guard_event = self.backend.record_event(compute)

        h2d = TransferEngine(
            direction="from_slow", backend=self.backend, stream=h2d_stream,
            ledger=ledger, pool=pool, table=table, trace=trace,
            bandwidth=program.bandwidth_from_slow, poison=poison,
            annotate_rename=annotate_rename,
        )
        d2h = TransferEngine(
            direction="to_slow", backend=self.backend, stream=d2h_stream,
            ledger=ledger, pool=pool, table=table, trace=trace,
            bandwidth=program.bandwidth_to_slow, poison=poison,
            annotate_rename=annotate_rename,
        )
        deferred_prefetches: dict[str, list[TransferJob]] = {}

        # --- setup: prewarm pool + load initial memory, then zero the clock ----
        if pool_prewarm:
            pool.prewarm(dict(pool_prewarm))
        setup_copies: list[tuple[Buffer, Buffer, int]] = []  # (dst_fast, src_backing, size)
        for spec in program.initial_objects:
            self._load_initial(spec, table, ledger, pool, initial_buffers, setup_copies)
        if setup_copies and getattr(self.backend, "physical", False):
            # upload provided initial values into pre-placed fast copies
            for dst, src, size in setup_copies:
                self.backend.memcpy_async(dst, src, size, h2d_stream)
            self.backend.sync_all()
        self.backend.mark_origin()

        # --- dispatch loop -----------------------------------------------------
        # Last chain position referencing each object (inputs or transfer
        # directives): a release at-or-after it, for an object with no
        # final_locations entry, frees the BACKING copy too — mirroring the
        # simulator's dead-everywhere rule (essential at high grad-accum where
        # per-round context would otherwise pile up dead pinned copies).
        last_ref: dict[str, int] = {}
        task_index: dict[str, int] = {}
        uses_by_obj: dict[str, list[int]] = {}  # input positions (eviction valve)
        directives_by_obj: dict[str, list[int]] = {}  # any release/offload/prefetch
        for idx, t in enumerate(program.tasks):
            task_index[t.id] = idx
            for obj_id in t.inputs:
                last_ref[obj_id] = idx
                uses_by_obj.setdefault(obj_id, []).append(idx)
            for obj_id in t.releases_after:
                directives_by_obj.setdefault(obj_id, []).append(idx)
            for trig in t.offload_after:
                last_ref[trig.object_id] = idx
                directives_by_obj.setdefault(trig.object_id, []).append(idx)
            for trig in t.prefetch_after:
                last_ref[trig.object_id] = idx
                directives_by_obj.setdefault(trig.object_id, []).append(idx)

        state = _RunState(
            engine=self, table=table, ledger=ledger, pool=pool, trace=trace,
            h2d=h2d, d2h=d2h, deferred=deferred_prefetches, poison=poison,
            last_ref=last_ref, task_index=task_index, uses_by_obj=uses_by_obj,
            directives_by_obj=directives_by_obj,
            final_locations=dict(program.final_locations),
            provided_buffer_ids={id(b) for b in initial_buffers.values()},
        )
        annotator = getattr(self.backend, "annotator", None) or _NOOP_ANNOTATOR
        import os as _os
        import time as _time
        stats = None
        if _os.environ.get("DATAFLOW_DISPATCH_STATS") == "1":
            stats = {k: 0.0 for k in (
                "prev_token_wait", "input_wait", "reserve_wait",
                "reserve_bookkeeping", "launch_enqueue",
            )}
            stats["tasks"] = 0
            stats["token_detect_lat"] = 0.0
            stats["token_detect_n"] = 0
            state.stats = stats
        for task_pos, task in enumerate(program.tasks):
            # service boundary-cancel: observed ONLY here, between task
            # dispatches — in-flight work drains normally, the ledger
            # stays consistent, and the finally-path releases buffers
            # exactly as an ExecutionError would
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledRun(
                    f"cancelled before task {task.id!r} "
                    f"({task_pos}/{len(program.tasks)})")
            fast_out = sum(o.size_bytes for o in task.outputs if o.location == "fast")
            backing_out = sum(o.size_bytes for o in task.outputs if o.location == "backing")
            state.cursor = task_pos
            state.current_task_id = task.id
            protected = frozenset(task.inputs)  # never evict what this task needs

            # 0. strict pacing: retire the previous task before dispatching the
            # next. Directives of ALL earlier tasks are then applied, so slot
            # states observed below reflect the plan position — without this, a
            # task whose input is live *now* could consume a copy the plan is
            # about to release/re-prefetch (stale-read), and no-input tasks
            # would reserve outputs earlier than the simulator charges them.
            _t = _time.perf_counter() if stats is not None else 0.0
            while state.outstanding_task is not None:
                state.step(f"waiting for task {state.outstanding_task!r} to retire",
                           exclude=protected)
            if stats is not None:
                stats["prev_token_wait"] += _time.perf_counter() - _t
                stats["tasks"] += 1

            # 1. inputs host-observed live in fast memory
            _t = _time.perf_counter() if stats is not None else 0.0
            while True:
                waiting = [
                    obj for obj in task.inputs
                    if table.fast_state(obj) != "live"
                ]
                if not waiting:
                    break
                state.step(f"task {task.id!r} waiting for inputs {waiting} to be live in fast memory",
                           exclude=protected)
            if stats is not None:
                stats["input_wait"] += _time.perf_counter() - _t

            # 2. fast capacity for outputs (the simulator's task stall) —
            # assigned mode adds per-offset availability to the same condition
            escaped_outputs: set[str] = set()
            was_reserve_blocked = False
            _t = _time.perf_counter() if stats is not None else 0.0
            while True:
                can_res = ledger.can_reserve("fast", fast_out)
                busy = [
                    o.id for o in task.outputs
                    if o.location == "fast" and o.id not in escaped_outputs
                    and not pool.can_get(o.location, o.size_bytes, tag=o.id)
                ]
                if can_res and not busy:
                    if stats is not None:
                        stats["reserve_wait"] += _time.perf_counter() - _t
                    break
                token = self.backend.next_completion()
                if token is not None:
                    state.handle(token)
                    continue
                # quiescent. A pure placed-offset conflict (ledger admits,
                # offsets busy) is a lifetime inversion vs the dry run — the
                # holder's release depends on a LATER task, so waiting is a
                # deadlock. Escape those instances to dynamic allocations
                # (counted, reported) and proceed; genuine capacity blocks
                # still deadlock loudly below.
                if can_res and busy:
                    for oid in busy:
                        escaped_outputs.add(oid)
                        trace.events.append(TraceEvent(
                            t=self.backend.host_now_us(), kind="placement_escape",
                            object_id=oid, task_id=task.id,
                        ))
                    continue
                state.step(
                    f"task {task.id!r} waiting to reserve {fast_out} fast bytes "
                    f"(used={ledger.used['fast']}, cap={ledger.fast_capacity})",
                    exclude=protected, pump_h2d=False,
                )
                was_reserve_blocked = True

            # 3. backing outputs never stall (mirror sim: immediate error)
            if not ledger.can_reserve("backing", backing_out):
                raise ExecutionError(
                    f"task {task.id!r} cannot allocate {backing_out} bytes on backing: "
                    f"used={ledger.used['backing']}, capacity={ledger.backing_capacity}"
                )

            # 4. reserve outputs
            _t = _time.perf_counter() if stats is not None else 0.0
            out_buffers: dict[str, Buffer] = {}
            for out in task.outputs:
                ledger.charge(out.location, out.size_bytes)
                if out.id in escaped_outputs:
                    buf = pool.get_escaped(out.location, out.size_bytes, tag=out.id)
                else:
                    buf = pool.get(out.location, out.size_bytes, tag=out.id)
                rec = table.add(
                    ObjectRecord(id=out.id, size_bytes=out.size_bytes, role=out.role, tensor=out.tensor)
                )
                if rec.slot(out.location) is not None:
                    raise ExecutionError(
                        f"task {task.id!r} output {out.id!r} collides with an existing "
                        f"{out.location} slot"
                    )
                rec.set_slot(out.location, Slot(buffer=buf, state="reserved", version=rec.version))
                out_buffers[out.id] = buf
                trace.events.append(TraceEvent(
                    t=self.backend.host_now_us(), kind="reserve", object_id=out.id, task_id=task.id,
                ))

            if stats is not None:
                stats["reserve_bookkeeping"] += _time.perf_counter() - _t
            if was_reserve_blocked:
                # the lane was suppressed while this reserve accumulated
                # its bytes — restart the parked head now that outputs are
                # charged
                state.h2d.try_start()

            # 5. launch (start/done timestamps come from the events themselves,
            # resolved in the token handler — identical mechanism on fake and
            # real backends)
            _t = _time.perf_counter() if stats is not None else 0.0
            in_buffers = {obj: table.get(obj).fast.buffer for obj in task.inputs}  # type: ignore[union-attr]
            mut_buffers = {obj: in_buffers[obj] for obj in task.mutates}
            self.backend.align_stream_to_host(compute)
            start_ev = self.backend.record_event(compute)
            annotator.range_push(annotate_rename(task.id) if annotate_rename else task.id)
            try:
                resolver(task).launch(TaskContext(
                    task=task, stream=compute, inputs=in_buffers, outputs=out_buffers,
                    mutates=mut_buffers, backend=self.backend,
                    run_args=run_args,
                ))
            finally:
                annotator.range_pop()
            done = self.backend.record_event(compute)
            self.backend.notify_after(
                compute, done, _TaskDone(task=task, start_event=start_ev, done_event=done),
                priority=PRIORITY_TASK_DONE,
            )
            if stats is not None:
                stats["launch_enqueue"] += _time.perf_counter() - _t
            state.outstanding_task = task.id

        # --- drain -------------------------------------------------------------
        while True:
            token = self.backend.next_completion()
            if token is None:
                break
            state.handle(token)

        stuck: list[str] = []
        if h2d.queue:
            stuck.append(f"from_slow queue: {[j.object_id for j in h2d.queue]}")
        if d2h.queue:
            stuck.append(f"to_slow queue: {[j.object_id for j in d2h.queue]}")
        if deferred_prefetches:
            stuck.append(f"deferred prefetches: {sorted(deferred_prefetches)}")
        if stuck:
            raise DeadlockError(
                "run finished with transfers still queued — destination capacity "
                "can never admit them. " + "; ".join(stuck)
            )

        if stats is not None:
            n = max(stats["tasks"], 1)
            print("dispatch stats (per task, us):")
            for k in ("prev_token_wait", "input_wait", "reserve_wait",
                      "reserve_bookkeeping", "launch_enqueue"):
                print(f"  {k:22} {stats[k] / n * 1e6:9.1f}")
            if stats["token_detect_n"]:
                print(f"  {'token_detect_latency':22} "
                      f"{stats['token_detect_lat'] / stats['token_detect_n'] * 1e6:9.1f}"
                      f"   (n={stats['token_detect_n']})")
        violations = self._final_location_violations(program, table)
        if violations and self.strict_final_locations:
            raise ExecutionError(
                "final_locations violated: " + "; ".join(violations)
            )

        # End-of-run reclamation: the program is over, so every pool-owned
        # buffer still held by an object returns to the pool (essential when a
        # Session reuses the pool across steps — otherwise the slab bleeds).
        # Caller-provided initial buffers are not pool-owned and stay out.
        # Contents remain readable until a later run reuses the buffer:
        # read results BEFORE the next execute().
        provided = {id(b) for b in initial_buffers.values()}
        for rec in table.records.values():
            for slot in (rec.fast, rec.backing):
                if slot is not None and id(slot.buffer) not in provided:
                    pool.put(slot.buffer)

        trace.peak_fast_bytes = ledger.peak_fast_bytes
        return RunResult(
            trace=trace,
            makespan_us=trace.makespan_us(),
            peak_fast_bytes=ledger.peak_fast_bytes,
            peak_backing_bytes=ledger.peak_backing_bytes,
            final_location_violations=tuple(violations),
            buffers_allocated=pool.allocated_count,
            buffers_reused=pool.reused_count,
            slab_overflows=pool.slab_overflows,
            placement_escapes=pool.placement_escapes,
            pressure_evictions=state.pressure_evictions,
            pool_demand=dict(pool.allocated_by_key),
            objects=table,
            _owned_pool=None if self.session is not None else pool,
        )

    # ------------------------------------------------------------------------

    def _load_initial(
        self,
        spec: ObjectSpec,
        table: ObjectTable,
        ledger: Ledger,
        pool: BufferPool,
        initial_buffers: Mapping[str, Buffer],
        setup_copies: list[tuple[Buffer, Buffer, int]],
    ) -> None:
        rec = table.add(ObjectRecord(
            id=spec.id, size_bytes=spec.size_bytes, role=spec.role, tensor=spec.tensor,
        ))
        if rec.slot(spec.location) is not None:
            raise ExecutionError(
                f"duplicate initial slot for ({spec.id!r}, {spec.location!r})"
            )
        ledger.charge(spec.location, spec.size_bytes)
        provided = initial_buffers.get(spec.id)
        if provided is not None and provided.size_bytes < spec.size_bytes:
            raise ExecutionError(
                f"initial buffer for {spec.id!r} is {provided.size_bytes} bytes; "
                f"object needs {spec.size_bytes}"
            )
        if provided is not None and provided.location == spec.location:
            buf = provided
        else:
            buf = pool.get(spec.location, spec.size_bytes, tag=spec.id)
            if provided is not None and spec.location == "fast" and provided.location == "backing":
                # provided pinned values feed a pre-placed fast copy at setup
                setup_copies.append((buf, provided, spec.size_bytes))
        rec.set_slot(spec.location, Slot(buffer=buf, state="live", version=0))

    def _final_location_violations(self, program: Program, table: ObjectTable) -> list[str]:
        violations: list[str] = []
        for obj_id, loc in program.final_locations.items():
            rec = table.records.get(obj_id)
            slot = rec.slot(loc) if rec is not None else None
            if slot is None or slot.state != "live":
                state = "absent" if slot is None else slot.state
                violations.append(f"{obj_id!r} not live on {loc} (state={state})")
            elif rec is not None and slot.version != rec.version:
                violations.append(
                    f"{obj_id!r} on {loc} is stale (slot v{slot.version} != latest v{rec.version})"
                )
        return violations


@dataclass
class _RunState:
    """Token handlers — every state change happens here, at token times."""

    engine: Engine
    table: ObjectTable
    ledger: Ledger
    pool: BufferPool
    trace: RunTrace
    h2d: TransferEngine
    d2h: TransferEngine
    deferred: dict[str, list[TransferJob]]
    poison: object = None
    last_ref: dict[str, int] = None  # type: ignore[assignment]
    task_index: dict[str, int] = None  # type: ignore[assignment]
    final_locations: dict[str, str] = None  # type: ignore[assignment]
    provided_buffer_ids: set[int] = None  # type: ignore[assignment]
    uses_by_obj: dict[str, list[int]] = None  # type: ignore[assignment]
    directives_by_obj: dict[str, list[int]] = None  # type: ignore[assignment]
    outstanding_task: str | None = None
    stats: dict | None = None
    cursor: int = 0                      # dispatch position (eviction valve)
    current_task_id: str | None = None
    pressure_evictions: int = 0
    prefetch_override: dict[str, float | None] = None  # type: ignore[assignment]

    def step(self, waiting_reason: str, *, exclude: frozenset = frozenset(),
             pump_h2d: bool = True) -> None:
        token = self.engine.backend.next_completion()
        if token is None:
            # a parked from_slow head may be admissible again (frees since
            # it parked). Input waits MUST pump — the awaited object
            # arrives via this lane. Output-reserve waits pass
            # pump_h2d=False: outputs never arrive by lane, and admitting
            # a head there hands the blocked task's bytes to a later
            # consumer (the poke-starvation spiral).
            if pump_h2d and self.h2d.inflight is None and self.h2d.queue:
                self.h2d.try_start()
                if self.h2d.inflight is not None:
                    return
            # quiescent: a prefetch head blocked only by a placed-offset
            # conflict is the same lifetime inversion — escape it and retry
            if self.h2d.escape_blocked_head():
                self.trace.events.append(TraceEvent(
                    t=self.engine.backend.host_now_us(), kind="placement_escape",
                    object_id=self.h2d.inflight.object_id if self.h2d.inflight else None,
                    detail="from_slow head",
                ))
                return
            # ledger inversion: real transfer/compute timing let admitted
            # prefetches race ahead of the release frontier and strand the
            # ledger in a state the (sim-verified) plan never visits. Evict
            # the farthest-next-use CLEAN resident back to its backing copy
            # and reload it before its use — semantically the sim's own
            # "deferred prefetch", decided late. Budget cap is never exceeded
            # (eviction only frees); genuine capacity deadlocks still raise.
            evicted = self._evict_for_pressure(exclude)
            if evicted in ("evicted", "evicted-poke"):
                # freed bytes go to the STALLED CALLER, never the transfer
                # head: control returns to the blocked loop, which re-checks
                # before any lane admission can run (single dispatcher).
                # Poking the head here — the sim's transfer-first tie —
                # was the poke-starvation feedback: each eviction's bytes
                # fed the next trigger-satisfied prefetch (or the prior
                # evictee's own reload) while the blocked task starved to
                # the thrash guard (observed 1080 = 10 x 108 tasks). The
                # valve is a runtime-only recovery path the simulator
                # never visits, so caller-priority here diverges from NO
                # sim-modeled timeline; gated prefetches simply admit on
                # the next token after the caller reserves.
                return
            raise DeadlockError(
                f"deadlock: {waiting_reason}; no in-flight work can unblock it "
                f"(from_slow queue={[j.object_id for j in self.h2d.queue]}, "
                f"to_slow queue={[j.object_id for j in self.d2h.queue]}, "
                f"deferred={sorted(self.deferred)}, "
                f"pressure_evictions={self.pressure_evictions})"
            )
        self.handle(token)

    def _evict_for_pressure(self, exclude: frozenset) -> str | None:
        """Evict ONE clean fast-resident object (valid backing copy, current
        version, not needed by the stalled task) with the farthest next use,
        and queue its reload. Returns None when no safe victim exists, else
        whether a pre-existing queue head should be poked with the bytes."""
        if self.pressure_evictions >= 10 * len(self.task_index):
            return None  # thrash guard: let the deadlock surface loudly
        best = None  # (next_use, size, oid, rec)
        for oid, rec in self.table.records.items():
            fast, backing = rec.fast, rec.backing
            if (
                oid in exclude
                or fast is None or fast.state != "live"
                or backing is None or backing.state != "live"
                or fast.version != rec.version
                or backing.version != rec.version
            ):
                continue
            uses = self.uses_by_obj.get(oid)
            if not uses or uses[-1] < self.cursor:
                continue  # no future use -> a later plan-release would misfire
            from bisect import bisect_left
            nxt = uses[bisect_left(uses, self.cursor)]
            dirs = self.directives_by_obj.get(oid, ())
            # window [cursor, nxt): the current task is un-launched, so its
            # directives are still pending and would misfire on an evicted slot
            if dirs and bisect_left(dirs, self.cursor) < bisect_left(dirs, nxt):
                continue  # plan touches this slot before the next use
            key = (nxt, rec.size_bytes)
            if best is None or key > (best[0], best[1]):
                best = (nxt, rec.size_bytes, oid, rec)
        if best is None:
            return None
        nxt, size, oid, rec = best
        had_head_before = bool(self.h2d.queue) or self.h2d.inflight is not None
        slot = rec.fast
        self.ledger.release("fast", size)
        if self.poison is not None:
            self.poison(slot.buffer)  # type: ignore[operator]
        self.pool.put(slot.buffer)
        rec.fast = None
        self.pressure_evictions += 1
        now = self.engine.backend.host_now_us()
        self.trace.events.append(TraceEvent(
            t=now, kind="pressure_evict", object_id=oid,
            task_id=self.current_task_id, detail=f"next_use_task={nxt}",
        ))
        # reload before its next use; anchor is already-complete by construction;
        # duration mirrors the object's own prefetch trigger (bandwidth fallback)
        override = (self.prefetch_override or {}).get(oid)
        self.h2d.enqueue(TransferJob(
            object_id=oid, size_bytes=size, runtime_override=override,
            anchor_event=self.engine.backend.record_event(self.h2d.stream),
            fired_by_task=self.current_task_id or "pressure_evict",
        ))
        return "evicted-poke" if had_head_before else "evicted"

    def handle(self, token: object) -> None:
        if self.stats is not None and isinstance(token, _TaskDone):
            lat = (self.engine.backend.host_now_us()
                   - self.engine.backend.event_time_us(token.done_event))
            self.stats["token_detect_lat"] += lat / 1e6
            self.stats["token_detect_n"] += 1
        if isinstance(token, _TaskDone):
            self._on_task_done(token)
        elif isinstance(token, TransferDone):
            self._on_transfer_done(token)
        else:  # pragma: no cover
            raise AssertionError(f"unknown completion token {token!r}")

    # --- task end -------------------------------------------------------------

    def _on_task_done(self, tok: _TaskDone) -> None:
        task = tok.task
        now = self.engine.backend.host_now_us()
        if self.outstanding_task == task.id:
            self.outstanding_task = None

        # outputs become live
        for out in task.outputs:
            rec = self.table.get(out.id)
            slot = rec.slot(out.location)
            assert slot is not None and slot.state == "reserved"
            slot.state = "live"
            slot.ready_event = tok.done_event

        # mutations advance versions; backing copies go stale
        for obj in task.mutates:
            rec = self.table.get(obj)
            rec.version += 1
            assert rec.fast is not None
            rec.fast.version = rec.version
            rec.fast.ready_event = tok.done_event
            self.trace.events.append(TraceEvent(t=now, kind="mutate", object_id=obj, task_id=task.id))

        self.trace.intervals.append(
            Interval(
                task_id=task.id,
                start=self.engine.backend.event_time_us(tok.start_event),
                end=self.engine.backend.event_time_us(tok.done_event),
                track="compute",
            )
        )

        # releases: instantaneous at task end; state must be live
        for obj in task.releases_after:
            rec = self.table.get(obj)
            slot = rec.fast
            if slot is None or slot.state != "live":
                state = "absent" if slot is None else slot.state
                raise ExecutionError(
                    f"task {task.id!r} cannot release {obj!r}: fast state={state!r} (must be live)"
                )
            self.ledger.release("fast", rec.size_bytes)
            if self.poison is not None:
                self.poison(slot.buffer)  # type: ignore[operator]
            self.pool.put(slot.buffer)
            rec.fast = None
            # dead everywhere: no later reference + no terminal placement ->
            # the backing copy frees too (mirrors the simulator's rule)
            if (
                self.task_index[task.id] >= self.last_ref.get(obj, -1)
                and obj not in self.final_locations
                and rec.backing is not None
                and rec.backing.state == "live"
            ):
                self.ledger.release("backing", rec.size_bytes)
                backing_buf = rec.backing.buffer
                rec.backing = None
                # caller-provided buffers (initial values) are not pool-owned:
                # dropping the slot suffices, ownership stays with the caller
                if id(backing_buf) not in self.provided_buffer_ids:
                    self.pool.put(backing_buf)
            self.trace.events.append(TraceEvent(t=now, kind="release", object_id=obj, task_id=task.id))

        # offloads: enqueue to_slow (no backing bytes charged until start)
        for trig in task.offload_after:
            rec = self.table.get(trig.object_id)
            src = rec.fast
            if src is None or src.state != "live":
                state = "absent" if src is None else src.state
                raise ExecutionError(
                    f"task {task.id!r} cannot offload {trig.object_id!r}: fast state={state!r}"
                )
            existing = rec.backing
            if existing is not None:
                if existing.state != "live":
                    raise ExecutionError(
                        f"task {task.id!r} cannot offload {trig.object_id!r}: backing entry "
                        f"not live (state={existing.state!r})"
                    )
                if existing.buffer.size_bytes != rec.size_bytes:
                    raise ExecutionError(
                        f"task {task.id!r} offload size mismatch for {trig.object_id!r}"
                    )
            src.state = "pending_outbound"
            self.d2h.enqueue(TransferJob(
                object_id=trig.object_id, size_bytes=rec.size_bytes,
                runtime_override=trig.runtime_us, anchor_event=tok.done_event,
                fired_by_task=task.id,
            ))

        # prefetches: enqueue from_slow, or defer while the object's offload
        # is still in flight (activated at that offload's completion)
        for trig in task.prefetch_after:
            rec = self.table.get(trig.object_id)
            dev, src = rec.fast, rec.backing
            if dev is not None and dev.state in ("live", "inbound"):
                raise ExecutionError(
                    f"task {task.id!r} cannot prefetch {trig.object_id!r}: fast copy "
                    f"already exists (state={dev.state!r})"
                )
            offload_in_flight = (
                (dev is not None and dev.state in ("pending_outbound", "outbound"))
                or (src is not None and src.state in ("pending_inbound", "inbound"))
            )
            job = TransferJob(
                object_id=trig.object_id, size_bytes=rec.size_bytes,
                runtime_override=trig.runtime_us, anchor_event=tok.done_event,
                fired_by_task=task.id,
            )
            if self.prefetch_override is None:
                self.prefetch_override = {}
            self.prefetch_override[trig.object_id] = trig.runtime_us
            if offload_in_flight:
                self.deferred.setdefault(trig.object_id, []).append(job)
                self.trace.events.append(TraceEvent(
                    t=now, kind="transfer_deferred", object_id=trig.object_id, task_id=task.id,
                ))
            elif src is None or src.state != "live":
                state = "absent" if src is None else src.state
                raise ExecutionError(
                    f"task {task.id!r} cannot prefetch {trig.object_id!r}: no recoverable "
                    f"backing source (state={state!r})"
                )
            else:
                self.h2d.enqueue(job)

        # sim step 11: poke queues (from_slow first — sim's tie order)
        self.h2d.try_start()
        self.d2h.try_start()

    # --- transfer end -----------------------------------------------------------

    def _on_transfer_done(self, tok: TransferDone) -> None:
        if tok.direction == "from_slow":
            self.h2d.complete(tok.job)
            self.h2d.try_start()
            return
        self.d2h.complete(tok.job)
        # activate any prefetches that were waiting on this offload
        for job in self.deferred.pop(tok.job.object_id, ()):
            self.h2d.enqueue(job)
        self.d2h.try_start()
        self.h2d.try_start()
