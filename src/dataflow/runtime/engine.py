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

Strict pacing (M1): the dispatcher launches a task only after its inputs'
ready events have *completed* (host-observed), so every ledger charge lands
at the same virtual time the simulator charges it — parity by construction.
The cost on real hardware is one host wake-up per task; an aggressive
dispatch-ahead mode (stream-wait on input events + committed-ahead
accounting) is an M2 experiment.
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
from .trace import Interval, RunTrace, TraceEvent
from .transfers import TransferDone, TransferEngine, TransferJob


class ExecutionError(RuntimeError):
    """A directive or task hit an invalid object state (plan/runtime bug)."""


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

    def pool_for(self, program: Program, *, placed: bool = False) -> "BufferPool":
        caps = {}
        if program.fast_memory_capacity is not None and not placed:
            # static placement replaces the fast slab + overflow arena wholesale
            caps["fast"] = program.fast_memory_capacity
        if program.backing_memory_capacity is not None:
            caps["backing"] = program.backing_memory_capacity
        if self.pool is None:
            self.pool = BufferPool(self.backend)
            if getattr(self.backend, "physical", False):
                for location, cap in caps.items():
                    self.pool.add_slab(
                        location, cap,
                        overflow_bytes=(3 * 1024**3 if location == "fast" else 0),
                    )
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
            pool = self.session.pool_for(program, placed=placement is not None)
        else:
            pool = BufferPool(self.backend)
            if getattr(self.backend, "physical", False):
                # one upfront device allocation sized to the budget (+headroom
                # +overflow arena): physical usage then tracks the ledger
                # instead of summing per-size-class maxima over time. Static
                # placement replaces the fast slab/arena wholesale.
                if program.fast_memory_capacity is not None and placement is None:
                    pool.add_slab(
                        "fast", program.fast_memory_capacity,
                        overflow_bytes=3 * 1024**3,
                    )
                if program.backing_memory_capacity is not None:
                    pool.add_slab("backing", program.backing_memory_capacity)
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
        )
        d2h = TransferEngine(
            direction="to_slow", backend=self.backend, stream=d2h_stream,
            ledger=ledger, pool=pool, table=table, trace=trace,
            bandwidth=program.bandwidth_to_slow, poison=poison,
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
        for idx, t in enumerate(program.tasks):
            task_index[t.id] = idx
            for obj_id in t.inputs:
                last_ref[obj_id] = idx
            for trig in t.offload_after:
                last_ref[trig.object_id] = idx
            for trig in t.prefetch_after:
                last_ref[trig.object_id] = idx

        state = _RunState(
            engine=self, table=table, ledger=ledger, pool=pool, trace=trace,
            h2d=h2d, d2h=d2h, deferred=deferred_prefetches, poison=poison,
            last_ref=last_ref, task_index=task_index,
            final_locations=dict(program.final_locations),
            provided_buffer_ids={id(b) for b in initial_buffers.values()},
        )
        for task in program.tasks:
            fast_out = sum(o.size_bytes for o in task.outputs if o.location == "fast")
            backing_out = sum(o.size_bytes for o in task.outputs if o.location == "backing")

            # 0. strict pacing: retire the previous task before dispatching the
            # next. Directives of ALL earlier tasks are then applied, so slot
            # states observed below reflect the plan position — without this, a
            # task whose input is live *now* could consume a copy the plan is
            # about to release/re-prefetch (stale-read), and no-input tasks
            # would reserve outputs earlier than the simulator charges them.
            while state.outstanding_task is not None:
                state.step(f"waiting for task {state.outstanding_task!r} to retire")

            # 1. inputs host-observed live in fast memory
            while True:
                waiting = [
                    obj for obj in task.inputs
                    if table.fast_state(obj) != "live"
                ]
                if not waiting:
                    break
                state.step(f"task {task.id!r} waiting for inputs {waiting} to be live in fast memory")

            # 2. fast capacity for outputs (the simulator's task stall) —
            # assigned mode adds per-offset availability to the same condition
            while not (
                ledger.can_reserve("fast", fast_out)
                and all(
                    pool.can_get(o.location, o.size_bytes, tag=o.id)
                    for o in task.outputs if o.location == "fast"
                )
            ):
                state.step(
                    f"task {task.id!r} waiting to reserve {fast_out} fast bytes "
                    f"(used={ledger.used['fast']}, cap={ledger.fast_capacity})"
                )

            # 3. backing outputs never stall (mirror sim: immediate error)
            if not ledger.can_reserve("backing", backing_out):
                raise ExecutionError(
                    f"task {task.id!r} cannot allocate {backing_out} bytes on backing: "
                    f"used={ledger.used['backing']}, capacity={ledger.backing_capacity}"
                )

            # 4. reserve outputs
            out_buffers: dict[str, Buffer] = {}
            for out in task.outputs:
                ledger.charge(out.location, out.size_bytes)
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

            # 5. launch (start/done timestamps come from the events themselves,
            # resolved in the token handler — identical mechanism on fake and
            # real backends)
            in_buffers = {obj: table.get(obj).fast.buffer for obj in task.inputs}  # type: ignore[union-attr]
            mut_buffers = {obj: in_buffers[obj] for obj in task.mutates}
            self.backend.align_stream_to_host(compute)
            start_ev = self.backend.record_event(compute)
            resolver(task).launch(TaskContext(
                task=task, stream=compute, inputs=in_buffers, outputs=out_buffers,
                mutates=mut_buffers, backend=self.backend,
            ))
            done = self.backend.record_event(compute)
            self.backend.notify_after(
                compute, done, _TaskDone(task=task, start_event=start_ev, done_event=done),
                priority=PRIORITY_TASK_DONE,
            )
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
    outstanding_task: str | None = None

    def step(self, waiting_reason: str) -> None:
        token = self.engine.backend.next_completion()
        if token is None:
            raise DeadlockError(
                f"deadlock: {waiting_reason}; no in-flight work can unblock it "
                f"(from_slow queue={[j.object_id for j in self.h2d.queue]}, "
                f"to_slow queue={[j.object_id for j in self.d2h.queue]}, "
                f"deferred={sorted(self.deferred)})"
            )
        self.handle(token)

    def handle(self, token: object) -> None:
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
