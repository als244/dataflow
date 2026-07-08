"""Program + run endpoints (S1.2) — installed at boot.

Registry: prog_id -> RegisteredProgram (parsed program, resolver spec,
cached placement + pool demand). Runs execute ON the dispatcher thread
(a run occupies it end to end); cancel is a critical-lane flag the
engine observes at its task-dispatch boundary. All engine/family
imports live in bridge.py.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .wire import ServiceError


@dataclass
class RegisteredProgram:
    prog_id: str
    name: str | None
    program: object                      # parsed core Program
    program_dict: dict
    resolver_spec: dict
    registered_t: float
    placement: object | None = None      # cached on first run
    pool_demand: dict | None = None
    runs: int = 0


@dataclass
class RunRecord:
    run_id: str
    prog_id: str
    args: dict
    state: str = "queued"
    started: float | None = None
    finished: float | None = None
    tasks_total: int = 0
    makespan_us: float | None = None
    peak_fast_bytes: int = 0
    slab_overflows: int = 0
    pressure_evictions: int = 0
    placement_escapes: int = 0
    error: dict | None = None
    torch_reserved_peak: int = 0        # device-scratch RESERVED (cache incl.)
    torch_allocated_peak: int = 0       # device-scratch LIVE (apples-to-apples)
    placement_extent_bytes: int = 0     # placed device slab component
    fetched: dict = field(default_factory=dict)
    trace_tail: list = field(default_factory=list)

    def status(self) -> dict:
        return {
            "run_id": self.run_id, "prog_id": self.prog_id,
            "state": self.state, "args": self.args,
            "started": self.started,
            "elapsed_s": ((self.finished or time.time()) - self.started
                          if self.started else 0.0),
            "tasks_total": self.tasks_total,
            "makespan_us": self.makespan_us,
            "peak_fast_bytes": self.peak_fast_bytes,
            "slab_overflows": self.slab_overflows,
            "pressure_evictions": self.pressure_evictions,
            "placement_escapes": self.placement_escapes,
            "torch_reserved_peak": self.torch_reserved_peak,
            "torch_allocated_peak": self.torch_allocated_peak,
            "placement_extent_bytes": self.placement_extent_bytes,
            "error": self.error,
        }


def install(server) -> None:
    import threading

    from . import bridge

    store = server.store
    st = server.state
    programs: dict[str, RegisteredProgram] = {}
    runs: dict[str, RunRecord] = {}
    run_order: deque[str] = deque(maxlen=1024)
    active_cancel = threading.Event()
    server.programs = programs
    server.runs = runs

    # ------------------------------------------------ helpers
    def _binding_report(program) -> dict:
        adopt, create, missing = [], [], []
        with store.catalog_lock:
            for spec in program.initial_objects:
                rec = store.objects.get(spec.id)
                if rec is None:
                    missing.append(spec.id)
                elif rec.size_bytes != spec.size_bytes:
                    raise ServiceError(
                        "BINDING_MISMATCH",
                        f"{spec.id}: resident {rec.size_bytes} B != "
                        f"program {spec.size_bytes} B")
                else:
                    adopt.append(spec.id)
        produced = {o.id for t in program.tasks for o in t.outputs}
        create = sorted(produced)[:64]
        return {"adopt": adopt, "create": create,
                "missing_inputs": missing}

    def _register(call, persist: bool):
        a = call.args
        pd = a["program"]
        if isinstance(pd, str):
            pd = json.loads(Path(pd).read_text())
        canonical = json.dumps(pd, sort_keys=True,
                               separators=(",", ":")).encode()
        prog_id = "p-" + hashlib.sha256(canonical).hexdigest()[:12]
        program = bridge.parse_program(pd)
        bridge.resolver_for(a["resolver"])          # validate + cache
        report = _binding_report(program)
        cap = {
            "backing_bytes_needed": sum(s.size_bytes
                                        for s in program.initial_objects),
            "fast_extent_bytes": None,
        }
        if persist and prog_id not in programs:
            programs[prog_id] = RegisteredProgram(
                prog_id=prog_id, name=a.get("name"), program=program,
                program_dict=pd, resolver_spec=a["resolver"],
                registered_t=time.time())
            with st.lock:
                st.counters["programs_registered"] += 1
        return {"prog_id": prog_id, "name": a.get("name"),
                "bindings": report, "capacity": cap, "warnings": []}

    def register_program(call):
        return _register(call, persist=True)

    def validate_program(call):
        return _register(call, persist=False)

    def unregister_program(call):
        if call.args["prog_id"] not in programs:
            raise ServiceError("UNKNOWN_PROGRAM", call.args["prog_id"])
        del programs[call.args["prog_id"]]
        released = bridge.close_session(call.args["prog_id"])
        return {"ok": True, "session_released": released}

    def profile_program_h(call):
        a = call.args
        pd = a["program"]
        if isinstance(pd, str):
            pd = json.loads(Path(pd).read_text())
        return bridge.profile_program(pd, a["resolver"],
                                      refresh=bool(a.get("refresh")))

    def load_plugin(call):
        return bridge.load_plugin(call.args["spec"])

    # ------------------------------------------------ run
    def run(call):
        a = call.args
        entry = programs.get(a["prog_id"])
        if entry is None:
            raise ServiceError("UNKNOWN_PROGRAM", a["prog_id"])
        program = entry.program
        args = a.get("args") or {}
        rebind = a.get("rebind") or {}
        fetch = a.get("fetch") or []

        run_id = st.next_id("r")
        rec = RunRecord(run_id=run_id, prog_id=entry.prog_id, args=args,
                        tasks_total=len(program.tasks))
        runs[run_id] = rec
        run_order.append(run_id)

        # bind: every initial object must be resident (post-rebind)
        values = {}
        with store.catalog_lock:
            for spec in program.initial_objects:
                target = rebind.get(spec.id, spec.id)
                robj = store.objects.get(target)
                if robj is None:
                    rec.state = "error"
                    rec.error = {"code": "MISSING_INPUTS",
                                 "message": f"{spec.id} -> {target}"}
                    raise ServiceError("MISSING_INPUTS",
                                       f"{spec.id} -> {target} not resident")
                if robj.size_bytes != spec.size_bytes:
                    rec.state = "error"
                    rec.error = {"code": "BINDING_MISMATCH",
                                 "message": f"{spec.id} -> {target}"}
                    raise ServiceError(
                        "BINDING_MISMATCH",
                        f"{spec.id}: {target} is {robj.size_bytes} B, "
                        f"program wants {spec.size_bytes} B")
                values[spec.id] = bridge.store_buffer(store, robj)

        if entry.placement is None:
            entry.placement, entry.pool_demand = bridge.prepare_placement(
                program, values)

        import os as _os
        if _os.environ.get("DATAFLOW_SVC_DEBUG"):
            tok = next((s.size_bytes for s in program.initial_objects
                        if s.id == "tokens_0_0"), None)
            with open("/tmp/svc_debug.log", "a") as f:
                f.write(f"RUN {run_id} prog={entry.prog_id} "
                        f"entry_id={id(entry)} prog_tokens={tok} "
                        f"n_progs={len(programs)}\n")

        rec.state = "running"
        rec.started = time.time()
        active_cancel.clear()
        with st.lock:
            st.current_run = rec.status()
            st.counters["runs_total"] += 1
        st.emit("run_started", run_id=run_id, prog_id=entry.prog_id,
                args=args)

        result, err_kind, err_msg = bridge.execute_run(
            program, bridge.resolver_for(entry.resolver_spec)[3], values,
            prog_id=entry.prog_id, store=store,
            placement=entry.placement, pool_demand=entry.pool_demand,
            run_args=args, cancel_event=active_cancel)

        rec.finished = time.time()
        entry.runs += 1
        try:
            if result is None:
                rec.state = ("cancelled" if err_kind == "CANCELLED"
                             else "error")
                rec.error = {"code": err_kind, "message": err_msg}
                with st.lock:
                    st.current_run = None
                    if err_kind != "CANCELLED":
                        st.counters["runs_failed"] += 1
                st.emit("run_error" if err_kind != "CANCELLED"
                        else "run_done", run_id=run_id, error=rec.error)
                raise ServiceError(
                    err_kind, err_msg or err_kind,
                    {"prog_id": entry.prog_id,
                     "placement_cached": entry.placement is not None,
                     "initial_sizes_sample": {
                         s.id: s.size_bytes
                         for s in list(program.initial_objects)[:4]}})

            rec.makespan_us = result.makespan_us
            rec.peak_fast_bytes = result.peak_fast_bytes
            rec.placement_extent_bytes = getattr(
                entry.placement, "extent_bytes", 0) or 0
            try:
                import torch

                rec.torch_reserved_peak = torch.cuda.max_memory_reserved()
                rec.torch_allocated_peak = torch.cuda.max_memory_allocated()
            except Exception:
                rec.torch_reserved_peak = 0
            rec.slab_overflows = result.slab_overflows
            rec.pressure_evictions = result.pressure_evictions
            rec.placement_escapes = result.placement_escapes
            tr = getattr(result, "trace", None)
            rec.trace_tail = list(tr.events)[-200:] if tr is not None else []
            with st.lock:
                st.counters["tasks_executed"] += len(program.tasks)

            bridge.capture_finals(store, program, values, result,
                                  writer=run_id)
            for oid in fetch:
                data = store.get_bytes(oid) if oid in store.objects else None
                if data is None:
                    rec.fetched[oid] = None
                elif len(data) == 4:
                    rec.fetched[oid] = struct.unpack("<f", data)[0]
                else:
                    raise ServiceError(
                        "BAD_REQUEST",
                        f"fetch {oid}: {len(data)} B — only 4-byte fp32 "
                        f"objects fetch inline; use get_object")
            rec.state = "done"
            with st.lock:
                st.current_run = None
            st.emit("run_done", run_id=run_id, prog_id=entry.prog_id,
                    makespan_us=rec.makespan_us)
            out = rec.status()
            out["fetched"] = rec.fetched
            return out
        finally:
            if result is not None:
                result.close()

    # ------------------------------------------------ critical: cancel
    def cancel_run(conn, args):
        rid = args["run_id"]
        rec = runs.get(rid)
        if rec is None:
            raise ServiceError("UNKNOWN_RUN", rid)
        if rec.state == "running":
            active_cancel.set()
            return {"ok": True, "state": "cancelling"}
        return {"ok": True, "state": rec.state}

    # ------------------------------------------------ fast path
    def run_status(conn, args):
        rec = runs.get(args["run_id"])
        if rec is None:
            raise ServiceError("UNKNOWN_RUN", args["run_id"])
        return rec.status()

    def list_runs(conn, args):
        limit = int(args.get("limit", 100))
        ids = list(run_order)[-limit:]
        return [runs[i].status() for i in reversed(ids) if i in runs]

    def run_events(conn, args):
        rec = runs.get(args["run_id"])
        if rec is None:
            raise ServiceError("UNKNOWN_RUN", args["run_id"])
        tail = int(args.get("tail", 200))
        return [str(e) for e in rec.trace_tail[-tail:]]

    def list_programs(conn, args):
        return [{"prog_id": e.prog_id, "name": e.name,
                 "registered_t": e.registered_t, "runs": e.runs}
                for e in programs.values()]

    def list_families(conn, args):
        return bridge.list_families()

    def export_trace(call):
        rec = runs.get(call.args["run_id"])
        if rec is None:
            raise ServiceError("UNKNOWN_RUN", call.args["run_id"])
        dest = Path(call.args["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(str(e) for e in rec.trace_tail) + "\n")
        return {"path": str(dest), "bytes": dest.stat().st_size}

    server.dispatcher.handlers.update({
        "register_program": register_program,
        "validate_program": validate_program,
        "unregister_program": unregister_program,
        "profile_program": profile_program_h,
        "load_plugin": load_plugin,
        "run": run,
        "export_trace": export_trace,
    })
    server.fast_handlers.update({
        "run_status": run_status, "list_runs": list_runs,
        "run_events": run_events, "list_programs": list_programs,
        "list_families": list_families,
    })
    server.critical_handlers["cancel_run"] = cancel_run
