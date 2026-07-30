"""Trace capture on the FAKE backend: every engine run records a
RunTrace, and trace_to_dict wires it into the exporter/service shape the
trace tool and the run verb's trace=True path serve. This is the cheap
gate for that surface — no GPU, no daemon; planning needs the sim
extra.

Tests:
- test_trace_to_dict_covers_every_task_interval: trace_to_dict yields the exporter dict with a positive makespan, an interval for every task, a non-negative peak, and a memory_trace list.
- test_dispatch_records_cover_every_task_in_order: every task gets a DispatchRecord with monotonic phase stamps, and trace_to_dict exports them.
"""
import pytest

from dataflow.runtime import Engine
from dataflow.runtime.device.fake import FakeBackend
from dataflow.runtime.trace import trace_to_dict
from dataflow_training.model_families.families import family
from dataflow_training.model_families.llama3 import ShapedLlamaConfig



def test_trace_to_dict_covers_every_task_interval():
    cfg = ShapedLlamaConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=160,
        vocab_size=256, seq_len=32, batch=1)
    from dataflow_training.lowering.planning import plan_program

    fam = family("llama3")
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=64 * 1024 * 1024)
    program = planned.program
    backend = FakeBackend()
    initial = {o.id: backend.alloc(o.location, o.size_bytes)
               for o in program.initial_objects}
    result = Engine(backend).execute(program, initial_buffers=initial)
    d = trace_to_dict(result.trace)

    assert d["intervals"], "no task intervals recorded"
    assert d["makespan_us"] > 0
    program_ids = {t.id for t in program.tasks}
    traced_ids = {iv[0] for iv in d["intervals"]}
    missing = program_ids - traced_ids
    assert not missing, f"tasks never traced: {sorted(missing)[:5]}"
    assert d["peak_fast_bytes"] >= 0
    assert isinstance(d["memory_trace"], list)
    result.close()


def test_dispatch_records_cover_every_task_in_order():
    cfg = ShapedLlamaConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=160,
        vocab_size=256, seq_len=32, batch=1)
    from dataflow_training.lowering.planning import plan_program

    fam = family("llama3")
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=64 * 1024 * 1024)
    program = planned.program
    backend = FakeBackend()
    initial = {o.id: backend.alloc(o.location, o.size_bytes)
               for o in program.initial_objects}
    result = Engine(backend).execute(program, initial_buffers=initial)
    trace = result.trace

    recorded = [r.task_id for r in trace.dispatch]
    assert recorded == [t.id for t in program.tasks], \
        "dispatch records must cover every task in chain order"
    for r in trace.dispatch:
        assert r.pacing_done <= r.inputs_live <= r.outputs_reserved \
            <= r.launched, f"phase stamps out of order for {r.task_id}"
        assert isinstance(r.stalled_inputs, tuple)
        assert isinstance(r.stalled_outputs, tuple)

    d = trace_to_dict(trace)
    assert len(d["dispatch"]) == len(program.tasks)
    task_id, pacing, inputs, reserved, launched, sin, sout, lblock = \
        d["dispatch"][0]
    assert task_id == program.tasks[0].id
    assert isinstance(sin, list) and isinstance(sout, list)
    assert isinstance(lblock, bool)
    result.close()
