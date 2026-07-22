"""Dead-everywhere semantics: an object whose last reference has passed and
that has no terminal placement frees its BACKING copy at release — in the
simulator and the runtime identically. Without this, grad-accum programs
pile up dead pinned copies round after round.

Tests:
- test_backing_freed_after_last_use: after the last use each round-tripped object has neither a fast nor a backing copy left.
- test_tight_backing_capacity_feasible_via_dead_free: a backing budget holding one object at a time still completes, at event parity with the sim.
- test_final_location_backing_survives: an object with a terminal backing placement stays live and reports no final-location violation.
"""
import pytest

from dataflow.core import ObjectSpec, OutputSpec, Program, TaskSpec, TransferDirective
from dataflow.runtime import Engine, compare_to_sim_eventlog
from dataflow.runtime.device.fake import FakeBackend
from dataflow.core.convert import to_sim_chain


def _roundtrip_chain(n_objects: int, backing_cap: int | None) -> Program:
    """n objects, each: produce -> offload -> filler -> prefetch -> consume+release.
    With dead-backing freeing, backing never holds more than one object."""
    objects = [ObjectSpec(id="seed", size_bytes=10, location="fast")]
    tasks = []
    for n in range(n_objects):
        tasks.append(TaskSpec(
            id=f"produce_{n}", inputs=("seed",), runtime_us=10.0,
            outputs=(OutputSpec(id=f"x{n}", size_bytes=100),),
            offload_after=(TransferDirective(object_id=f"x{n}", runtime_us=5.0),),
        ))
        tasks.append(TaskSpec(
            id=f"filler_{n}", inputs=("seed",), runtime_us=20.0,
            prefetch_after=(TransferDirective(object_id=f"x{n}", runtime_us=5.0),),
        ))
        tasks.append(TaskSpec(
            id=f"consume_{n}", inputs=(f"x{n}", "seed"), runtime_us=10.0,
            releases_after=(f"x{n}",),
        ))
    return Program(
        name=f"roundtrip-{n_objects}",
        initial_objects=tuple(objects),
        tasks=tuple(tasks),
        fast_memory_capacity=1_000,
        backing_memory_capacity=backing_cap,
    )


def test_backing_freed_after_last_use():
    program = _roundtrip_chain(3, backing_cap=None)
    result = Engine(FakeBackend()).execute(program)
    for n in range(3):
        rec = result.objects.get(f"x{n}")
        assert rec.fast is None and rec.backing is None, f"x{n} not fully dead"


@pytest.mark.sim
def test_tight_backing_capacity_feasible_via_dead_free():
    """Backing fits ONE object at a time: only possible when each dead
    object's backing copy frees. Sim and runtime must agree (parity too)."""
    from dataflow_sim.engine.simulator import run as sim_run

    program = _roundtrip_chain(3, backing_cap=120)  # < 2 x 100 bytes
    result = Engine(FakeBackend()).execute(program)
    log = sim_run(to_sim_chain(program), snapshots=False)
    diff = compare_to_sim_eventlog(result.trace, log)
    assert diff.ok, f"missing={diff.missing}, extra={diff.extra}"


def test_final_location_backing_survives():
    """An object with a terminal backing placement must NOT be auto-freed."""
    program = Program(
        name="keep-final",
        initial_objects=(ObjectSpec(id="w", size_bytes=100, location="fast"),),
        tasks=(
            TaskSpec(
                id="t0", inputs=("w",), runtime_us=10.0,
                offload_after=(TransferDirective(object_id="w", runtime_us=5.0),),
            ),
            TaskSpec(
                id="t1", runtime_us=20.0,
                prefetch_after=(TransferDirective(object_id="w", runtime_us=5.0),),
            ),
            TaskSpec(id="t2", inputs=("w",), runtime_us=10.0, releases_after=("w",)),
        ),
        final_locations={"w": "backing"},
        fast_memory_capacity=1_000,
    )
    result = Engine(FakeBackend()).execute(program)
    rec = result.objects.get("w")
    assert rec.backing is not None and rec.backing.state == "live"
    assert result.final_location_violations == ()