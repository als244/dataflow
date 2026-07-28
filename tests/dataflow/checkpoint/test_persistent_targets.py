"""The persistent marker and Program-form targets: checkpoint
selection is the marker filter, and a program's binding sizes are
validated against the record's geometry before any plan is emitted.

Tests:
- test_marker_default_and_emit_when_true: ObjectSpec.persistent defaults False and serializes only when true, so an unmarked program's dict — and its content id — never carries the field.
- test_persisted_objects_filter: persisted_objects returns exactly the marked specs.
- test_program_targets_identity_and_keyed: an identity-sized program resolves as the logical view; a shard-sized program under its writer key resolves to that writer's windows into a local object.
- test_program_targets_size_mismatch_refuses: a shard-sized program without a key refuses pointing at the missing writer key; a wrong-sized binding under a key refuses naming both byte counts.
"""
import pytest

from dataflow.checkpoint import (CheckpointError, persisted_objects,
                                 resolve_targets)
from dataflow.core.jsonio import program_from_dict, program_to_dict
from dataflow.core.program import ObjectSpec, Program

from test_record_layer import O_BYTES, W_BYTES, base_record_kwargs, \
    build_record


def state_program(o_bytes: int) -> Program:
    return Program(
        name="state",
        initial_objects=(
            ObjectSpec("W_0", W_BYTES, persistent=True),
            ObjectSpec("O_0", o_bytes, persistent=True),
            ObjectSpec("tokens_0_0", 128, role="input"),
        ))


def test_marker_default_and_emit_when_true():
    assert ObjectSpec("x", 4).persistent is False
    prog = state_program(O_BYTES)
    d = program_to_dict(prog)
    by_id = {o["id"]: o for o in d["initial_objects"]}
    assert by_id["W_0"]["persistent"] is True
    assert "persistent" not in by_id["tokens_0_0"], \
        "unmarked specs must not carry the field (prog_id stability)"
    back = program_from_dict(d)
    marks = {s.id: s.persistent for s in back.initial_objects}
    assert marks == {"W_0": True, "O_0": True, "tokens_0_0": False}


def test_persisted_objects_filter():
    prog = state_program(O_BYTES)
    assert [s.id for s in persisted_objects(prog)] == ["W_0", "O_0"]


def test_program_targets_identity_and_keyed():
    record = build_record(base_record_kwargs())

    # identity-sized program: the logical view
    plan = resolve_targets(record, state_program(O_BYTES))
    remaps = {step["snapshot"]: step["remap"] for step in plan}
    assert set(remaps[0]) == {"W_0", "O_0"}
    assert remaps[1]["O_0"][0]["logical"] == [O_BYTES // 2, O_BYTES]

    # shard-sized program under its writer key: the rank view
    plan = resolve_targets(record, {"1": state_program(O_BYTES // 2)})
    assert [step["snapshot"] for step in plan] == [1]
    remap = plan[0]["remap"]
    assert remap["O_0"] == [
        {"logical": [O_BYTES // 2, O_BYTES], "id": "O_0",
         "local": [0, O_BYTES // 2], "bytes": O_BYTES // 2}]
    assert remap["W_0"][0]["local"] == [0, W_BYTES]


def test_program_targets_size_mismatch_refuses():
    record = build_record(base_record_kwargs())
    with pytest.raises(CheckpointError, match="writer key"):
        resolve_targets(record, state_program(O_BYTES // 2))
    with pytest.raises(CheckpointError) as ei:
        resolve_targets(record, {"1": state_program(O_BYTES)})
    message = str(ei.value)
    assert str(O_BYTES) in message and str(O_BYTES // 2) in message
