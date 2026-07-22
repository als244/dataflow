"""Program IR validation: validate_program accepts well-formed programs and
collects every structural error found in a malformed one.

Tests:
- test_valid_minimal_program: a one-object, one-task program with a single output validates.
- test_errors_collected: one raise reports duplicate initial object, missing input, negative runtime, and duplicate task id together.
- test_dual_location_initial_allowed: the same id may be an initial object in both backing and fast.
- test_dual_location_initial_size_mismatch_rejected: dual-location initials with differing sizes are rejected as inconsistent.
- test_future_input_rejected: an input produced only by a later task is rejected as not existing yet.
- test_output_collision_rejected: an output id that collides with an existing object is rejected.
- test_mutates_must_be_input: a mutates entry that is not one of the task's inputs is rejected.
- test_tensor_size_mismatch_rejected: an object whose size_bytes disagrees with its dense tensor shape and dtype is rejected.
- test_tensor_size_exact_ok: an object whose size_bytes exactly matches its tensor footprint validates.
- test_directive_requires_bandwidth: a transfer directive needs bandwidth_to_slow unless it carries a per-directive runtime override.
- test_release_and_offload_contradiction: releasing and offloading the same object after one task is rejected.
"""
import pytest

from dataflow.core import (
    ObjectSpec,
    OutputSpec,
    Program,
    TaskSpec,
    TensorMeta,
    TransferDirective,
    ValidationError,
    validate_program,
)


def _obj(oid: str, size: int = 100, loc: str = "backing") -> ObjectSpec:
    return ObjectSpec(id=oid, size_bytes=size, location=loc)


def _prog(objects, tasks, **kw) -> Program:
    return Program(name="t", initial_objects=tuple(objects), tasks=tuple(tasks), **kw)


def test_valid_minimal_program():
    p = _prog(
        [_obj("a")],
        [TaskSpec(id="t0", inputs=("a",), outputs=(OutputSpec(id="b", size_bytes=10),), runtime_us=5.0)],
    )
    validate_program(p)


def test_errors_collected():
    p = _prog(
        [_obj("a"), _obj("a")],  # duplicate initial (same location)
        [
            TaskSpec(id="t0", inputs=("missing",), runtime_us=-1.0),
            TaskSpec(id="t0", inputs=()),  # duplicate task id
        ],
    )
    with pytest.raises(ValidationError) as exc:
        validate_program(p)
    msgs = "\n".join(exc.value.errors)
    assert "duplicate initial object" in msgs
    assert "does not exist" in msgs
    assert "runtime_us must be >= 0" in msgs
    assert "duplicate task id" in msgs


def test_dual_location_initial_allowed():
    p = _prog(
        [_obj("w", loc="backing"), _obj("w", loc="fast")],
        [TaskSpec(id="t0", inputs=("w",))],
    )
    validate_program(p)


def test_dual_location_initial_size_mismatch_rejected():
    p = _prog(
        [_obj("w", size=100, loc="backing"), _obj("w", size=200, loc="fast")],
        [],
    )
    with pytest.raises(ValidationError, match="inconsistent"):
        validate_program(p)


def test_future_input_rejected():
    p = _prog(
        [_obj("a")],
        [
            TaskSpec(id="t0", inputs=("b",)),
            TaskSpec(id="t1", inputs=("a",), outputs=(OutputSpec(id="b", size_bytes=1),)),
        ],
    )
    with pytest.raises(ValidationError, match="does not exist"):
        validate_program(p)


def test_output_collision_rejected():
    p = _prog(
        [_obj("a")],
        [TaskSpec(id="t0", inputs=("a",), outputs=(OutputSpec(id="a", size_bytes=1),))],
    )
    with pytest.raises(ValidationError, match="collides"):
        validate_program(p)


def test_mutates_must_be_input():
    p = _prog([_obj("a"), _obj("b")], [TaskSpec(id="t0", inputs=("a",), mutates=("b",))])
    with pytest.raises(ValidationError, match="not one of its inputs"):
        validate_program(p)


def test_tensor_size_mismatch_rejected():
    bad = ObjectSpec(id="x", size_bytes=100, tensor=TensorMeta(dtype="bf16", shape=(4, 4)))
    p = _prog([bad], [])
    with pytest.raises(ValidationError, match="does not match dense tensor size"):
        validate_program(p)


def test_tensor_size_exact_ok():
    ok = ObjectSpec(id="x", size_bytes=32, tensor=TensorMeta(dtype="bf16", shape=(4, 4)))
    validate_program(_prog([ok], []))


def test_directive_requires_bandwidth():
    p = _prog(
        [_obj("a")],
        [TaskSpec(id="t0", inputs=("a",), offload_after=(TransferDirective(object_id="a"),))],
    )
    with pytest.raises(ValidationError, match="bandwidth_to_slow"):
        validate_program(p)
    # runtime override lifts the requirement
    p2 = _prog(
        [_obj("a")],
        [TaskSpec(id="t0", inputs=("a",), offload_after=(TransferDirective(object_id="a", runtime_us=3.0),))],
    )
    validate_program(p2)


def test_release_and_offload_contradiction():
    p = _prog(
        [_obj("a")],
        [
            TaskSpec(
                id="t0",
                inputs=("a",),
                releases_after=("a",),
                offload_after=(TransferDirective(object_id="a", runtime_us=1.0),),
            )
        ],
    )
    with pytest.raises(ValidationError, match="both released and offloaded"):
        validate_program(p)
