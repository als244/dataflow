"""Checkpoint record layer: checkpoint_record.json is the entire
contract — written atomically LAST, validated totally or refused
loudly with named ranges. Pure CPU, no daemon.

Tests:
- test_record_round_trip: write_record validates, writes atomically (record absent until complete), and read_record returns the same content with the dataflow-checkpoint/v1 schema.
- test_read_refuses_incomplete_and_foreign: a step dir without checkpoint_record.json refuses (completeness marker), as does a record carrying a foreign schema string.
- test_coverage_gap_refuses_with_named_ranges: a logical object whose slices do not union to its full byte span refuses naming the missing ranges.
- test_ambiguous_overlap_refuses: overlapping slices with no authoritative among them refuse; adding the authoritative flag on one makes the same layout legal.
- test_authoritative_twins_must_hash_equal: identical-span authoritative slices from two writers demand equal hashes (the replication drift certificate) and refuse naming both writers when they differ.
- test_misaligned_slice_refuses: with field schemas given, a slice endpoint inside an element refuses naming the field; on an element boundary it passes.
- test_digest_mismatch_refuses: with field schemas given, a schema digest that does not match the record's logical_objects entry refuses naming the object.
- test_slice_reference_bounds: a slice naming a snapshot index outside the snapshots list refuses.
"""
import json

import pytest

from dataflow.checkpoint import (
    CheckpointError,
    RECORD_SCHEMA,
    read_record,
    schema_digest,
    validate_record,
    write_record,
)

W_BYTES = 4096
O_BYTES = 8192


def base_record_kwargs():
    """A minimal valid two-writer record: replicated W_0 saved whole
    by both writers (writer 0 authoritative), O_0 sharded half/half."""
    return dict(
        step=4, seed=11,
        logical_objects={"W_0": {"bytes": W_BYTES},
                         "O_0": {"bytes": O_BYTES}},
        snapshots=[{"path": "rank0", "writer": "0"},
                   {"path": "rank1", "writer": "1"}],
        slices={
            "W_0": [
                {"snapshot": 0, "snapshot_range": [0, W_BYTES],
                 "object_range": [0, W_BYTES], "hash": "aa" * 16,
                 "authoritative": True},
                {"snapshot": 1, "snapshot_range": [0, W_BYTES],
                 "object_range": [0, W_BYTES], "hash": "aa" * 16},
            ],
            "O_0": [
                {"snapshot": 0, "snapshot_range": [0, O_BYTES // 2],
                 "object_range": [0, O_BYTES // 2], "hash": "bb" * 16,
                 "authoritative": True},
                {"snapshot": 1, "snapshot_range": [0, O_BYTES // 2],
                 "object_range": [O_BYTES // 2, O_BYTES],
                 "hash": "cc" * 16, "authoritative": True},
            ],
        },
        engine_spec={"0": {"backing_gib": 1.0},
                     "1": {"backing_gib": 1.0}},
        scheme={"world": 2, "kind": "zero1rs"},
        client_payload={"losses": [5.0, 4.0]},
    )


def test_record_round_trip(tmp_path):
    kwargs = base_record_kwargs()
    assert not (tmp_path / "checkpoint_record.json").exists()
    write_record(tmp_path, **kwargs)
    rec = read_record(tmp_path)
    assert rec["schema"] == RECORD_SCHEMA
    assert rec["step"] == 4 and rec["seed"] == 11
    assert rec["logical_objects"]["O_0"]["bytes"] == O_BYTES
    assert rec["slices"]["O_0"][1]["object_range"] == \
        [O_BYTES // 2, O_BYTES]
    assert rec["client_payload"] == {"losses": [5.0, 4.0]}
    assert rec["scheme"] == {"world": 2, "kind": "zero1rs"}


def test_read_refuses_incomplete_and_foreign(tmp_path):
    with pytest.raises(CheckpointError, match="checkpoint_record.json"):
        read_record(tmp_path)                # no record = incomplete

    write_record(tmp_path, **base_record_kwargs())
    doc = json.loads((tmp_path / "checkpoint_record.json").read_text())
    doc["schema"] = "some-other-format/v9"
    (tmp_path / "checkpoint_record.json").write_text(json.dumps(doc))
    with pytest.raises(CheckpointError, match="some-other-format/v9"):
        read_record(tmp_path)


def test_coverage_gap_refuses_with_named_ranges(tmp_path):
    kwargs = base_record_kwargs()
    del kwargs["slices"]["O_0"][1]           # second shard missing
    with pytest.raises(CheckpointError) as ei:
        write_record(tmp_path, **kwargs)
    message = str(ei.value)
    assert "O_0" in message
    assert f"[{O_BYTES // 2}, {O_BYTES})" in message, \
        "the refusal must NAME the uncovered range"
    assert not (tmp_path / "checkpoint_record.json").exists(), \
        "an invalid record must never land on disk"


def test_ambiguous_overlap_refuses(tmp_path):
    kwargs = base_record_kwargs()
    for s in kwargs["slices"]["W_0"]:
        s.pop("authoritative", None)         # two full copies, no winner
    with pytest.raises(CheckpointError, match="W_0"):
        write_record(tmp_path, **kwargs)

    kwargs["slices"]["W_0"][0]["authoritative"] = True
    write_record(tmp_path, **kwargs)         # one winner: legal


def test_authoritative_twins_must_hash_equal(tmp_path):
    kwargs = base_record_kwargs()
    twins = kwargs["slices"]["W_0"]
    twins[1]["authoritative"] = True
    twins[1]["hash"] = "dd" * 16             # replication drift
    with pytest.raises(CheckpointError) as ei:
        write_record(tmp_path, **kwargs)
    message = str(ei.value)
    assert "W_0" in message and "0" in message and "1" in message, \
        "drift refusal must name the object and both writers"

    twins[1]["hash"] = twins[0]["hash"]      # agreement: legal
    write_record(tmp_path, **kwargs)


def test_misaligned_slice_refuses():
    record = dict(base_record_kwargs())
    fields = {"O_0": [{"name": "m", "shape": [512], "dtype": "fp32",
                       "offset_bytes": 0, "size_bytes": 2048},
                      {"name": "v", "shape": [1536], "dtype": "fp32",
                       "offset_bytes": 2048, "size_bytes": 6144}]}
    record["logical_objects"]["O_0"]["schema_digest"] = \
        schema_digest(fields["O_0"])
    validate_record(build_record(record), field_schemas=fields)

    bad = dict(base_record_kwargs())
    bad["logical_objects"]["O_0"]["schema_digest"] = \
        schema_digest(fields["O_0"])
    bad["slices"]["O_0"][0]["object_range"] = [0, O_BYTES // 2 + 2]
    bad["slices"]["O_0"][0]["snapshot_range"] = [0, O_BYTES // 2 + 2]
    bad["slices"]["O_0"][1]["object_range"] = [O_BYTES // 2 + 2, O_BYTES]
    bad["slices"]["O_0"][1]["snapshot_range"] = \
        [0, O_BYTES // 2 - 2]
    with pytest.raises(CheckpointError, match="element"):
        validate_record(build_record(bad), field_schemas=fields)


def test_digest_mismatch_refuses():
    record = dict(base_record_kwargs())
    fields = {"W_0": [{"name": "w", "shape": [1024], "dtype": "fp32",
                       "offset_bytes": 0, "size_bytes": 4096}]}
    record["logical_objects"]["W_0"]["schema_digest"] = "00" * 16
    with pytest.raises(CheckpointError, match="W_0"):
        validate_record(build_record(record), field_schemas=fields)


def test_slice_reference_bounds(tmp_path):
    kwargs = base_record_kwargs()
    kwargs["slices"]["W_0"][0]["snapshot"] = 7
    with pytest.raises(CheckpointError, match="snapshot"):
        write_record(tmp_path, **kwargs)


def build_record(kwargs: dict) -> dict:
    """The record dict write_record would emit, for direct validator
    tests that need no filesystem."""
    doc = {"schema": RECORD_SCHEMA, "created_t": 0.0}
    doc.update(kwargs)
    return doc
