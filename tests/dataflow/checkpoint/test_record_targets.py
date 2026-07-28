"""Targets resolution: a record plus a target set becomes
per-snapshot fetch plans in restore's remap wire shape — every
requested byte sourced exactly once, total or loud.

Resolution has two modes. LOGICAL-view targets ("all", bare ids)
assemble logical-sized objects; where coverage overlaps (hash-equal
replicas), the reader rule takes the lowest-index covering snapshot.
KEYED targets ({source_key: ids}) are the rank view: they use only
that source's own snapshot — the self-sufficiency contract — and
land shard bytes in a LOCAL object at local coordinates.

Tests:
- test_all_targets_resolve_each_byte_once: "all" resolves every logical byte exactly once — replicated spans read the lowest-index covering snapshot — into per-snapshot remap plans in restore's wire shape.
- test_id_targets_subset: a single-id target touches only the snapshots that hold its bytes.
- test_keyed_targets_infer_shard_geometry: {key: [ids]} yields that source's shard window into a local object at local coordinates, sized by the source's own ranges.
- test_uncovered_target_names_ranges: a target whose coverage has a hole refuses naming the logical object and the missing range.
- test_unknown_target_id_refuses: a target id absent from logical_objects refuses loudly.
"""
import pytest

from dataflow.checkpoint import CheckpointError, resolve_targets

from test_record_layer import O_BYTES, W_BYTES, base_record_kwargs, \
    build_record


def plan_by_snapshot(plan):
    return {step["snapshot"]: step for step in plan}


def test_all_targets_resolve_each_byte_once():
    record = build_record(base_record_kwargs())
    plan = resolve_targets(record, "all")
    steps = plan_by_snapshot(plan)

    # W_0 is covered twice by certified replicas; the reader rule
    # takes the lowest-index covering snapshot
    assert steps[0]["path"] == "rank0"
    assert steps[0]["remap"]["W_0"] == [
        {"logical": [0, W_BYTES], "id": "W_0",
         "local": [0, W_BYTES], "bytes": W_BYTES}]
    assert "W_0" not in steps.get(1, {}).get("remap", {})

    # O_0's two disjoint shards each contribute their half
    assert steps[0]["remap"]["O_0"] == [
        {"logical": [0, O_BYTES // 2], "id": "O_0",
         "local": [0, O_BYTES // 2], "bytes": O_BYTES}]
    assert steps[1]["remap"]["O_0"] == [
        {"logical": [O_BYTES // 2, O_BYTES], "id": "O_0",
         "local": [O_BYTES // 2, O_BYTES], "bytes": O_BYTES}]


def test_id_targets_subset():
    record = build_record(base_record_kwargs())
    plan = resolve_targets(record, ["W_0"])
    assert len(plan) == 1
    step = plan[0]
    assert step["snapshot"] == 0 and step["path"] == "rank0"
    assert set(step["remap"]) == {"W_0"}


def test_keyed_targets_infer_shard_geometry():
    record = build_record(base_record_kwargs())
    plan = resolve_targets(record, {"1": ["O_0", "W_0"]})
    steps = plan_by_snapshot(plan)
    assert set(steps) == {1}, \
        "the rank view uses ONLY that source's own snapshot"
    remap = steps[1]["remap"]

    # source 1's O_0 shard: logical upper half -> local [0, half)
    assert remap["O_0"] == [
        {"logical": [O_BYTES // 2, O_BYTES], "id": "O_0",
         "local": [0, O_BYTES // 2], "bytes": O_BYTES // 2}]
    # source 1's replicated W_0: full copy, local == logical size
    assert remap["W_0"] == [
        {"logical": [0, W_BYTES], "id": "W_0",
         "local": [0, W_BYTES], "bytes": W_BYTES}]


def test_uncovered_target_names_ranges():
    kwargs = base_record_kwargs()
    del kwargs["slices"]["O_0"][1]
    record = build_record(kwargs)
    with pytest.raises(CheckpointError) as ei:
        resolve_targets(record, ["O_0"], validate=False)
    message = str(ei.value)
    assert "O_0" in message
    assert f"[{O_BYTES // 2}, {O_BYTES})" in message


def test_unknown_target_id_refuses():
    record = build_record(base_record_kwargs())
    with pytest.raises(CheckpointError, match="W_missing"):
        resolve_targets(record, ["W_missing"])
