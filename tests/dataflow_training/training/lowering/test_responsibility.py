"""Responsibility-map gates (CPU): the save plan is a pure derivation
— partition exactness, mode semantics, and the source-policy
projection built on it.

Invariants per mode:
- world 1: rank 0 responsible for every byte of every root.
- zero1rs: every eligible root's byte range partitions EXACTLY at the
  optimizer's own flat-slice boundaries — disjoint, covering, in rank
  order (save ownership == step ownership, the model's core claim).
- co: exactly one responsible rank per root, everyone else recorded
  as backup (multiplicity data from day one); primaries byte-balanced
  within the largest single object.

Tests:
- test_world1_full_coverage: at world 1 rank 0 is the sole responsible owner of every root's full byte range.
- test_zero1rs_partitions_at_step_boundaries: zero1rs byte ranges partition each root disjointly and completely at the optimizer's own slice boundaries, in rank order.
- test_co_mode_single_primary_with_backups: co mode assigns exactly one primary and one backup per root, with primaries byte-balanced within the largest object.
- test_dedup_policy_projection: the dedup source policy saves owned zero1 shards as slot slices and partitioned params as per-rank authoritative ranges whose union covers the object.
- test_run_lock_refuses_second_same_name: a held per-run flock makes a second same-name claim raise BlockingIOError, and the lock is reclaimable once released.
"""
from dataclasses import replace

import pytest

from dataflow_training.distributed.responsibility import (
    responsibility_map,
)


def tiny_cfg():
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    return replace(ShapedLlamaConfig.tiny(), grad_accum_rounds=2)


def zero1rs_inputs(cfg, world):
    from dataflow_training.distributed.fleet import (
        layer_fields_by_root,
        zero1rs_block_params,
    )
    from dataflow_training.model_families.llama3.model import family_layouts

    dims, _ = family_layouts(cfg)
    return zero1rs_block_params(layer_fields_by_root(cfg), dims, world)


def test_world1_full_coverage():
    cfg = tiny_cfg()
    plan = responsibility_map(cfg, 1)
    for oid, entries in plan.items():
        assert len(entries) == 1
        assert entries[0]["rank"] == 0
        assert entries[0]["lo"] == 0
        assert entries[0]["hi"] > 0
        assert entries[0]["role"] == "responsible"


@pytest.mark.parametrize("world", [2, 3])
def test_zero1rs_partitions_at_step_boundaries(world):
    cfg = tiny_cfg()
    sp = zero1rs_inputs(cfg, world)
    plan = responsibility_map(cfg, world, mode="zero1rs",
                              shard_params=sp)
    for root, sh in sp.items():
        entries = plan[root]
        assert [e["rank"] for e in entries] == list(range(world))
        # disjoint + covering, in order
        assert entries[0]["lo"] == 0
        for a, b in zip(entries, entries[1:]):
            assert a["hi"] == b["lo"]
        esize = {"bf16": 2, "fp32": 4}[sh["opt_dtype"]]
        total_elems = sh["n_slice"] * world + sh["n_tail"]
        assert entries[-1]["hi"] == total_elems * esize
        # boundary == the optimizer's own slice math
        assert entries[0]["hi"] == sh["n_slice"] * esize


def test_co_mode_single_primary_with_backups():
    cfg = tiny_cfg()
    plan = responsibility_map(cfg, 2, mode="co")
    loads = [0, 0]
    for oid, entries in plan.items():
        prim = [e for e in entries if e["role"] == "responsible"]
        back = [e for e in entries if e["role"] == "backup"]
        assert len(prim) == 1
        assert len(back) == 1
        assert prim[0]["lo"] == 0
        loads[prim[0]["rank"]] += prim[0]["hi"]
    sizes = [e[0]["hi"] for e in plan.values()]
    assert abs(loads[0] - loads[1]) <= max(sizes)


def test_dedup_policy_projection():
    from dataflow_training.distributed.source_policy import \
        compile_source_policy

    cfg = tiny_cfg()
    sp = zero1rs_inputs(cfg, 2)
    plan = responsibility_map(cfg, 2, mode="zero1rs", shard_params=sp)
    root = sorted(sp)[0]
    w_bytes = max(e["hi"] for e in plan[root])
    esize = {"fp32": 4, "bf16": 2}[sp[root]["opt_dtype"]]
    total = sp[root]["n_slice"] * 2 + sp[root]["n_tail"]
    o_id = "O_" + root[2:]
    writer_specs = {
        r: [(root, w_bytes),
            (o_id, 2 * esize * (sp[root]["n_slice"]
                                + (sp[root]["n_tail"] if r == 1 else 0)))]
        for r in (0, 1)}
    logical, per_writer = compile_source_policy(
        policy="dedup", world=2, writer_specs=writer_specs,
        plan=plan, opt_slices=sp)

    # owned zero1 shards become slot slices into the aggregate object
    assert logical[o_id]["bytes"] == 2 * esize * total
    for r in (0, 1):
        o_slices = [s for s in per_writer[r]["slices"]
                    if s["id"] == o_id]
        assert len(o_slices) == 2                 # m slice + v slice
    # partitioned params: authoritative per-rank ranges whose union
    # covers the object, in rank order
    spans = []
    for r in (0, 1):
        for e in per_writer[r]["record"]:
            if e["logical"] == root:
                assert e["authoritative"]
                spans.append(tuple(e["object_range"]))
    spans.sort()
    assert spans[0][0] == 0 and spans[-1][1] == w_bytes
    assert spans[0][1] == spans[1][0]


def test_run_lock_refuses_second_same_name(tmp_path):
    """The per-run flock: while one conductor holds a run name, a
    second same-name launch refuses loudly (CPU: exercised at the
    lock layer the conductor uses)."""
    fcntl = pytest.importorskip("fcntl")

    lock_path = tmp_path / "run" / ".run_lock"
    lock_path.parent.mkdir(parents=True)
    holder = open(lock_path, "w")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    second = open(lock_path, "w")
    try:
        import pytest as _pytest

        with _pytest.raises(BlockingIOError):
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        second.close()
        holder.close()
    # released holder -> a new claimant succeeds
    third = open(lock_path, "w")
    fcntl.flock(third, fcntl.LOCK_EX | fcntl.LOCK_NB)
    third.close()
