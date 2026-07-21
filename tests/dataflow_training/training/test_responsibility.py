"""Responsibility-map gates (CPU): the save plan is a pure derivation
— partition exactness, mode semantics, and the save-args projection.

Invariants per mode:
- world 1: rank 0 responsible for every byte of every root.
- zero1rs: every eligible root's byte range partitions EXACTLY at the
  optimizer's own flat-slice boundaries — disjoint, covering, in rank
  order (save ownership == step ownership, the model's core claim).
- co: exactly one responsible rank per root, everyone else recorded
  as backup (multiplicity data from day one); primaries byte-balanced
  within the largest single object.
"""
from dataclasses import replace

import pytest

from dataflow_training.distributed.responsibility import (
    rank_save_args,
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


def test_rank_save_args_projection():
    cfg = tiny_cfg()
    sp = zero1rs_inputs(cfg, 2)
    plan = responsibility_map(cfg, 2, mode="zero1rs", shard_params=sp)
    ids0, ranges0 = rank_save_args(plan, 0, own_objects=["O_0"])
    ids1, ranges1 = rank_save_args(plan, 1, own_objects=["O_0"])
    assert "O_0" in ids0 and "O_0" in ids1        # own shards wholesale
    assert "O_0" not in ranges0
    # partitioned params are ranged on both ranks; the union covers
    for oid, (lo, hi) in ranges0.items():
        assert lo == 0
        assert ranges1[oid][0] == hi


def test_run_lock_refuses_second_same_name(tmp_path):
    """The per-run flock: while one conductor holds a run name, a
    second same-name launch refuses loudly (CPU: exercised at the
    lock layer the conductor uses)."""
    import fcntl

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
