"""Z1 lowering gates (pure, no daemons): sharded-optimizer lowering —
per-rank O objects shrink to owned slots while W/dW stay full, shard
block_params carry an identical collective sequence on every rank with
disjoint whole-field ownership, programs stay JSON-serializable, and a
plain (no-shard) lowering is byte-indistinguishable from before."""
import json

import pytest

torch = pytest.importorskip("torch")

from dataflow.core.jsonio import program_to_dict
from dataflow.pretrain.fleet import lower_with_group
from dataflow.pretrain.presets import preset
from dataflow.pretrain.sharding import (
    ParallelConfig,
    layer_fields_by_root,
    zero1_halves,
)

CFG = preset("l3_125m")


def build(parallel=None):
    return lower_with_group(CFG, "dp", parallel=parallel)


def two_rank_builds():
    plan = zero1_halves(layer_fields_by_root(CFG), "dp", 2)
    return plan, [build(ParallelConfig("dp", r, 2, plan))
                  for r in (0, 1)]


def o_sizes(prog) -> dict:
    return {s.id: s.size_bytes for s in prog.initial_objects
            if s.id.startswith("O_")}


def produced_sizes(prog) -> dict:
    return {o.id: o.size_bytes for t in prog.tasks for o in t.outputs}


def test_zero1_o_shrinks_w_and_dw_do_not():
    plain = build()
    plan, (r0, r1) = two_rank_builds()
    full, s0, s1 = o_sizes(plain), o_sizes(r0), o_sizes(r1)
    assert set(full) == set(s0) == set(s1)
    tot_full = sum(full.values())
    for tot in (sum(s0.values()), sum(s1.values())):
        # half the sharded bytes + the replicated (ALL_RANKS) slack;
        # a balanced builder keeps both ranks near 0.5
        assert tot < 0.57 * tot_full, (tot, tot_full)
    assert sum(s0.values()) + sum(s1.values()) >= tot_full
    # weights identical everywhere
    w_plain = {s.id: s.size_bytes for s in plain.initial_objects
               if s.id.startswith("W_")}
    for prog in (r0, r1):
        w = {s.id: s.size_bytes for s in prog.initial_objects
             if s.id.startswith("W_")}
        assert w == w_plain
    # gradients stay FULL on every rank (reduce needs local sums)
    dw_plain = {k: v for k, v in produced_sizes(plain).items()
                if k.startswith("dW_")}
    for prog in (r0, r1):
        dw = {k: v for k, v in produced_sizes(prog).items()
              if k.startswith("dW_")}
        assert dw == dw_plain


def test_shard_block_params_consistent_across_ranks():
    plan, (r0, r1) = two_rank_builds()

    def opt_shards(prog):
        return {t.id: t.block_params.get("shard")
                for t in prog.tasks if t.group == "optimizer"}

    sh0, sh1 = opt_shards(r0), opt_shards(r1)
    assert set(sh0) == set(sh1) and sh0
    saw_rows = False
    for tid in sh0:
        a, b = sh0[tid], sh1[tid]
        assert a is not None and b is not None, tid
        # the collective sequence must match rank to rank
        assert a["comm"] == b["comm"], tid
        assert all(e["owner"] in (0, 1) for e in a["comm"])
        # whole-field sharded regions are updated by exactly one rank
        whole_sharded = {e["field"] for e in a["comm"]
                         if e["rows"] is None}
        both = set(a["update"]) & set(b["update"])
        assert not (both & whole_sharded), (tid, both & whole_sharded)
        for e in a["comm"]:
            if e["rows"] is not None:
                saw_rows = True
                lo, hi = e["rows"]
                assert 0 <= lo < hi
    # the embed (single-matrix root) shards by rows
    assert saw_rows


def test_programs_json_serializable_and_plain_unchanged():
    plan, (r0, _) = two_rank_builds()
    json.dumps(program_to_dict(r0))     # shard params survive the wire
    plain = build()
    for t in plain.tasks:
        assert "shard" not in (t.block_params or {}), t.id
    # equivalence with the parallel=None spelling
    again = lower_with_group(CFG, "dp", parallel=None)
    assert program_to_dict(again) == program_to_dict(plain)


def test_tp_lowering_params_sizes_and_serialization():
    from dataflow.pretrain.sharding import tp_mlp_shards

    plan = tp_mlp_shards(layer_fields_by_root(CFG), "dp", 2)
    plan.consumable("tp")
    plain = build()
    r0 = build(ParallelConfig("dp", 0, 2, plan))
    json.dumps(program_to_dict(r0))
    layer_w = {s.id: s.size_bytes for s in plain.initial_objects
               if s.id.startswith("W_")
               and s.id not in ("W_embed", "W_head")}
    for s in r0.initial_objects:
        if s.id in layer_w:
            # mlp halves; attention/norms replicated
            assert s.size_bytes < layer_w[s.id], s.id
    tp_fwd = tp_bwd = tp_opt = 0
    for t in r0.tasks:
        bp = t.block_params or {}
        if t.id.startswith("block_fwd_"):
            assert t.compute_block_key == "tp_block_fwd", t.id
            assert t.comm_groups == {"tp": "dp"}, t.id
            assert set(bp["tp_slices"]) == {"w1", "w3", "w2"}
            tp_fwd += 1
        if t.id.startswith("block_bwd_"):
            assert t.compute_block_key == "tp_block_bwd", t.id
            assert t.comm_groups == {"tp": "dp"}, t.id
            assert "tp_slices" in bp, t.id
            tp_bwd += 1
        if t.id.startswith("optimizer_") and "layer" in bp:
            assert bp["shard"]["grads"] == "replica", t.id
            assert t.comm_groups == {"dp": "dp"}, t.id
            assert "tp_slices" in bp, t.id
            # tp shard fields are local: no comm entries for them
            comm_fields = {e["field"] for e in bp["shard"]["comm"]}
            assert not comm_fields & {"w1", "w3", "w2"}, t.id
            assert bp["shard"]["update"].get("w1", "absent") is None
            tp_opt += 1
    assert tp_fwd and tp_bwd and tp_opt
    # embed/head replicated roots: standard owner+broadcast, replica
    for t in r0.tasks:
        if t.id.startswith("optimizer_embed"):
            sh = t.block_params["shard"]
            assert sh["grads"] == "replica"
            assert sh["comm"], "embed should broadcast owner slices"


def test_runtime_o_layout_matches_lowered_size():
    """The AdamW block sizes its O layout from the SAME update_regions
    the lowering used — assert the contract holds field by field."""
    from dataflow.pretrain.sharding import update_regions
    from dataflow.tasks.layouts import opt_state_layout
    from dataflow.training.models.llama3 import family_layouts

    plan, (r0, _) = two_rank_builds()
    dims, fl = family_layouts(CFG)
    regions = update_regions(plan, 0)
    op = getattr(dims, "opt_policy", None)
    sizes = o_sizes(r0)
    for i, ll in enumerate(fl.layers):
        got = opt_state_layout(ll.weights, dims.dtypes, layer=i,
                               opt_policy=op,
                               update_regions=regions.get(f"W_{i}"))
        assert got.total_bytes == sizes[f"O_{i}"], f"O_{i}"
    got_e = opt_state_layout(fl.embed, dims.dtypes, ns=fl.embed_ns,
                             opt_policy=op,
                             update_regions=regions.get("W_embed"))
    assert got_e.total_bytes == sizes["O_embed"]
