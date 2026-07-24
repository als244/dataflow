"""Group-annotation pass equivalence: the parallelism-blind pipeline
(fam.lower -> annotate_groups -> exact sizes with the rank view) must
reproduce the old in-builder grouped lowering EXACTLY — program dicts
identical — before any non-llama3 family rides the pass. CPU-only.

Tests:
- test_dp_annotation_matches_builder: the dp annotation pipeline reproduces the old grouped-lowering program dict and its pinned digest.
- test_zero1rs_annotation_matches_builder: the zero1rs annotation pipeline reproduces the builder's program dict and pinned digest.
- test_tp_annotation_matches_builder_both_ranks: the tp annotation pipeline reproduces the builder's program dict and pinned digest for each rank.
"""
from dataclasses import replace as dc_replace

from dataflow.core.jsonio import program_to_dict
from dataflow_training.distributed.fleet import lower_with_group
from dataflow_training.distributed.group_annotation import annotate_groups
from dataflow_training.lowering.emit import apply_exact_sizes, object_size_factory
from dataflow_training.model_families.llama3 import (
    ShapedLlamaConfig,
    family_layouts,
    lower_llama3,
)

TINY = dc_replace(ShapedLlamaConfig.tiny(), grad_accum_rounds=2)
GROUP = "dp"

# Digests of the CERTIFIED grouped lowerings — the anti-tautology anchor: both
# code paths share the pipeline, so these constants are what prove a rewire
# changed nothing.
#
# Re-pinned DELIBERATELY when tasks gained a cost_key (the geometry two tasks
# can differ in while their buffers look identical). The move was proved
# additive rather than semantic: stripping cost_key from the new program
# reproduces the previous digest exactly (dp: 63b7dd11deb8839b), and both code
# paths still agree at every rank.
PINNED = {
    "dp": "ecf68f9b0a05792e",
    "zero1rs": "25a3f4029fcea871",
    "tp_r0": "28559715f557430f",
    "tp_r1": "0a89ebb312e31990",
}


def digest(prog) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(program_to_dict(prog),
                   sort_keys=True).encode()).hexdigest()[:16]


def annotated(cfg, *, shard_params=None, tp_params=None,
              opt_regions=None, opt_slices=None, rank_view=None):
    """The new pipeline: blind lower -> annotate -> narrow -> re-size."""
    from dataflow_training.lowering.emit import narrow_layouts

    program = lower_llama3(cfg)
    program = annotate_groups(program, group=GROUP,
                              shard_params=shard_params,
                              tp_params=tp_params)
    dims, fl = family_layouts(cfg)
    if rank_view:
        fl = narrow_layouts(fl, rank_view)
    return apply_exact_sizes(
        program, "llama3-exact",
        object_size=object_size_factory(
            dims, fl, opt_update_regions=opt_regions,
            opt_slice_by_root=opt_slices))


def test_dp_annotation_matches_builder():
    old = lower_with_group(TINY, GROUP)
    assert digest(old) == PINNED["dp"]
    assert program_to_dict(old) == program_to_dict(annotated(TINY))


def test_zero1rs_annotation_matches_builder():
    from dataflow_training.distributed.fleet import (
        layer_fields_by_root,
        zero1rs_block_params,
    )

    world = 2
    dims0, _ = family_layouts(TINY)
    shard_params = zero1rs_block_params(
        layer_fields_by_root(TINY), dims0, world)
    opt_slices = {root: {"n_slice": sh["n_slice"],
                         "n_tail": sh["n_tail"],
                         "opt_dtype": sh["opt_dtype"]}
                  for root, sh in shard_params.items()}
    old = lower_with_group(TINY, GROUP, zero1rs_world=world)
    assert digest(old) == PINNED["zero1rs"]
    assert program_to_dict(old) == program_to_dict(
        annotated(TINY, shard_params=shard_params, opt_slices=opt_slices))


def test_tp_annotation_matches_builder_both_ranks():
    from dataflow_training.distributed.fleet import layer_fields_by_root
    from dataflow_training.distributed.sharding import (
        ParallelConfig,
        tp_mlp_shards,
        tp_opt_block_params,
        tp_view,
    )

    world = 2
    plan = tp_mlp_shards(layer_fields_by_root(TINY), GROUP, world)
    for rank in range(world):
        parallel = ParallelConfig(group=GROUP, rank=rank, world=world, plan=plan)
        rank_view = tp_view(plan, rank)
        tp_params = {
            root: {name: list(sl) for name, sl in slices.items()}
            for root, slices in rank_view.items()}
        shard_params = tp_opt_block_params(plan, rank)
        opt_regions = {root: dict(sh["update"])
                       for root, sh in shard_params.items()}
        old = lower_with_group(TINY, GROUP, parallel=parallel)
        assert digest(old) == PINNED[f"tp_r{rank}"]
        assert program_to_dict(old) == program_to_dict(
            annotated(TINY, shard_params=shard_params, tp_params=tp_params,
                      opt_regions=opt_regions, rank_view=rank_view)), \
            f"rank {rank} diverged"
