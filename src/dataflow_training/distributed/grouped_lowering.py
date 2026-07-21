"""Grouped lowering: the three composable passes behind every rank
program — blind family lowering, group annotation, exact sizes with
the narrowed rank view. Split from the conductor (fleet.py) at phase
close; fleet re-exports these names."""
from dataflow_training.lowering.planning import plan_program  # noqa: F401
from dataflow_training.lowering.shaped_program import ShapedHardware

from .sharding import (
    ALL_RANKS,
    layer_fields_by_root,
    shard_block_params,
    tp_opt_block_params,
    tp_view,
    update_regions,
    zero1rs_block_params,
)

def lower_with_group(cfg, dp_group: str, recompute_levels=None,
                     parallel=None,
                     zero1rs_world: int | None = None):
    """``parallel`` (sharding.ParallelConfig with a plan) makes this a
    PER-RANK lowering. An optimizer-consumable plan (zero1): optimizer
    tasks gain shard block_params and the rank's O objects shrink to
    owned slots. A resident-narrowed plan (tensor parallelism):
    fwd/recompute/bwd tasks additionally gain tp block_params, W/dW/
    A/O objects take their sizes from the per-rank sliced layouts,
    and the optimizer runs in replica-grads mode (no reduce; local
    update; owner broadcast)."""
    hw = ShapedHardware()
    shard_params = None
    tp_params = None
    opt_regions = None
    rank_view = None
    opt_slices = None
    if zero1rs_world is not None:
        from dataflow_training.model_families.families import resolve_family

        dims0, fl0 = resolve_family(cfg).family_layouts(cfg)
        shard_params = zero1rs_block_params(
            layer_fields_by_root(cfg), dims0, zero1rs_world)
        if not shard_params:
            raise ValueError("zero1rs: no root is byte-equal eligible "
                             "(needs uniform adamw + uniform dtypes "
                             "with param==grad)")
        opt_slices = {root: {"n_slice": sh["n_slice"],
                             "n_tail": sh["n_tail"],
                             "opt_dtype": sh["opt_dtype"]}
                      for root, sh in shard_params.items()}
    elif parallel is not None and parallel.plan is not None:
        plan = parallel.plan
        narrowed = any(a.resident != ALL_RANKS for a in plan.assignments)
        if narrowed:
            plan.consumable("tp")
            rank_view = tp_view(plan, parallel.rank)
            tp_params = {
                root: {name: list(sl) for name, sl in slices.items()}
                for root, slices in rank_view.items()}
            shard_params = tp_opt_block_params(plan, parallel.rank)
            opt_regions = {root: dict(sh["update"])
                           for root, sh in shard_params.items()}
        else:
            shard_params = shard_block_params(plan, parallel.rank)
            opt_regions = update_regions(plan, parallel.rank)
    # Three composable passes (family lowering stays parallelism-blind):
    # fam.lower -> annotate_groups -> exact sizes with the rank view.
    # The equivalence gates (test_group_annotation, digest-pinned) prove
    # this path identical to the retired in-builder grouped lowering.
    from dataflow_training.model_families.families import resolve_family

    from .group_annotation import annotate_groups

    fam = resolve_family(cfg)
    program = fam.lower(cfg, recompute_levels=recompute_levels)
    if dp_group is None:
        # world-1: the solo program IS the rank program — no group
        # handles, no shard/tp (validated upstream)
        if shard_params or tp_params:
            raise ValueError("shard/tp params need a group")
        return program
    program = annotate_groups(program, group=dp_group,
                              shard_params=shard_params,
                              tp_params=tp_params)
    if opt_regions is None and opt_slices is None and rank_view is None:
        return program          # plain DP: solo sizes are the rank sizes
    # sharded/narrowed ranks re-size with the rank view. The layout
    # pieces are llama3's until the family object-sizer hook lands
    # with the responsibility map (plan S4); zero1rs/tp reach fleet
    # only via llama3 today (equivalence-certified).
    from dataflow_training.lowering.emit import (
        apply_exact_sizes,
        narrow_layouts,
        object_size_factory,
    )

    dims, fl = fam.family_layouts(cfg)
    if rank_view:
        fl = narrow_layouts(fl, rank_view)
    return apply_exact_sizes(
        program, f"{fam.name}-exact",
        object_size=object_size_factory(dims, fl, opt_update_regions=opt_regions,
                                opt_slice_by_root=opt_slices))


class GroupedBuildVariant:
    """plan_program's recompute rebuilder for dp_group lowerings."""

    def __init__(self, cfg, dp_group: str,
                 parallel=None, zero1rs_world=None):
        self.cfg = cfg
        self.dp_group = dp_group
        self.parallel = parallel
        self.zero1rs_world = zero1rs_world

    def __call__(self, levels):
        return lower_with_group(self.cfg, self.dp_group,
                                recompute_levels=levels,
                                parallel=self.parallel,
                                zero1rs_world=self.zero1rs_world)


