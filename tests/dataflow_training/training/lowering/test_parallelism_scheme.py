"""ParallelismScheme contract pins (CPU, pure data): the mesh-form
properties and validate() refusals the conductor and future axis
roles (ep/pp) build against. Notably: responsibility is a DATA-axis
notion — None on solo and pure-tensor schemes (the fallback that once
leaked "zero1rs" into a tensor run's save plan).

Tests:
- test_solo_is_the_empty_mesh: the solo scheme has no axes, world 1, all axis/responsibility views None, and validates.
- test_data_parallel_axis_views: a data-parallel scheme exposes its dp axis and rank rounds, defaults responsibility to zero1rs, and honors a co override.
- test_tensor_scheme_has_no_responsibility: a tensor-parallel scheme exposes its tp axis and plan but carries no responsibility mode.
- test_validate_refusals: validate() rejects mixed data+tensor meshes, bad round sums, wrong world sizes, unsupported responsibility modes, non-FULL tensor plans, and tp size below two.
"""
import pytest

from dataflow_training.distributed.parallelism import (
    Axis,
    ParallelismScheme,
)


class FakePlan:
    world = 2


def test_solo_is_the_empty_mesh():
    s = ParallelismScheme.solo()
    assert s.axes == ()
    assert s.world == 1
    assert s.data_axis is None and s.tensor_axis is None
    assert s.rank_rounds is None
    assert s.responsibility is None
    assert s.tensor_plan is None
    s.validate(world=1, ga_rounds=8)


def test_data_parallel_axis_views():
    s = ParallelismScheme.data_parallel((6, 2))
    assert s.world == 2
    assert s.data_axis.name == "dp"
    assert s.rank_rounds == (6, 2)
    assert s.responsibility == "zero1rs"      # the default mode
    assert s.tensor_plan is None
    s.validate(world=2, ga_rounds=8)
    co = ParallelismScheme.data_parallel((1, 1), responsibility="co")
    assert co.responsibility == "co"


def test_tensor_scheme_has_no_responsibility():
    s = ParallelismScheme.tensor_parallel(FakePlan())
    assert s.world == 2
    assert s.tensor_axis.name == "tp"
    assert s.tensor_plan is not None
    assert s.data_axis is None
    # who-steps is placement itself on a partitioning axis; a mode
    # here would leak zero1rs machinery into a tensor run
    assert s.responsibility is None
    assert s.rank_rounds is None
    s.validate(world=2, ga_rounds=4)


def test_validate_refusals():
    dp = ParallelismScheme.data_parallel((1, 1))
    tp = ParallelismScheme.tensor_parallel(FakePlan())
    composed = ParallelismScheme(axes=dp.axes + tp.axes)
    assert composed.world == 4                # buildable + inspectable
    with pytest.raises(ValueError, match="mesh group machinery"):
        composed.validate(world=4, ga_rounds=2)
    with pytest.raises(ValueError, match="sum"):
        ParallelismScheme.data_parallel((3, 3)).validate(
            world=2, ga_rounds=8)
    with pytest.raises(ValueError, match="world"):
        dp.validate(world=3, ga_rounds=2)
    with pytest.raises(ValueError, match="responsibility"):
        ParallelismScheme(
            axes=(Axis("dp", size=2, rounds=(1, 1),
                       responsibility="zero3"),)
        ).validate(world=2, ga_rounds=2)
    with pytest.raises(ValueError, match="FULL"):
        ParallelismScheme(
            axes=(Axis("tp", size=2, rounds=(1, 1), plan=FakePlan()),)
        ).validate(world=2, ga_rounds=2)
    with pytest.raises(ValueError, match=">= 2"):
        ParallelismScheme.tensor_parallel(FakePlan(), size=1).validate(
            world=1, ga_rounds=2)
