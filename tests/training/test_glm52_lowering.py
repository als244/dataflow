"""glm52 (IndexShare) lowering gates — CPU-only (M-I1).

Pins the M/dM metadata grammar: every layer's discrete artifacts live in
its M object (never recomputed — Shein invariant); IndexShare followers
additionally consume the PRODUCER layer's M (the shared dsa selection);
the KL-target accumulator dM follows the reverse-order
create/mutate/consume chain; singleton groups have no dM."""
import pytest

from dataflow.core import validate_program
from dataflow.training.glm52 import ShapedGlm52Config, lower_glm52


def test_tiny_lowers_with_correct_sp_grammar():
    cfg = ShapedGlm52Config.tiny()   # roles F F S S F S
    p = lower_glm52(cfg)
    validate_program(p)
    fwd = {t.id: t for t in p.tasks if t.id.startswith("block_fwd_0_0_")}
    bwd = {t.id: t for t in p.tasks if t.id.startswith("block_bwd_0_0_")}
    # every layer emits its own M; followers ALSO consume the producer's
    assert any(o.id == "M_0_0_1" for o in fwd["block_fwd_0_0_1"].outputs)
    assert "M_0_0_1" in fwd["block_fwd_0_0_2"].inputs
    assert "M_0_0_1" in fwd["block_fwd_0_0_3"].inputs
    assert "M_0_0_4" in fwd["block_fwd_0_0_5"].inputs
    # dM chain: last member creates, middles mutate, producer consumes
    assert any(o.id == "dM_0_0_1" for o in bwd["block_bwd_0_0_3"].outputs)
    assert "dM_0_0_1" in bwd["block_bwd_0_0_2"].mutates
    assert "dM_0_0_1" in bwd["block_bwd_0_0_1"].inputs
    assert "dM_0_0_1" not in bwd["block_bwd_0_0_1"].mutates
    # singleton leader (layer 0): no dM anywhere
    assert not any(
        "dM_0_0_0" in t.inputs or any(o.id == "dM_0_0_0" for o in t.outputs)
        for t in p.tasks
    )
    # every group member's bwd consumes the producer's M
    for i, g in ((0, 0), (1, 1), (2, 1), (3, 1), (4, 4), (5, 4)):
        assert f"M_0_0_{g}" in bwd[f"block_bwd_0_0_{i}"].inputs


def test_recompute_never_reselects():
    cfg = ShapedGlm52Config.tiny()
    p = lower_glm52(cfg, recompute_levels={"A_0_0_2": 1, "A_0_0_1": 1})
    validate_program(p)
    rc = {t.id: t for t in p.tasks if t.id.startswith("block_recompute_")}
    assert "M_0_0_1" in rc["block_recompute_0_0_2"].inputs   # follower rc
    assert "M_0_0_1" in rc["block_recompute_0_0_1"].inputs   # producer rc too


def test_full_scale_presets_lower():
    for ctor, layers, leaders in ((ShapedGlm52Config.glm52, 78, 21),
                                  (ShapedGlm52Config.glm52_mini, 18, 6)):
        p = lower_glm52(ctor(seq_len=128))
        validate_program(p)
        # dM objects exist exactly for multi-member groups
        dm = {o.id for t in p.tasks for o in t.outputs if o.id.startswith("dM_")}
        cfg = ctor(seq_len=128)
        from dataflow.training.glm52 import dims_of_glm52
        dims = dims_of_glm52(cfg)
        multi = [ld for ld in dims.leaders() if len(dims.group_members(ld)) > 1]
        assert len(dm) == len(multi)
        # follower fwds consume their producer's M
        for ld in multi:
            for f in dims.group_members(ld)[1:]:
                task = next(t for t in p.tasks if t.id == f"block_fwd_0_0_{f}")
                assert f"M_0_0_{ld}" in task.inputs
        n_fwd = sum(1 for t in p.tasks if t.compute_block_key.endswith("_fwd")
                    and t.compute_block_key.startswith("g"))
        assert n_fwd == layers


def test_dense_warmup_rejected_frozen_indexer_supported():
    from dataclasses import replace as rep

    with pytest.raises(NotImplementedError):
        lower_glm52(rep(ShapedGlm52Config.tiny(), sparse_mode=False))
    # frozen indexer is a SUPPORTED mode (RL post-training consumes
    # saved selections verbatim): lowers cleanly, emits no dM chain
    prog = lower_glm52(rep(ShapedGlm52Config.tiny(), train_indexer=False))
    assert not [o for o in prog.initial_objects if o.id.startswith("dM_")]


def test_bad_patterns_rejected():
    from dataclasses import replace as rep

    with pytest.raises(ValueError):
        lower_glm52(rep(ShapedGlm52Config.tiny(),
                        indexer_types=("shared",) * 6))
    with pytest.raises(ValueError):
        lower_glm52(rep(ShapedGlm52Config.tiny(),
                        indexer_types=("full",) * 5))
