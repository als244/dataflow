"""glm52 (IndexShare) lowering gates — CPU-only (M-I1).

Pins the S/P cross-layer grammar: leaders emit the selection object S,
followers/recomputes/bwds consume it (SELECTION IS NEVER RECOMPUTED —
Shein invariant), the KL-target accumulator P follows the reverse-order
create/mutate/consume chain, singleton groups have no P."""
import pytest

from dataflow.core import validate_program
from dataflow.training.glm52 import ShapedGlm52Config, lower_glm52


def test_tiny_lowers_with_correct_sp_grammar():
    cfg = ShapedGlm52Config.tiny()   # roles F F S S F S
    p = lower_glm52(cfg)
    validate_program(p)
    fwd = {t.id: t for t in p.tasks if t.id.startswith("block_fwd_0_0_")}
    bwd = {t.id: t for t in p.tasks if t.id.startswith("block_bwd_0_0_")}
    assert any(o.id == "S_0_0_1" for o in fwd["block_fwd_0_0_1"].outputs)
    assert "S_0_0_1" in fwd["block_fwd_0_0_2"].inputs
    assert "S_0_0_1" in fwd["block_fwd_0_0_3"].inputs
    assert "S_0_0_4" in fwd["block_fwd_0_0_5"].inputs
    # P chain: last member creates, middles mutate, leader consumes
    assert any(o.id == "P_0_0_1" for o in bwd["block_bwd_0_0_3"].outputs)
    assert "P_0_0_1" in bwd["block_bwd_0_0_2"].mutates
    assert "P_0_0_1" in bwd["block_bwd_0_0_1"].inputs
    assert "P_0_0_1" not in bwd["block_bwd_0_0_1"].mutates
    # singleton leader (layer 0): no P anywhere
    assert not any(
        "P_0_0_0" in t.inputs or any(o.id == "P_0_0_0" for o in t.outputs)
        for t in p.tasks
    )
    # every member bwd consumes its group's S
    for i, g in ((0, 0), (1, 1), (2, 1), (3, 1), (4, 4), (5, 4)):
        assert f"S_0_0_{g}" in bwd[f"block_bwd_0_0_{i}"].inputs


def test_recompute_never_reselects():
    cfg = ShapedGlm52Config.tiny()
    p = lower_glm52(cfg, recompute_levels={"A_0_0_2": 1, "A_0_0_1": 1})
    validate_program(p)
    rc = {t.id: t for t in p.tasks if t.id.startswith("block_recompute_")}
    assert "S_0_0_1" in rc["block_recompute_0_0_2"].inputs   # follower rc
    assert "S_0_0_1" in rc["block_recompute_0_0_1"].inputs   # leader rc too


def test_full_scale_presets_lower():
    for ctor, layers, leaders in ((ShapedGlm52Config.glm52, 78, 21),
                                  (ShapedGlm52Config.glm52_mini, 18, 6)):
        p = lower_glm52(ctor(seq_len=128))
        validate_program(p)
        s_objs = {o.id for t in p.tasks for o in t.outputs if o.id.startswith("S_")}
        assert len(s_objs) == leaders
        n_fwd = sum(1 for t in p.tasks if t.compute_block_key.endswith("_fwd")
                    and t.compute_block_key.startswith("g"))
        assert n_fwd == layers


def test_dense_warmup_and_frozen_indexer_rejected_loudly():
    from dataclasses import replace as rep

    with pytest.raises(NotImplementedError):
        lower_glm52(rep(ShapedGlm52Config.tiny(), sparse_mode=False))
    with pytest.raises(NotImplementedError):
        lower_glm52(rep(ShapedGlm52Config.tiny(), train_indexer=False))


def test_bad_patterns_rejected():
    from dataclasses import replace as rep

    with pytest.raises(ValueError):
        lower_glm52(rep(ShapedGlm52Config.tiny(),
                        indexer_types=("shared",) * 6))
    with pytest.raises(ValueError):
        lower_glm52(rep(ShapedGlm52Config.tiny(),
                        indexer_types=("full",) * 5))
