"""glm52 (IndexShare) lowering gates — CPU-only.

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


def test_dense_warmup_and_frozen_indexer_modes():
    from dataclasses import replace as rep

    # dense warm-up is SUPPORTED: no selection anywhere (no dsa_idx in
    # any M — gdl's M drops entirely), dM chains carry FULL-PREFIX rows
    prog = lower_glm52(rep(ShapedGlm52Config.tiny(), sparse_mode=False))
    cfg = ShapedGlm52Config.tiny()
    sizes = prog.object_sizes()
    dms = {k: v for k, v in sizes.items() if k.startswith("dM_")}
    assert dms and all(v == 4 * cfg.tokens * cfg.seq_len for v in dms.values())
    assert not any(oid.startswith("M_") and oid.rsplit("_", 1)[1] == "0"
                   for oid in sizes), \
        "gdl (dense leader, layer 0) must have no M in warm-up"
    # LEAN GRADS: frozen params carry no dW/O. Followers (gmf) have no
    # dW at all; leaders' dW is indexer-only; dW_head/dW_embed and the
    # embed_bwd + frozen optimizer tasks are pruned outright.
    from dataflow.tasks.layouts import dsv32_dense_weight_layout, grad_layout
    from dataflow.training.glm52 import dims_of_glm52

    wdims = dims_of_glm52(rep(ShapedGlm52Config.tiny(), sparse_mode=False))
    idx_dw = grad_layout(dsv32_dense_weight_layout(wdims), wdims.dtypes,
                         layer=0, opt_policy=wdims.opt_policy).total_bytes
    dws = {k: v for k, v in sizes.items() if k.startswith("dW_")}
    follower_layers = {i for i in range(cfg.n_layers)
                       if wdims.kind_of(i) == "gmf"}
    for oid, b in dws.items():
        layer = int(oid.rsplit("_", 1)[1])
        assert layer not in follower_layers, f"follower {oid} should be pruned"
        assert b == idx_dw, (oid, b, idx_dw)
    assert "dW_head" not in sizes and "dW_embed" not in sizes
    task_ids = set(prog.task_by_id())
    # indexer-only objective: no head, no CE, no dy chain — the loss
    # object is the KL accumulator threaded through contributor bwds
    assert not any(t.startswith("head_loss") for t in task_ids)
    assert not any(k.startswith(("targets_", "dy_")) for k in sizes)
    assert sizes.get("loss_0_0") == 4
    assert not any(t.startswith("embed_bwd") for t in task_ids)
    assert not any(t.startswith(("optimizer_embed", "optimizer_head"))
                   for t in task_ids)
    for i in follower_layers:
        assert not any(t.startswith("optimizer_") and t.endswith(f"_{i}")
                       for t in task_ids), f"frozen layer {i} optimizer stays"
    # O objects: leaders idx-only (adamw m+v), followers none
    for i in range(cfg.n_layers):
        if i in follower_layers:
            assert f"O_{i}" not in sizes
        else:
            assert sizes.get(f"O_{i}", 0) > 0
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
