"""Z0 gates: the sharding API — plan invariants, realizability
classification, the muon rejection, the expert-shards builder's
owned/redundant views, and serialization. Pure CPU."""
import pytest

from dataflow.pretrain.sharding import (
    ALL_RANKS,
    Assignment,
    FieldInfo,
    ParallelConfig,
    Region,
    ShardPlan,
    expert_shards,
    zero1_halves,
)


def toy_fields():
    # mimics a packed layout: 2 tiny norms + 3 big matrices
    return {"W_0": [
        FieldInfo("attn_norm_w", (64,), "bf16", 0, 128),
        FieldInfo("wq", (64, 64), "bf16", 128, 8192),
        FieldInfo("w1", (64, 128), "bf16", 8320, 16384),
        FieldInfo("w2", (128, 64), "bf16", 24704, 16384),
        FieldInfo("ffn_norm_w", (64,), "bf16", 41088, 128),
    ]}


def test_zero1_halves_invariants_and_views():
    plan = zero1_halves(toy_fields(), "dp", 2,
                        replicate_below_bytes=256)
    plan.validate()
    plan.v1_consumable()
    owned0 = {r.field for r in plan.owned(0)}
    owned1 = {r.field for r in plan.owned(1)}
    assert owned0 and owned1
    assert not owned0 & owned1
    assert {r.field for r in plan.redundant()} == {"attn_norm_w",
                                                   "ffn_norm_w"}
    b0 = sum(hi - lo for lo, hi in plan.owned_ranges(0, "W_0"))
    b1 = sum(hi - lo for lo, hi in plan.owned_ranges(1, "W_0"))
    assert abs(b0 - b1) <= 16384          # within one field


def test_equal_shards_honest():
    plan = zero1_halves(toy_fields(), "dp", 2,
                        replicate_below_bytes=256)
    # field-snapped halves here are unequal => rs/ag fast path is
    # honestly reported unavailable
    assert plan.equal_shards("W_0") is False
    # a hand-built equal split of one 2-field root IS realizable
    fbr = {"W_0": [FieldInfo("a", (4, 4), "bf16", 0, 32),
                   FieldInfo("b", (4, 4), "bf16", 32, 32)]}
    plan2 = ShardPlan("dp", 2, (
        Assignment(Region("W_0", "a"), 0),
        Assignment(Region("W_0", "b"), 1)), fbr)
    plan2.validate()
    assert plan2.equal_shards("W_0") is True


def test_row_split_and_cover_validation():
    fbr = {"W_0": [FieldInfo("m", (8, 4), "bf16", 0, 64)]}
    good = ShardPlan("dp", 2, (
        Assignment(Region("W_0", "m", rows=(0, 4)), 0),
        Assignment(Region("W_0", "m", rows=(4, 8)), 1)), fbr)
    good.validate()
    assert good.equal_shards("W_0") is True
    with pytest.raises(ValueError, match="gap"):
        ShardPlan("dp", 2, (
            Assignment(Region("W_0", "m", rows=(0, 3)), 0),
            Assignment(Region("W_0", "m", rows=(4, 8)), 1)),
            fbr).validate()
    with pytest.raises(ValueError, match="out of bounds"):
        ShardPlan("dp", 2, (
            Assignment(Region("W_0", "m", rows=(0, 9)), 0),),
            fbr).validate()


def test_muon_rejects_row_splits():
    fbr = {"W_0": [FieldInfo("m", (8, 4), "bf16", 0, 64)]}
    plan = ShardPlan("dp", 2, (
        Assignment(Region("W_0", "m", rows=(0, 4)), 0),
        Assignment(Region("W_0", "m", rows=(4, 8)), 1)), fbr)
    with pytest.raises(ValueError, match="muon"):
        plan.validate(opt_policy="muon")
    # whole-field muon assignment is fine
    whole = ShardPlan("dp", 2, (
        Assignment(Region("W_0", "m"), 0),), fbr)
    whole.validate(opt_policy="muon")


def test_expert_shards_views_and_ep_rejection():
    fbr = {"W_3": [FieldInfo("router_w", (8, 4), "bf16", 0, 64)]
           + [FieldInfo(f"expert{e}_w1", (16, 16), "bf16",
                        64 + e * 512, 512) for e in range(4)]}
    def expert_of(name):
        if name.startswith("expert"):
            return int(name[len("expert"):].split("_")[0])
        return None
    plan = expert_shards(fbr, "dp", 2, expert_field_of=expert_of,
                         replicated_fields=("router_w",))
    plan.validate()
    plan.v1_consumable()
    assert {r.field for r in plan.owned(0)} == {"expert0_w1",
                                                "expert2_w1"}
    assert {r.field for r in plan.owned(1)} == {"expert1_w1",
                                                "expert3_w1"}
    assert {r.field for r in plan.redundant()} == {"router_w"}
    # true-EP residency narrowing is REJECTED by the v1 consumer
    ep = ShardPlan("dp", 2, (
        Assignment(Region("W_3", "expert0_w1"), 0, resident=0),),
        fbr)
    with pytest.raises(ValueError, match="residency"):
        ep.v1_consumable()


def tp_toy_fields():
    # llama3-ish layer: attention + norms replicated, mlp sharded
    return {"W_0": [
        FieldInfo("attn_norm_w", (64,), "bf16", 0, 128),
        FieldInfo("wq", (64, 64), "bf16", 128, 8192),
        FieldInfo("w1", (64, 128), "bf16", 8320, 16384),
        FieldInfo("w3", (64, 128), "bf16", 24704, 16384),
        FieldInfo("w2", (128, 64), "bf16", 41088, 16384),
    ]}


def test_tp_mlp_shards_plan_and_views():
    from dataflow.pretrain.sharding import (
        tp_mlp_shards,
        tp_view,
        update_regions,
    )

    plan = tp_mlp_shards(tp_toy_fields(), "tp", 2)
    plan.validate()
    plan.consumable("tp")
    # the optimizer (zero1) consumer refuses narrowed residency
    with pytest.raises(ValueError, match="residency"):
        plan.consumable("optimizer")
    with pytest.raises(ValueError, match="residency"):
        plan.v1_consumable()
    # per-rank layout transforms: w1/w3 column halves, w2 row halves
    v0, v1 = tp_view(plan, 0), tp_view(plan, 1)
    assert v0["W_0"]["w1"] == (1, 0, 64)
    assert v1["W_0"]["w1"] == (1, 64, 128)
    assert v0["W_0"]["w2"] == (0, 0, 64)
    assert v1["W_0"]["w2"] == (0, 64, 128)
    # replicated fields ride the standard optimizer configuration —
    # and under tp EVERY one has an owner: redundant (ALL_RANKS)
    # updates from replica grads drift across an arch boundary with
    # nothing to re-pin them (the 1B norm-drift incident)
    assert plan.field_owner("W_0", "wq") in (0, 1)
    assert plan.field_owner("W_0", "attn_norm_w") in (0, 1)
    assert not plan.redundant(), [r.field for r in plan.redundant()]
    # the zero1 O-sizing map refuses tp plans (per-rank layouts own it)
    with pytest.raises(ValueError, match="tp_view"):
        update_regions(plan, 0)
    # byte-range machinery refuses column shards
    with pytest.raises(ValueError, match="dim-0 only"):
        plan.owned_ranges(0, "W_0")


def test_tp_axis_validation():
    from dataflow.pretrain.sharding import tp_mlp_shards

    fbr = {"W_0": [FieldInfo("m", (8, 4), "bf16", 0, 64)]}
    with pytest.raises(ValueError, match="mixed shard axes"):
        ShardPlan("tp", 2, (
            Assignment(Region("W_0", "m", rows=(0, 4), dim=0), 0),
            Assignment(Region("W_0", "m", rows=(0, 2), dim=1), 1),
        ), fbr).validate()
    with pytest.raises(ValueError, match="gap"):
        ShardPlan("tp", 2, (
            Assignment(Region("W_0", "m", rows=(0, 1), dim=1), 0,
                       resident=0),
            Assignment(Region("W_0", "m", rows=(2, 4), dim=1), 1,
                       resident=1),
        ), fbr).validate()
    with pytest.raises(ValueError, match="does not divide"):
        tp_mlp_shards({"W_0": [FieldInfo("w1", (8, 6), "bf16", 0, 96)]},
                      "tp", 4)


def test_tp_serialization_roundtrip():
    from dataflow.pretrain.sharding import tp_mlp_shards

    fbr = tp_toy_fields()
    plan = tp_mlp_shards(fbr, "tp", 2)
    back = ShardPlan.from_dict(plan.to_dict(), fbr)
    assert [a.region.key() for a in back.assignments] == \
           [a.region.key() for a in plan.assignments]
    assert [a.resident for a in back.assignments] == \
           [a.resident for a in plan.assignments]
    back.consumable("tp")


def test_serialization_roundtrip_and_required_groups():
    fbr = toy_fields()
    plan = zero1_halves(fbr, "dp", 2,
                        replicate_below_bytes=256)
    back = ShardPlan.from_dict(plan.to_dict(), fbr)
    assert back.group == "dp" and back.world == 2
    assert [a.region.key() for a in back.assignments] == \
           [a.region.key() for a in plan.assignments]
    assert plan.required_groups() == [{"name": "dp", "purpose": "root"}]
    cfg = ParallelConfig(group="dp", rank=1, world=2, plan=plan)
    assert cfg.plan is plan


def test_real_llama3_layouts_shard():
    torch = pytest.importorskip("torch")
    from dataflow.pretrain.presets import preset
    from dataflow.pretrain.sharding import layer_fields_by_root

    fbr = layer_fields_by_root(preset("l3_125m"))
    plan = zero1_halves(fbr, "dp", 2)
    plan.validate(opt_policy=None)
    plan.v1_consumable()
    agg = [0, 0]
    for root in plan.roots():
        o0 = sum(hi - lo for lo, hi in plan.owned_ranges(0, root))
        o1 = sum(hi - lo for lo, hi in plan.owned_ranges(1, root))
        total = sum(f.nbytes for f in fbr[root])
        assert o0 + o1 <= total
        if o0 + o1 > 0:
            # within one field of even (the 63/37 greedy-overshoot
            # regression tripwire)
            assert min(o0, o1) / max(o0, o1) > 0.6, (root, o0, o1)
        agg[0] += o0
        agg[1] += o1
    assert min(agg) / max(agg) > 0.85, agg
