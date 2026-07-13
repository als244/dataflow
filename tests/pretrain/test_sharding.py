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


def test_world_n_plans():
    """WN0: the builders and block-param derivations at world 4 and
    8 — balance, cover, divisibility, and the cross-rank comm-
    sequence identity that collective pairing depends on."""
    from dataflow.pretrain.presets import preset
    from dataflow.pretrain.sharding import (
        layer_fields_by_root,
        shard_block_params,
        tp_mlp_shards,
        tp_view,
        zero1_halves,
    )

    fbr = layer_fields_by_root(preset("l3_125m"))
    for world in (4, 8):
        plan = zero1_halves(fbr, "dp", world)
        plan.validate()
        plan.v1_consumable()
        owned = []
        for root in plan.roots():
            per = [sum(hi - lo for lo, hi in plan.owned_ranges(r, root))
                   for r in range(world)]
            owned.append(per)
        agg = [sum(per[r] for per in owned) for r in range(world)]
        # field-snapped whole-field shards lose balance as world
        # approaches the per-root field count (7 big fields into 8
        # buckets); the byte-equal rs/ag builder is the world-8
        # answer — this gates the honest v1 floor, not aspiration
        floor = {4: 0.4, 8: 0.2}[world]
        assert min(agg) / max(agg) > floor, (world, agg)
        params = [shard_block_params(plan, r) for r in range(world)]
        for root in params[0]:
            comms = [p[root]["comm"] for p in params]
            assert all(c == comms[0] for c in comms), (world, root)
            assert all(e["owner"] in range(world)
                       for e in comms[0]), (world, root)

    # tp at world 4 (d_ff 3072 divides); world 8 too (3072 % 8 == 0)
    for world in (4, 8):
        plan = tp_mlp_shards(fbr, "tp", world)
        plan.validate()
        plan.consumable("tp")
        assert not plan.redundant()
        views = [tp_view(plan, r) for r in range(world)]
        w1 = [v["W_0"]["w1"] for v in views]
        assert len({sl[1:] for sl in w1}) == world      # distinct slices
        assert all(sl[0] == 1 for sl in w1)             # column axis


def test_zero1rs_block_params_sizing():
    """Byte-equal builder: every root of the real tiny-llama3 layouts
    is eligible under the default policy; slices are exact world-
    fractions of the packed element count (alignment gaps included);
    the flat opt-state layout sizes to slice+tail elements per slot."""
    from dataflow.pretrain.sharding import (
        layer_fields_by_root,
        zero1rs_block_params,
    )
    from dataflow.tasks.layouts import opt_state_slice_layout
    from dataflow.training.models.llama3 import (
        ShapedLlamaConfig,
        dims_of,
    )

    cfg = ShapedLlamaConfig.tiny()
    dims = dims_of(cfg)
    fbr = layer_fields_by_root(cfg)
    totals = {}
    for world in (2, 3, 4, 8):
        params = zero1rs_block_params(fbr, dims, world)
        assert set(params) == set(fbr), (world, sorted(params))
        for root, sh in params.items():
            assert sh["mode"] == "rs" and sh["grads"] == "partial"
            assert sh["dtype"] == "bf16" and sh["opt_dtype"] == "bf16"
            total = sh["n_slice"] * world + sh["n_tail"]
            assert totals.setdefault(root, total) == total, (
                world, root)                    # world-invariant total
            assert 0 <= sh["n_tail"] < world
            if world in (2, 4, 8):
                # 256B-aligned packing => elements divisible by any
                # power-of-2 world we target; tails are a 3/5/6/7-way
                # concern
                assert sh["n_tail"] == 0, (world, root, sh)
            ol = opt_state_slice_layout(sh["n_slice"], sh["n_tail"],
                                        sh["opt_dtype"])
            names = [f.name for f in ol.fields]
            want = (["m_slice", "v_slice", "m_tail", "v_tail"]
                    if sh["n_tail"] else ["m_slice", "v_slice"])
            assert names == want, (world, root, names)
            per = {f.name: f.nbytes for f in ol.fields}
            assert per["m_slice"] == sh["n_slice"] * 2  # bf16 moments
            assert per["m_slice"] == per["v_slice"]


def test_zero1rs_eligibility_rejections():
    """Roots drop out (fall back to the field-snapped path) whenever
    ONE flat hyper/rule/dtype story doesn't hold: a layer routed to
    the muon recipe, a hyper override touching any field (namespaced
    keys for embed/head), or non-uniform dtypes inside the root."""
    from dataclasses import replace

    from dataflow.pretrain.sharding import (
        layer_fields_by_root,
        zero1rs_block_params,
    )
    from dataflow.tasks.layouts import DTypePolicy, ParamDTypes
    from dataflow.tasks.optim import OptPolicy
    from dataflow.training.models.llama3 import (
        ShapedLlamaConfig,
        dims_of,
    )

    cfg = ShapedLlamaConfig.tiny()
    dims = dims_of(cfg)
    fbr = layer_fields_by_root(cfg)
    n_layers = cfg.n_layers
    all_roots = set(fbr)

    # layer 0 routed to the muon recipe: W_0 mixes muon+adamw -> out;
    # other layers and the loose tables keep the flat path
    d = replace(dims, opt_policy=OptPolicy(
        default="adamw", layer_overrides=(((0,), "muon"),)))
    params = zero1rs_block_params(fbr, d, 2)
    assert "W_0" not in params
    assert set(params) == all_roots - {"W_0"}

    # per-field hyper override: every root containing a match drops
    d = replace(dims, opt_policy=OptPolicy(
        default="adamw",
        hyper_overrides=(("*_norm_w", {"weight_decay": 0.0}),)))
    params = zero1rs_block_params(fbr, d, 2)
    blocks = {f"W_{i}" for i in range(n_layers)}
    assert not (set(params) & blocks)           # every block has norms
    assert "W_head" not in params               # head.final_norm_w
    assert "W_embed" in params                  # embed.w only

    # ...and the override key is NAMESPACED for the loose tables
    d = replace(dims, opt_policy=OptPolicy(
        default="adamw", hyper_overrides=(("embed.*", {"lr": 1e-5}),)))
    params = zero1rs_block_params(fbr, d, 2)
    assert "W_embed" not in params
    assert set(params) == all_roots - {"W_embed"}

    # non-uniform dtypes inside a root (wq fp32 island) -> out; a
    # UNIFORM fp32 opt dtype is fine (still one flat story)
    d = replace(dims, dtypes=DTypePolicy(
        overrides=(("wq", ParamDTypes("fp32", "fp32", "fp32")),)))
    params = zero1rs_block_params(fbr, d, 2)
    assert not (set(params) & blocks)
    assert {"W_embed", "W_head"} <= set(params)
    d = replace(dims, dtypes=DTypePolicy(
        default=ParamDTypes("bf16", "bf16", "fp32")))
    params = zero1rs_block_params(fbr, d, 2)
    assert set(params) == all_roots
    assert all(sh["opt_dtype"] == "fp32" for sh in params.values())

    # param != grad dtype: flat W/dW byte-coincidence breaks -> out
    d = replace(dims, dtypes=DTypePolicy(
        default=ParamDTypes("bf16", "fp32", "fp32")))
    assert zero1rs_block_params(fbr, d, 2) == {}
