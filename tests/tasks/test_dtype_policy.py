"""DTypePolicy / ParamDTypes: matching semantics and validation (CPU).

The policy layer only; lowering/exec/golden integration is tested in the
per-family ladders once wired (docs/notes/dtype-policy-design.md).
"""
import pytest

from dataflow.tasks.layouts import DTypePolicy, ParamDTypes


def test_default_policy_is_all_bf16():
    p = DTypePolicy()
    for name in ("w_qkvz", "attn_norm_w", "A_log", "anything"):
        dts = p.for_field(name)
        assert (dts.param, dts.grad, dts.opt) == ("bf16", "bf16", "bf16")


def test_first_matching_override_wins_else_default():
    fp32_all = ParamDTypes(param="fp32", grad="fp32", opt="fp32")
    fp32_opt = ParamDTypes(opt="fp32")
    p = DTypePolicy(overrides=(
        ("A_log", fp32_all),
        ("*_norm_w", fp32_opt),
        ("*norm*", ParamDTypes(param="fp32")),  # shadowed for *_norm_w names
    ))
    assert p.for_field("A_log") is fp32_all
    assert p.for_field("attn_norm_w") is fp32_opt          # first match wins
    assert p.for_field("q_norm_w") is fp32_opt
    assert p.for_field("norm_scale").param == "fp32"        # third pattern
    assert p.for_field("w_qkvz") is p.default


def test_mixed_roles_carry_independently():
    dts = ParamDTypes(param="bf16", grad="bf16", opt="fp32")
    assert (dts.param, dts.grad, dts.opt) == ("bf16", "bf16", "fp32")


def test_unknown_dtype_rejected_with_role():
    with pytest.raises(ValueError, match="param"):
        ParamDTypes(param="fp13")
    with pytest.raises(ValueError, match="opt"):
        ParamDTypes(opt="quaternion")


def test_layer_overrides_select_sub_policy():
    fp32_all = ParamDTypes(param="fp32", grad="fp32", opt="fp32")
    deep = DTypePolicy(default=ParamDTypes(opt="fp32"),
                       overrides=(("*_norm_w", fp32_all),))
    p = DTypePolicy(overrides=(("wq", fp32_all),),
                    layer_overrides=(((0, 1), deep), ((1, 2), DTypePolicy())))
    # no layer -> outer policy (loose objects)
    assert p.for_field("wq") is fp32_all
    assert p.for_field("attn_norm_w").opt == "bf16"
    # layer in the first entry -> its sub-policy owns ALL lookups (no
    # fallthrough into the outer overrides)
    assert p.for_field("wq", layer=0).param == "bf16"
    assert p.for_field("wq", layer=0).opt == "fp32"
    assert p.for_field("attn_norm_w", layer=0) is fp32_all
    # first matching entry wins for overlapping layer sets
    assert p.for_field("attn_norm_w", layer=1) is fp32_all
    assert p.for_field("attn_norm_w", layer=2).param == "bf16"
    # unlisted layer -> outer policy
    assert p.for_field("wq", layer=7) is fp32_all
    assert p.depth_dependent and not DTypePolicy().depth_dependent
