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
