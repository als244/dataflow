"""Mixed dtype-policy E2E: real engine vs golden under non-default dtypes.

The acceptance gates for the dtype-policy-design design note: a policy with
fp32 norm weights (param+grad+opt), fp32 moments everywhere, and — for
qwen3.5 — fp32 A_log/dt_bias, must train through the REAL engine and match
the golden model exactly as the bf16 ladders do. Exercises: mixed-dtype
packed layouts (alignment gaps), grad_layout-typed dW accumulation,
per-field AdamW at mixed storage dtypes, and dtype-true golden updates.
"""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.tasks.layouts import DTypePolicy, ParamDTypes  # noqa: E402
from dataflow.training.llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow.training.testing.gradcheck import check_block_backward, check_model_step  # noqa: E402

pytestmark = pytest.mark.gpu

FP32_ALL = ParamDTypes(param="fp32", grad="fp32", opt="fp32")
MIXED = DTypePolicy(
    default=ParamDTypes(param="bf16", grad="bf16", opt="fp32"),  # fp32 moments
    overrides=(
        ("*_norm_w", FP32_ALL),  # block norms + head.final_norm_w
    ),
)


def _llama_cfg(**over):
    kw = dict(
        n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
        vocab_size=512, seq_len=128, batch=1, dtypes=MIXED,
    )
    kw.update(over)
    return ShapedLlamaConfig(**kw)


def test_mixed_policy_layout_shapes():
    from dataflow.training.llama3 import dims_of
    from dataflow.tasks.layouts import grad_layout, opt_state_layout, weight_layout

    dims = dims_of(_llama_cfg())
    wl = weight_layout(dims)
    assert wl.field("attn_norm_w").dtype == "fp32"
    assert wl.field("wq").dtype == "bf16"
    gl = grad_layout(wl, dims.dtypes)
    assert gl.field("attn_norm_w").dtype == "fp32"
    assert gl.field("wq").dtype == "bf16"
    ol = opt_state_layout(wl, dims.dtypes)
    assert ol.field("m_wq").dtype == "fp32"          # default opt fp32
    assert ol.field("v_attn_norm_w").dtype == "fp32"
    # fp32 norm weights + fp32 moments grow the packed objects
    assert gl.total_bytes > 0 and ol.total_bytes > 2 * wl.total_bytes // 2


def test_llama_block_ladder2_mixed_policy():
    from dataflow.training.llama3 import dims_of

    check_block_backward(dims_of(_llama_cfg())).assert_ok()


def test_llama_model_step_mixed_policy():
    check_model_step(_llama_cfg(), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()


def test_qwen35_model_step_mixed_policy():
    from dataclasses import replace

    from dataflow.training.qwen35 import ShapedQwen35Config

    policy = DTypePolicy(
        default=ParamDTypes(opt="fp32"),
        overrides=(
            ("A_log", FP32_ALL),
            ("dt_bias", FP32_ALL),
            ("*_norm_w", FP32_ALL),
        ),
    )
    cfg = replace(ShapedQwen35Config.tiny(), dtypes=policy)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()


DEPTH = DTypePolicy(
    layer_overrides=(
        # layer 0 trains "hot": fp32 params/grads/moments on the norms,
        # fp32 moments on everything
        ((0,), DTypePolicy(default=ParamDTypes(opt="fp32"),
                           overrides=(("*_norm_w", FP32_ALL),))),
    ),
)


def test_depth_dependent_layer_sizes_diverge():
    from dataflow.training.llama3 import lower_llama3

    prog = lower_llama3(_llama_cfg(dtypes=DEPTH))
    sizes = {o.id: o.size_bytes for o in prog.initial_objects}
    for t in prog.tasks:
        sizes.update({o.id: o.size_bytes for o in t.outputs})
    assert sizes["O_0"] > sizes["O_1"]          # fp32 moments on layer 0
    assert sizes["W_0"] > sizes["W_1"]          # fp32 norm weights on layer 0
    assert sizes["dW_0_0"] > sizes["dW_0_1"]    # fp32 norm grads on layer 0


def test_llama_model_step_depth_dependent():
    check_model_step(
        _llama_cfg(dtypes=DEPTH), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
    ).assert_ok()


def test_qwen35_model_step_depth_dependent():
    from dataclasses import replace

    from dataflow.training.qwen35 import ShapedQwen35Config

    policy = DTypePolicy(
        default=ParamDTypes(),
        layer_overrides=(
            # one lin layer (0) and THE attn layer (3) of the LLLF tiny get
            # fp32 elementwise params + fp32 moments; layers 1-2 stay bf16
            ((0, 3), DTypePolicy(default=ParamDTypes(opt="fp32"), overrides=(
                ("*_norm_w", FP32_ALL),
                ("A_log", FP32_ALL),
                ("dt_bias", FP32_ALL),
            ))),
        ),
    )
    cfg = replace(ShapedQwen35Config.tiny(), dtypes=policy)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2).assert_ok()
