"""DeepSeek-V3 family ladder (GPU): golden + full programs through the
real engine. Block-level MLA/moe pins live in tests/dataflow_training/modules/test_mla.py.

Family-specific pins here: MIXED depth (dense + moe kinds in one chain),
sigmoid_noaux_tc end to end (bias counts through the dW slot, the sign
rule applied by the optimizer's per-field special — golden mirrors it
exactly, so multistep parity IS the bias-rule test), sequence-wise aux,
fixed-seed bitwise determinism.

w_router_bias gate envelope: sign(mean - count) is DISCONTINUOUS — a
+-1-token near-tie selection flip (the known, tolerance-covered class)
that lands a count on the mean boundary flips a whole +-speed bias step
(measured: rel 0.378 = sqrt(1/7), exactly one slot). Same phenomenon as
qwen35moe's dt_bias sign lottery; the honest comparison is the
field_atol envelope |db| <= 2*speed + slack, not a relative bound.

Tests:
- test_dsv3_full_scale_presets_lower_and_validate: the mini, 671b, and kimi_k2 presets lower and validate with the expected MLA block depth.
- test_dsv3_partial_ownership_lowering_rejected: a partial-ownership MoE expert set raises NotImplementedError at lowering.
- test_dsv3_aux_zero_model_step_vs_golden: an aux-off model step matches the golden reference with bias fields held under the atol envelope.
- test_dsv3_ga2_matches_reference: two grad-accum rounds with the LBL composite and noaux bias rule leave engine weights matching the isolated twin.
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings.runtime")  # real CudaBackend + SpinKernel

from dataflow_training.testing.gradcheck import (  # noqa: E402
    check_model_step,
    family_gate_kwargs,
    rel_l2,
    match_field_atol,
)

pytestmark = [pytest.mark.gpu]


def _tiny_cfg(**over):
    from dataflow_training.model_families.dsv3 import ShapedDsv3Config

    return replace(ShapedDsv3Config.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow_training.model_families.dsv3 import derive_dims

    return derive_dims(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------




# --- lowering ----------------------------------------------------------------------


def test_dsv3_full_scale_presets_lower_and_validate():
    from dataflow.core import validate_program
    from dataflow_training.model_families.dsv3 import ShapedDsv3Config, lower_dsv3

    for ctor, layers in ((ShapedDsv3Config.dsv3_mini, 18),
                         (ShapedDsv3Config.dsv3_671b, 61),
                         (ShapedDsv3Config.kimi_k2, 61)):
        cfg = ctor(seq_len=128)
        program = lower_dsv3(cfg)
        validate_program(program)
        n_blocks = sum(1 for t in program.tasks
                       if t.compute_block_key.endswith("_fwd")
                       and t.compute_block_key.startswith("mla"))
        assert n_blocks == layers


def test_dsv3_partial_ownership_lowering_rejected():
    import dataclasses
    import unittest.mock as mock

    from dataflow_training.model_families.dsv3 import derive_dims, lower_dsv3

    cfg = _tiny_cfg()
    part = dataclasses.replace(derive_dims(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow_training.model_families.dsv3.model.derive_moe_spec", return_value=part):
            lower_dsv3(cfg)


# --- full program through the real engine ------------------------------------------


_BIAS_ATOL = {"router_bias": 2.5e-3}   # 2.5x speed: +-2 steps + fp
# slack. KEY IS THE TWIN BUFFER NAME ("router_bias"; dsv32/glm52 use
# "w_router_bias") — a w_-prefixed key never suffix-matches here and
# silently disables the envelope (bitten three times, see ledger)




def test_dsv3_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        field_atol=_BIAS_ATOL, **family_gate_kwargs("dsv3"),
    ).assert_ok()


def test_dsv3_ga2_matches_reference():
    """Two grad-accum rounds with the LBL composite + noaux bias rule:
    engine == the isolated twin (whose per-step aggregate-count bias
    semantics were certified against the engine at 2B scale)."""
    from dataflow_training.model_families import bridges
    from dataflow_training.run.driver import adamw_field_step
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow_training.blocks.base_blocks import AdamWHyper
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import (
        EngineFinalBytes,
        rel_l2,
    )

    cfg = _tiny_cfg(grad_accum_rounds=2)
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=16 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=3)

    model = bridges.build_reference_model(cfg)
    bridges.load_reference_init(model, cfg, dims,
                                bridges.get_bytes_from_values(values))
    model.train()
    B = dims.max_tokens // cfg.seq_len
    total = None
    for r in range(cfg.grad_accum_rounds):
        toks = torch_view(values[f"tokens_0_{r}"], (dims.max_tokens,),
                          torch.int32).long().cuda().view(B, cfg.seq_len)
        tgts = torch_view(values[f"targets_0_{r}"], (dims.max_tokens,),
                          torch.int32).long().cuda().view(B, cfg.seq_len)
        loss_r = model.loss(toks, tgts, aux_coef=cfg.aux_coef)
        total = loss_r if total is None else total + loss_r
    total.backward()
    hp = AdamWHyper()
    for par in model.parameters():
        if par.grad is None:
            continue
        m = torch.zeros_like(par)
        v = torch.zeros_like(par)
        adamw_field_step(par.data, par.grad, m, v, lr=hp.lr,
                         beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                         weight_decay=hp.weight_decay, step=1)
    for module in model.modules():
        if hasattr(module, "apply_bias_update"):
            module.apply_bias_update(cfg.bias_update_speed)

    from dataflow_training.data.segments import uniform_segments

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)},
    )
    engine_state = bridges.to_reference_state_dict(
        cfg, EngineFinalBytes(result))
    twin_state = dict(model.state_dict())
    for name, engine_tensor in engine_state.items():
        atol = match_field_atol(name, _BIAS_ATOL)
        if atol is not None:
            gap = float((engine_tensor.float().cpu()
                         - twin_state[name].float().cpu()).abs().max())
            assert gap <= atol, (name, gap)
            continue
        err = rel_l2(engine_tensor, twin_state[name])
        assert err < 3e-2, (name, err)
    result.close()


# --- engine gates -------------------------------------------------------------------


def _run(engine_kwargs=None, program=None, seed=7, resolver_wrapper=None):
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    prog = program if program is not None else plan_program(
        fam.lower(cfg), fast_memory_capacity=8 * 1024 * 1024,
    ).program

    backend = CudaBackend()
    values = fam.initial_values(prog, cfg, backend, seed=seed)
    from dataflow_training.data.segments import uniform_segments

    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    resolver = fam.build_resolver(fam.derive_dims(cfg))
    if resolver_wrapper is not None:
        resolver = resolver_wrapper(resolver, backend)
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver,
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(fam.derive_dims(cfg), prog)},
    )
    out = {}
    for obj_id in ["W_embed", "W_head"] + [f"W_{i}" for i in range(cfg.n_layers)]:
        rec = result.objects.get(obj_id)
        slot = rec.backing or rec.fast
        # BYTES, not bf16: fp32 fields (w_router_bias) reinterpreted as
        # bf16 can alias NaN bit patterns and torch.equal fails NaN != NaN
        # on identical bytes (this file passed by bit-pattern luck until
        # the dsv32 harness fix; the luck ran out 2026-07-07)
        out[obj_id] = torch_view(slot.buffer, (rec.size_bytes,), torch.uint8).clone()
    loss_rec = result.objects.get("loss_0_0")
    out["loss"] = float(torch_view((loss_rec.backing or loss_rec.fast).buffer, (1,), torch.float32)[0])
    result.close()
    dry.close()
    for buf in values.values():
        backend.free(buf)
    return out


def _assert_same(a: dict, b: dict, tol: float = 1e-3):
    assert abs(a["loss"] - b["loss"]) / max(abs(b["loss"]), 1e-9) < tol, (a["loss"], b["loss"])
    for k in a:
        if k == "loss":
            continue
        if torch.equal(a[k], b[k]):        # byte-identical fast path
            continue
        va = torch.nan_to_num(a[k].view(torch.bfloat16).float())
        vb = torch.nan_to_num(b[k].view(torch.bfloat16).float())
        err = rel_l2(va, vb)
        assert err < tol, f"{k}: rel_l2={err}"

