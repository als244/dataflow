"""OLMoE correctness ladder (GPU): the first family on the pluggable MoE
module, mirrored on tests/dataflow_training/models/test_qwen35.py.

Family-specific pins: full-row qk-norm (one rstd per token), the MoE tail
spliced after resid1_norm2, aux load-balance gradient injection (the
golden's autograd objective is CE + aux while its reported loss is CE),
recompute reproducing the ROUTING DECISION bit-exactly (int ctx fields
compared with torch.equal), and end-to-end engine determinism (fixed seed
twice -> identical bytes).

Tests:
- test_olmoe_stage_context_completeness: the forward block's emitted context fields exactly equal the activation layout and only the y-only combine epilogue sits past the recompute boundary.
- test_olmoe_partial_ownership_lowering_rejected: lowering with a partial expert-ownership MoE spec raises NotImplementedError.
- test_olmoe_aux_zero_model_step_vs_golden: an aux_coef=0 model-step matches the golden twin.
- test_olmoe_grad_accum_two_rounds_matches_reference: two grad-accum rounds (per-round CE+aux summed, one backward) leave engine final weights matching the twin.
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from dataflow_training.testing.gradcheck import (  # noqa: E402
    check_model_step,
    family_gate_kwargs,
    rel_l2,
)

pytestmark = pytest.mark.gpu


def _tiny_cfg(**over):
    from dataflow_training.model_families.olmoe import ShapedOlmoeConfig

    return replace(ShapedOlmoeConfig.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow_training.model_families.olmoe import derive_dims

    return derive_dims(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------




# --- ladder 2: block fwd/recompute/bwd vs golden autograd (with aux) --------------

# block-level ladder retired with the golden models: block math is
# gated by the per-op kernel pins, the model-level dW comparison
# (grad: entries), and per-block isolation (tools/deep_compare.py
# --isolate); see docs/correctness_compare.md.


# --- structure + lowering ----------------------------------------------------------


def test_olmoe_stage_context_completeness():
    from dataflow_training.blocks.layouts import olmoe_activation_layout
    from dataflow_training.model_families.olmoe.blocks import OlmoeBlockFwd

    cl = olmoe_activation_layout(_tiny_dims())
    declared = {f.name for f in cl.fields}
    emitted = OlmoeBlockFwd.context_fields_emitted()
    assert declared == emitted, declared ^ emitted
    assert OlmoeBlockFwd.recompute_stage_count() < len(OlmoeBlockFwd.STAGES)
    # combine (the y-only epilogue) must sit past the recompute boundary
    names = [s[0] for s in OlmoeBlockFwd.STAGES]
    assert names[OlmoeBlockFwd.recompute_stage_count():] == ["moe_experts2_combine"]


def test_olmoe_partial_ownership_lowering_rejected():
    """Accounting for partial expert ownership is plumbed and unit-tested
    (tests/dataflow_training/modules/test_moe.py); the PROGRAM path refuses until a multi-rank
    runtime exists."""
    import dataclasses
    import unittest.mock as mock

    from dataflow_training.model_families.olmoe import derive_dims, lower_olmoe

    cfg = _tiny_cfg()
    part = dataclasses.replace(derive_dims(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow_training.model_families.olmoe.model.derive_moe_spec", return_value=part):
            lower_olmoe(cfg)


# --- ladder 3: full program through the real engine --------------------------------




def test_olmoe_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        **family_gate_kwargs("olmoe"),
    ).assert_ok()


def test_olmoe_grad_accum_two_rounds_matches_reference():
    """Grad accumulation with the aux objective: per-round CE + per-round
    aux summed across rounds, ONE backward on the total — engine == the
    isolated twin (each round is one packed forward, so the twin's
    forward-global aux IS the engine's round-global default; the runtime
    accumulates injected aux gradients per round)."""
    from dataflow_training.model_families import bridges
    from dataflow_training.run.driver import adamw_field_step
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow_training.blocks.base_blocks import AdamWHyper
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import EngineFinalBytes

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
        err = rel_l2(engine_tensor, twin_state[name])
        assert err < 3e-2, (name, err)
    result.close()


# --- engine-level gates: poison / interleave / measured-replan / determinism -------


def _run(engine_kwargs=None, resolver_wrapper=None, program=None, seed=7):
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
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    resolver = fam.build_resolver(fam.derive_dims(cfg))
    if resolver_wrapper is not None:
        resolver = resolver_wrapper(resolver, backend)
    from dataflow_training.data.segments import uniform_segments
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver, initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(fam.derive_dims(cfg), prog)},
    )
    out = {}
    for obj_id in ["W_embed", "W_head"] + [f"W_{i}" for i in range(cfg.n_layers)]:
        rec = result.objects.get(obj_id)
        slot = rec.backing or rec.fast
        out[obj_id] = torch_view(slot.buffer, (rec.size_bytes // 2,), torch.bfloat16).clone()
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
        err = rel_l2(a[k], b[k])
        assert err < tol, f"{k}: rel_l2={err}"

