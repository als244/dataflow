"""Qwen3-MoE correctness ladder (GPU): third family on the pluggable MoE
module, mirrored on tests/dataflow_training/models/test_olmoe.py.

Family-specific pins: qwen3's PER-HEAD qk-norm inherited verbatim (dense
qwen3 classes; per-head rstds), GQA exercised in the tiny config (4 q /
2 kv heads — the real models are 32/4 and 64/4), topk_then_softmax
routing (norm_topk_prob=true), aux at 0.001, NO shared expert, recompute
reproducing the routing decision bit-exactly, fixed-seed engine
determinism.

Tests:
- test_qwen3moe_stage_context_completeness: the forward block's emitted context fields equal the activation layout and only the y-only combine epilogue sits past the recompute boundary.
- test_qwen3moe_lowering_validates_and_plans: tiny cfg lowers and validates as qwen3moe with untied embed/head and q3moeattn/head_loss block keys, and plans/simulates with nonzero task intervals.
- test_qwen3moe_full_scale_presets_lower_and_validate: the 30B and 235B presets lower, validate, and emit one attention block per layer despite oversized weight footprints.
- test_qwen3moe_partial_ownership_lowering_rejected: lowering with a partial expert-ownership MoE spec raises NotImplementedError.
- test_qwen3moe_aux_zero_model_step_vs_golden: an aux_coef=0 model-step matches the golden twin.
- test_qwen3moe_plan_invariance: the model-step matches golden across memory budgets and recompute levels.
- test_qwen3moe_batch2_packed_sequences_vs_golden: a batch=2 packed-sequence model-step matches golden.
- test_qwen3moe_grad_accum_two_rounds_matches_reference: two grad-accum rounds (per-round CE+aux, one backward) leave engine final weights matching the twin.
- test_qwen3moe_fixed_seed_bitwise_deterministic: two runs at the same seed produce identical loss and weights.
- test_qwen3moe_measured_costs_replan_still_golden: profiling then replanning on measured costs leaves the math unchanged.
- test_qwen3moe_poison_on_free_changes_nothing: the poison_on_free engine option leaves loss and weights unchanged and non-NaN.
- test_qwen3moe_interleaving_stress_changes_nothing: random per-task launch jitter leaves loss and weights unchanged.
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
    from dataflow_training.model_families.qwen3moe import ShapedQwen3MoeConfig

    return replace(ShapedQwen3MoeConfig.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow_training.model_families.qwen3moe import derive_dims

    return derive_dims(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------




# --- ladder 2: block fwd/recompute/bwd vs golden autograd (with aux) --------------

# block-level ladder retired with the golden models: block math is
# gated by the per-op kernel pins, the model-level dW comparison
# (grad: entries), and per-block isolation (tools/deep_compare.py
# --isolate); see docs/correctness_compare.md.


# --- structure + lowering ----------------------------------------------------------


def test_qwen3moe_stage_context_completeness():
    from dataflow_training.blocks.layouts import qwen3moe_activation_layout
    from dataflow_training.model_families.qwen3moe.blocks import Qwen3MoeBlockFwd

    cl = qwen3moe_activation_layout(_tiny_dims())
    declared = {f.name for f in cl.fields}
    emitted = Qwen3MoeBlockFwd.context_fields_emitted()
    assert declared == emitted, declared ^ emitted
    assert Qwen3MoeBlockFwd.recompute_stage_count() < len(Qwen3MoeBlockFwd.STAGES)
    names = [s[0] for s in Qwen3MoeBlockFwd.STAGES]
    assert names[Qwen3MoeBlockFwd.recompute_stage_count():] == ["moe_experts2_combine"]


@pytest.mark.sim
def test_qwen3moe_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program, simulate_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    assert fam.name == "qwen3moe"
    program = fam.lower(cfg)
    validate_program(program)
    assert program.metadata["family"] == "qwen3moe-shaped"
    ids = {spec.id for spec in program.initial_objects}
    assert {"W_embed", "W_head", "O_head"} <= ids  # untied
    keys = {t.compute_block_key for t in program.tasks}
    assert {"q3moeattn_fwd", "q3moeattn_bwd", "head_loss"} <= keys
    planned = plan_program(program, fast_memory_capacity=8 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


def test_qwen3moe_full_scale_presets_lower_and_validate():
    """30B (48L) and 235B are definition-validated: lowering + exact sizes
    succeed even though their weight footprints (183 GiB / 1.4 TiB) exceed
    any single small-VRAM device — lowering-only here (sizes documented in
    training/models/qwen3moe.py)."""
    from dataflow.core import validate_program
    from dataflow_training.model_families.qwen3moe import ShapedQwen3MoeConfig, lower_qwen3moe

    for cfg in (ShapedQwen3MoeConfig.qwen3moe_30b(seq_len=128),
                ShapedQwen3MoeConfig.qwen3moe_235b(seq_len=128)):
        program = lower_qwen3moe(cfg)
        validate_program(program)
        n_blocks = sum(1 for t in program.tasks if t.compute_block_key == "q3moeattn_fwd")
        assert n_blocks == cfg.n_layers


def test_qwen3moe_partial_ownership_lowering_rejected():
    import dataclasses
    import unittest.mock as mock

    from dataflow_training.model_families.qwen3moe import derive_dims, lower_qwen3moe

    cfg = _tiny_cfg()
    part = dataclasses.replace(derive_dims(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow_training.model_families.qwen3moe.model.derive_moe_spec", return_value=part):
            lower_qwen3moe(cfg)


# --- ladder 3: full program through the real engine --------------------------------




@pytest.mark.sim
def test_qwen3moe_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        **family_gate_kwargs("qwen3moe"),
    ).assert_ok()


@pytest.mark.sim
def test_qwen3moe_plan_invariance():
    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                          **family_gate_kwargs("qwen3moe"))
    r2 = check_model_step(cfg, fast_memory_capacity=8 * 1024 * 1024, tol=3e-2,
                          **family_gate_kwargs("qwen3moe"))
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=8 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
        **family_gate_kwargs("qwen3moe"),
    )
    for r in (r1, r2, r3):
        r.assert_ok()


@pytest.mark.sim
def test_qwen3moe_batch2_packed_sequences_vs_golden():
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     **family_gate_kwargs("qwen3moe")).assert_ok()


@pytest.mark.sim
def test_qwen3moe_grad_accum_two_rounds_matches_reference():
    """Two grad-accum rounds with per-round CE + per-round aux, ONE
    backward on the total — engine == the isolated twin (each round is
    one packed forward, so the twin's forward-global aux IS the engine's
    round-global default)."""
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


# --- engine-level gates: determinism / measured-replan / multistep ------------------


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
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    resolver = fam.build_resolver(fam.derive_dims(cfg))
    if resolver_wrapper is not None:
        resolver = resolver_wrapper(resolver, backend)
    from dataflow_training.data.segments import uniform_segments
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver,
        initial_buffers=values, pool_prewarm=dry.pool_demand,
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


@pytest.mark.sim
def test_qwen3moe_fixed_seed_bitwise_deterministic():
    a = _run()
    b = _run()
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert torch.equal(a[k], b[k]), k


@pytest.mark.sim
def test_qwen3moe_measured_costs_replan_still_golden():
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.run.profiling import apply_measured_costs, profile_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    backend = CudaBackend()
    profiles = profile_program(program, fam.build_resolver(fam.derive_dims(cfg)), backend, soak_seconds=0)
    measured = apply_measured_costs(program, profiles)
    assert all("measured" in t.metadata for t in measured.tasks)

    base = _run()
    replanned = plan_program(measured, fast_memory_capacity=8 * 1024 * 1024).program
    again = _run(program=replanned)
    _assert_same(again, base)




@pytest.mark.sim
def test_qwen3moe_poison_on_free_changes_nothing():
    base = _run()
    poisoned = _run(engine_kwargs={"poison_on_free": True})
    _assert_same(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]  # not NaN


@pytest.mark.sim
def test_qwen3moe_interleaving_stress_changes_nothing():
    from dataflow.runtime.device.cuda_spin import SpinKernel

    def wrapper(resolver, backend):
        kernel = SpinKernel()
        rng = torch.Generator().manual_seed(123)

        class Jitter:
            def __init__(self, inner):
                self.inner = inner

            def launch(self, ctx):
                delay = float(torch.randint(20, 400, (1,), generator=rng)[0])
                kernel.launch_us(ctx.stream, delay)
                self.inner.launch(ctx)

        return lambda task: Jitter(resolver(task))

    base = _run()
    jittered = _run(resolver_wrapper=wrapper)
    _assert_same(jittered, base)
