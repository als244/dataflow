"""OLMoE correctness ladder (GPU): the first family on the pluggable MoE
module, mirrored on tests/models/test_qwen35.py.

Family-specific pins: full-row qk-norm (one rstd per token), the MoE tail
spliced after resid1_norm2, aux load-balance gradient injection (the
golden's autograd objective is CE + aux while its reported loss is CE),
recompute reproducing the ROUTING DECISION bit-exactly (int ctx fields
compared with torch.equal), and end-to-end engine determinism (fixed seed
twice -> identical bytes).
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.training.testing.gradcheck import (  # noqa: E402
    check_model_step,
    family_gate_kwargs,
    rel_l2,
)

pytestmark = pytest.mark.gpu


def _tiny_cfg(**over):
    from dataflow.training.models.olmoe import ShapedOlmoeConfig

    return replace(ShapedOlmoeConfig.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow.training.models.olmoe import dims_of_olmoe

    return dims_of_olmoe(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------




# --- ladder 2: block fwd/recompute/bwd vs golden autograd (with aux) --------------

# block-level ladder retired with the golden models: block math is
# gated by the per-op kernel pins, the model-level dW comparison
# (grad: entries), and per-block isolation (tools/deep_compare.py
# --isolate); see docs/correctness_compare.md.


# --- structure + lowering ----------------------------------------------------------


def test_olmoe_stage_context_completeness():
    from dataflow.tasks.layouts import olmoe_activation_layout
    from dataflow.tasks.models.olmoe_blocks import OlmoeBlockFwd

    cl = olmoe_activation_layout(_tiny_dims())
    declared = {f.name for f in cl.fields}
    emitted = OlmoeBlockFwd.context_fields_emitted()
    assert declared == emitted, declared ^ emitted
    assert OlmoeBlockFwd.recompute_stage_count() < len(OlmoeBlockFwd.STAGES)
    # combine (the y-only epilogue) must sit past the recompute boundary
    names = [s[0] for s in OlmoeBlockFwd.STAGES]
    assert names[OlmoeBlockFwd.recompute_stage_count():] == ["moe_experts2_combine"]


def test_olmoe_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program, simulate_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    assert fam.name == "olmoe"
    program = fam.lower(cfg)
    validate_program(program)
    assert program.metadata["family"] == "olmoe-shaped"
    ids = {spec.id for spec in program.initial_objects}
    assert {"W_embed", "W_head", "O_head"} <= ids  # untied
    keys = {t.compute_block_key for t in program.tasks}
    assert {"moeattn_fwd", "moeattn_bwd", "head_loss"} <= keys
    planned = plan_program(program, fast_memory_capacity=8 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


def test_olmoe_partial_ownership_lowering_rejected():
    """Accounting for partial expert ownership is plumbed and unit-tested
    (tests/modules/test_moe.py); the PROGRAM path refuses until a multi-rank
    runtime exists."""
    import dataclasses
    import unittest.mock as mock

    from dataflow.training.models.olmoe import dims_of_olmoe, lower_olmoe

    cfg = _tiny_cfg()
    part = dataclasses.replace(dims_of_olmoe(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow.training.models.olmoe.moe_spec_of", return_value=part):
            lower_olmoe(cfg)


# --- ladder 3: full program through the real engine --------------------------------




def test_olmoe_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        **family_gate_kwargs("olmoe"),
    ).assert_ok()


def test_olmoe_plan_invariance():
    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                          **family_gate_kwargs("olmoe"))
    r2 = check_model_step(cfg, fast_memory_capacity=8 * 1024 * 1024, tol=3e-2,
                          **family_gate_kwargs("olmoe"))
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=8 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
        **family_gate_kwargs("olmoe"),
    )
    for r in (r1, r2, r3):
        r.assert_ok()


def test_olmoe_batch2_packed_sequences_vs_golden():
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     **family_gate_kwargs("olmoe")).assert_ok()


def test_olmoe_ga2_matches_reference():
    """Grad accumulation with the aux objective: per-round CE + per-round
    aux summed across rounds, ONE backward on the total — engine == the
    isolated twin (each round is one packed forward, so the twin's
    forward-global aux IS the engine's round-global default; the runtime
    accumulates injected aux gradients per round)."""
    from dataflow.pretrain import bridges
    from dataflow.pretrain.driver import adamw_field_step
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.base_blocks import AdamWHyper
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program
    from dataflow.training.testing.gradcheck import EngineFinalBytes

    cfg = _tiny_cfg(grad_accum_rounds=2)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=16 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=3)

    model = bridges.build_reference_model(cfg)
    bridges.load_reference_init(model, cfg, dims,
                                bridges.get_bytes_from_values(values))
    model.train()
    B = dims.tokens // cfg.seq_len
    total = None
    for r in range(cfg.grad_accum_rounds):
        toks = torch_view(values[f"tokens_0_{r}"], (dims.tokens,),
                          torch.int32).long().cuda().view(B, cfg.seq_len)
        tgts = torch_view(values[f"targets_0_{r}"], (dims.tokens,),
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

    from dataflow.runtime.engine import uniform_segments

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
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    prog = program if program is not None else plan_program(
        fam.lower(cfg), fast_memory_capacity=8 * 1024 * 1024,
    ).program

    backend = CudaBackend()
    values = fam.initial_values(prog, cfg, backend, seed=seed)
    dry = Engine(FakeBackend()).execute(prog, initial_buffers=values)
    resolver = fam.build_resolver(fam.dims_of(cfg))
    if resolver_wrapper is not None:
        resolver = resolver_wrapper(resolver, backend)
    from dataflow.runtime.engine import uniform_segments
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver, initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(fam.dims_of(cfg), prog)},
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


def test_olmoe_fixed_seed_bitwise_deterministic():
    """Same seed, same plan, twice -> identical LOSS BYTES and weights
    (routing ties, sort, grouped GEMMs, combine: all deterministic)."""
    a = _run()
    b = _run()
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert torch.equal(a[k], b[k]), k


def test_olmoe_poison_on_free_changes_nothing():
    base = _run()
    poisoned = _run(engine_kwargs={"poison_on_free": True})
    _assert_same(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]  # not NaN


def test_olmoe_interleaving_stress_changes_nothing():
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


def test_olmoe_measured_costs_replan_still_golden():
    """The end-to-end profiling gate: every signature (incl. moeattn_bwd,
    whose packed ctx carries int32 routing fields) must profile through the
    profile_fill hook without garbage-index crashes, and re-planning on
    measured costs must not change the math."""
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program
    from dataflow.training.profiling import apply_measured_costs, profile_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    backend = CudaBackend()
    profiles = profile_program(program, fam.build_resolver(fam.dims_of(cfg)), backend, soak_seconds=0)
    measured = apply_measured_costs(program, profiles)
    assert all("measured" in t.metadata for t in measured.tasks)

    base = _run()
    replanned = plan_program(measured, fast_memory_capacity=8 * 1024 * 1024).program
    again = _run(program=replanned)
    _assert_same(again, base)


