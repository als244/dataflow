"""DeepSeek-V3.2 family ladder (DSA sparse mode) (GPU): golden + full programs through the
real engine. Block-level MLA/moe pins live in tests/modules/test_mla.py.

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
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.testing.gradcheck import (  # noqa: E402
    check_model_step,
    family_gate_kwargs,
    rel_l2,
)

pytestmark = pytest.mark.gpu


def _tiny_cfg(**over):
    from dataflow_training.model_families.glm52 import ShapedGlm52Config

    return replace(ShapedGlm52Config.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow_training.model_families.glm52 import derive_dims

    return derive_dims(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------




# --- lowering ----------------------------------------------------------------------


def test_glm52_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program, simulate_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    assert fam.name == "glm52"
    program = fam.lower(cfg)
    validate_program(program)
    assert program.metadata["family"] == "glm52-shaped"
    keys = {t.compute_block_key for t in program.tasks}
    assert {"gdl_fwd", "gml_fwd", "gmf_fwd", "gmf_bwd", "head_loss"} <= keys
    # role mix (tiny: F F S S F S, first_k=1): 1 gdl, 2 gml, 3 gmf
    per = cfg.grad_accum_rounds
    assert sum(1 for t in program.tasks if t.compute_block_key == "gdl_fwd") == 1 * per
    assert sum(1 for t in program.tasks if t.compute_block_key == "gml_fwd") == 2 * per
    assert sum(1 for t in program.tasks if t.compute_block_key == "gmf_fwd") == 3 * per
    planned = plan_program(program, fast_memory_capacity=8 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


def test_glm52_full_scale_presets_lower_and_validate():
    from dataflow.core import validate_program
    from dataflow_training.model_families.glm52 import ShapedGlm52Config, lower_glm52

    for ctor, layers in ((ShapedGlm52Config.glm52_mini, 18),
                         (ShapedGlm52Config.glm52, 78)):
        cfg = ctor(seq_len=128)
        program = lower_glm52(cfg)
        validate_program(program)
        n_blocks = sum(1 for t in program.tasks
                       if t.compute_block_key.endswith("_fwd")
                       and t.compute_block_key.startswith("g"))
        assert n_blocks == layers


def test_glm52_partial_ownership_lowering_rejected():
    import dataclasses
    import unittest.mock as mock

    from dataflow_training.model_families.glm52 import derive_dims, lower_glm52

    cfg = _tiny_cfg()
    part = dataclasses.replace(derive_dims(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow_training.model_families.glm52.model.derive_moe_spec", return_value=part):
            lower_glm52(cfg)


# --- full program through the real engine ------------------------------------------


_BIAS_ATOL = {
    "w_router_bias": 2.5e-3,  # 2.5x speed: +-2 steps + fp slack
    # LayerNorm bias: zero-init, sub-noise KL grads -> AdamW first-step
    # sign lottery (the dt_bias class; measured rel 0.4986 from sign
    # flips on ~1e-6 grads while every other field sat at 1e-3)
    "idx_k_ln_b": 2.5e-4,
}




def test_glm52_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        field_atol=_BIAS_ATOL, **family_gate_kwargs("glm52"),
    ).assert_ok()


def test_glm52_plan_invariance():
    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                          field_atol=_BIAS_ATOL, **family_gate_kwargs("glm52"))
    r2 = check_model_step(cfg, fast_memory_capacity=8 * 1024 * 1024, tol=3e-2,
                          field_atol=_BIAS_ATOL, **family_gate_kwargs("glm52"))
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=8 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
        field_atol=_BIAS_ATOL, **family_gate_kwargs("glm52"),
    )
    for r in (r1, r2, r3):
        r.assert_ok()


def test_glm52_batch2_packed_sequences_vs_golden():
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL,
                     **family_gate_kwargs("glm52")).assert_ok()


def test_glm52_ga2_matches_reference():
    """Two grad-accum rounds with the LBL composite, the leader-group
    indexer KL and the noaux bias rule: engine == the isolated twin. The
    twin stashes per-FORWARD assignment counts only, so the
    STEP-AGGREGATE bias rule (the engine's dW count accumulation) is
    applied here by summing each MoE module's counts across rounds
    before its own sign rule; sign-lottery bias fields compare under the
    _BIAS_ATOL envelope (see module docstring)."""
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
        match_field_atol,
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
    drive_idx = (bool(getattr(cfg, "train_indexer", False))
                 and hasattr(model, "indexer_loss"))
    if drive_idx:
        model.enable_indexer_kl(True)
    B = dims.tokens // cfg.seq_len
    total = None
    agg_counts = {}
    for r in range(cfg.grad_accum_rounds):
        toks = torch_view(values[f"tokens_0_{r}"], (dims.tokens,),
                          torch.int32).long().cuda().view(B, cfg.seq_len)
        tgts = torch_view(values[f"targets_0_{r}"], (dims.tokens,),
                          torch.int32).long().cuda().view(B, cfg.seq_len)
        loss_r = model.loss(toks, tgts, aux_coef=cfg.aux_coef)
        if drive_idx:
            loss_r = loss_r + model.indexer_loss()
        total = loss_r if total is None else total + loss_r
        for mod_name, module in model.named_modules():
            if hasattr(module, "apply_bias_update"):
                prev = agg_counts.get(mod_name)
                agg_counts[mod_name] = (
                    module.last_counts if prev is None
                    else prev + module.last_counts)
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
    for mod_name, module in model.named_modules():
        if hasattr(module, "apply_bias_update"):
            module.last_counts = agg_counts[mod_name]
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
        if atol is not None:  # sign-lottery fields: absolute envelope
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
        # BYTES, not bf16: fp32 fields (w_idx_w, w_router_bias) reinterpreted
        # as bf16 can alias NaN bit patterns, and torch.equal says NaN != NaN
        # even on identical bytes (dsv3's test passed by bit-pattern luck)
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


def test_glm52_fixed_seed_bitwise_deterministic():
    a = _run()
    b = _run()
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert torch.equal(a[k], b[k]), k


def test_glm52_measured_costs_replan_still_golden():
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.run.profiling import apply_measured_costs, profile_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    backend = CudaBackend()
    resolver = fam.build_resolver(fam.derive_dims(cfg))
    profiles = profile_program(program, resolver, backend, soak_seconds=0)
    # the policy-frozen router bias puts these families on the
    # frozen-fingerprint signature path: apply needs the same resolver
    measured = apply_measured_costs(program, profiles, resolver=resolver)
    assert all("measured" in t.metadata for t in measured.tasks)

    base = _run()
    replanned = plan_program(measured, fast_memory_capacity=8 * 1024 * 1024).program
    again = _run(program=replanned)
    _assert_same(again, base)





# block-level ladder retired with the golden models: block math is
# gated by the per-op kernel pins, the model-level dW comparison
# (grad: entries), and per-block isolation (tools/deep_compare.py
# --isolate); see docs/correctness_compare.md.


def test_glm52_poison_on_free_changes_nothing():
    base = _run()
    poisoned = _run(engine_kwargs={"poison_on_free": True})
    _assert_same(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]  # not NaN


def test_glm52_interleaving_stress_changes_nothing():
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


def test_glm52_frozen_indexer_ablation():
    """train_indexer=False: model-step matches the frozen golden, the
    leader indexer fields are BIT-FROZEN across the step, and lowering
    emits NO dM chain (no KL => no metadata gradient)."""
    from dataflow_training.model_families.families import resolve_family

    cfg = _tiny_cfg(train_indexer=False)
    fam = resolve_family(cfg)
    prog = fam.lower(cfg)
    assert not [o for o in prog.initial_objects if o.id.startswith("dAuxTemp_")]
    assert not [oid for task in prog.task_by_id().values()
                for oid in (task.outputs and [o.id for o in task.outputs] or [])
                if str(oid).startswith("dAuxTemp_")]
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL,
                     **family_gate_kwargs("glm52")).assert_ok()


def test_glm52_dense_warmup_model_step():
    """Dense warm-up gate (sparse_mode=False) — model-step matches golden.
    IndexShare twist over dsv32's warm-up: followers deposit FULL-PREFIX
    rows into the group dM and the leader trains on the group centroid
    (p_own + dM)/N — the tiny config's N=3 group [1,2,3] and N=2 group
    [4,5] pin the averaging. Main wgrads are SKIPPED in the engine (dW
    zeroed); the frozen optimizer makes that invisible to param/state
    comparisons, which is exactly the point."""
    cfg = _tiny_cfg(sparse_mode=False)
    idx_only = ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w")
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL, reference_train_only=idx_only,
                     **family_gate_kwargs("glm52")).assert_ok()


def test_glm52_dense_warmup_freeze_and_movement():
    """Across a REAL engine warm-up step: every non-indexer field —
    embed/head/router-bias included, and EVERY follower field — is
    BIT-FROZEN; leader indexer fields move (full-prefix group KL live)."""
    import dataclasses

    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow_training.blocks.layouts import (
        dsv3_moe_weight_layout,
        dsv32_dense_weight_layout,
        dsv32_moe_weight_layout,
    )
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    cfg = _tiny_cfg(sparse_mode=False)
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=13)
    idx_fields = ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w")
    wl_of = {}
    for i in range(cfg.n_layers):
        kind = dims.kinds[i]
        wl_of[i] = {"gdl": dsv32_dense_weight_layout,
                    "gml": dsv32_moe_weight_layout,
                    "gmf": dsv3_moe_weight_layout}[kind](dims)
    before = {}
    for i in range(cfg.n_layers):
        buf = values[f"W_{i}"]
        before[i] = {
            f.name: torch_view(buf, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes).clone()
            for f in wl_of[i].fields
        }
    embed_before = torch_view(values["W_embed"],
                              (values["W_embed"].size_bytes,), torch.uint8).clone()
    head_before = torch_view(values["W_head"],
                             (values["W_head"].size_bytes,), torch.uint8).clone()
    from dataflow_training.data.segments import uniform_segments

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)},
    )
    moved = 0
    for i in range(cfg.n_layers):
        rec = result.objects.get(f"W_{i}")
        slot = rec.backing or rec.fast
        for f in wl_of[i].fields:
            after = torch_view(slot.buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes)
            if f.name in idx_fields:
                moved += int(not torch.equal(after, before[i][f.name]))
            else:
                assert torch.equal(after, before[i][f.name]), \
                    (i, f.name, "main field moved in warm-up")
    assert moved > 0, "no leader indexer field moved — group KL not training"
    for obj, ref in (("W_embed", embed_before), ("W_head", head_before)):
        rec = result.objects.get(obj)
        slot = rec.backing or rec.fast
        got = torch_view(slot.buffer, (rec.size_bytes,), torch.uint8)
        assert torch.equal(got, ref), f"{obj} moved in warm-up"
    result.close()
    dry.close()
    for buf in values.values():
        backend.free(buf)
