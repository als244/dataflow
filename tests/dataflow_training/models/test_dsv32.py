"""DeepSeek-V3.2 family ladder (DSA sparse mode) (GPU): golden + full programs through the
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
- test_dsv32_lowering_validates_and_plans: dsv32 lowering validates, tags the family metadata, carries the DSA dense/moe block keys and the expected dense-block count, and plans to a positive-makespan schedule.
- test_dsv32_full_scale_presets_lower_and_validate: the mini, 671b, and glm5 presets lower and validate with the expected DSA block depth.
- test_dsv32_partial_ownership_lowering_rejected: a partial-ownership MoE expert set raises NotImplementedError at lowering.
- test_dsv32_aux_zero_model_step_vs_golden: an aux-off model step matches golden with the bias and index-LayerNorm-bias fields held under the atol envelope.
- test_dsv32_plan_invariance: the model step matches golden across fast-memory capacities and a recompute-levels plan.
- test_dsv32_batch2_packed_sequences_vs_golden: a batch-2 packed-sequence model step matches golden.
- test_dsv32_short_sequences_lt_index_topk_vs_golden: sequences shorter than index_topk clamp DSA selection to min(k, L) per sequence and match golden.
- test_dsv32_ga2_matches_reference: two grad-accum rounds with the LBL composite, indexer KL, and noaux bias rule match the twin under step-aggregated assignment counts.
- test_dsv32_fixed_seed_bitwise_deterministic: two fixed-seed engine runs produce bit-identical loss and weight bytes.
- test_dsv32_measured_costs_replan_still_golden: profiling, applying measured costs, and replanning still reproduce the base run.
- test_dsv32_frozen_indexer_ablation: with train_indexer off the model step matches golden and the indexer fields stay bit-frozen across the step.
- test_dsv32_dense_warmup_model_step: in dense warm-up the main model (including embed and head) stays bit-frozen while the indexer fields move under a live full-prefix KL.
- test_dsv32_poison_on_free_changes_nothing: enabling poison-on-free leaves loss and weights unchanged and non-NaN.
- test_dsv32_interleaving_stress_changes_nothing: random per-launch jitter leaves loss and weights unchanged.
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("cuda.bindings.runtime")  # real CudaBackend + SpinKernel
pytest.importorskip("dataflow_sim")            # plan_program / simulate_program

from dataflow_training.testing.gradcheck import (  # noqa: E402
    check_model_step,
    family_gate_kwargs,
    rel_l2,
)

pytestmark = [pytest.mark.gpu, pytest.mark.sim]


def _tiny_cfg(**over):
    from dataflow_training.model_families.dsv32 import ShapedDsv32Config

    return replace(ShapedDsv32Config.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow_training.model_families.dsv32 import derive_dims

    return derive_dims(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------




# --- lowering ----------------------------------------------------------------------


def test_dsv32_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program, simulate_program

    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    assert fam.name == "dsv32"
    program = fam.lower(cfg)
    validate_program(program)
    assert program.metadata["family"] == "dsv32-shaped"
    keys = {t.compute_block_key for t in program.tasks}
    assert {"dsadense_fwd", "dsamoe_fwd", "dsamoe_bwd", "head_loss"} <= keys
    # depth mix: exactly first_k_dense dense blocks
    n_dense = sum(1 for t in program.tasks if t.compute_block_key == "dsadense_fwd")
    assert n_dense == cfg.first_k_dense * cfg.grad_accum_rounds
    planned = plan_program(program, fast_memory_capacity=8 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


def test_dsv32_full_scale_presets_lower_and_validate():
    from dataflow.core import validate_program
    from dataflow_training.model_families.dsv32 import ShapedDsv32Config, lower_dsv32

    for ctor, layers in ((ShapedDsv32Config.dsv32_mini, 18),
                         (ShapedDsv32Config.dsv32_671b, 61),
                         (ShapedDsv32Config.glm5, 78)):
        cfg = ctor(seq_len=128)
        program = lower_dsv32(cfg)
        validate_program(program)
        n_blocks = sum(1 for t in program.tasks
                       if t.compute_block_key.endswith("_fwd")
                       and t.compute_block_key.startswith("dsa"))
        assert n_blocks == layers


def test_dsv32_partial_ownership_lowering_rejected():
    import dataclasses
    import unittest.mock as mock

    from dataflow_training.model_families.dsv32 import derive_dims, lower_dsv32

    cfg = _tiny_cfg()
    part = dataclasses.replace(derive_dims(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow_training.model_families.dsv32.model.derive_moe_spec", return_value=part):
            lower_dsv32(cfg)


# --- full program through the real engine ------------------------------------------


_BIAS_ATOL = {
    "w_router_bias": 2.5e-3,  # 2.5x speed: +-2 steps + fp slack
    # LayerNorm bias: zero-init, sub-noise KL grads -> AdamW first-step
    # sign lottery (the dt_bias class; measured rel 0.4986 from sign
    # flips on ~1e-6 grads while every other field sat at 1e-3)
    "idx_k_ln_b": 2.5e-4,
}




def test_dsv32_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        field_atol=_BIAS_ATOL, **family_gate_kwargs("dsv32"),
    ).assert_ok()


def test_dsv32_plan_invariance():
    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                          field_atol=_BIAS_ATOL, **family_gate_kwargs("dsv32"))
    r2 = check_model_step(cfg, fast_memory_capacity=8 * 1024 * 1024, tol=3e-2,
                          field_atol=_BIAS_ATOL, **family_gate_kwargs("dsv32"))
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=8 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
        field_atol=_BIAS_ATOL, **family_gate_kwargs("dsv32"),
    )
    for r in (r1, r2, r3):
        r.assert_ok()


def test_dsv32_batch2_packed_sequences_vs_golden():
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL,
                     **family_gate_kwargs("dsv32")).assert_ok()


def test_dsv32_short_sequences_lt_index_topk_vs_golden():
    """Packed round with sequences SHORTER than index_topk (24): DSA
    selection must clamp to min(k, L) PER SEQUENCE (DeepSeek's
    topk(min(index_topk, seqlen))) — a short sequence attends densely to
    its causal prefix. Regression gate: pre-fix the engine crashed
    (torch.topk k>L) and the reference did a boundary-agnostic global
    topk. seq_lens=(104, 16, 8): the 16- and 8-token sequences are < 24."""
    cfg = _tiny_cfg(seq_lens=(104, 16, 8))
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL,
                     **family_gate_kwargs("dsv32")).assert_ok()


def test_dsv32_ga2_matches_reference():
    """Two grad-accum rounds with the LBL composite, the indexer KL and
    the noaux bias rule: engine == the isolated twin. The twin stashes
    per-FORWARD assignment counts only, so the STEP-AGGREGATE bias rule
    (the engine's dW count accumulation) is applied here by summing each
    MoE module's counts across rounds before its own sign rule;
    sign-lottery bias fields compare under the _BIAS_ATOL envelope (see
    module docstring)."""
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
    B = dims.max_tokens // cfg.seq_len
    total = None
    agg_counts = {}
    for r in range(cfg.grad_accum_rounds):
        toks = torch_view(values[f"tokens_0_{r}"], (dims.max_tokens,),
                          torch.int32).long().cuda().view(B, cfg.seq_len)
        tgts = torch_view(values[f"targets_0_{r}"], (dims.max_tokens,),
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


def test_dsv32_fixed_seed_bitwise_deterministic():
    a = _run()
    b = _run()
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert torch.equal(a[k], b[k]), k


def test_dsv32_measured_costs_replan_still_golden():
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




def test_dsv32_frozen_indexer_ablation():
    """train_indexer=False (the ablation knob): model-step still matches
    golden AND the five indexer fields are BIT-FROZEN across the step
    (no gradients, no AdamW, not even weight decay)."""
    import dataclasses

    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    cfg = _tiny_cfg(train_indexer=False)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL,
                     **family_gate_kwargs("dsv32")).assert_ok()

    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=11)
    before = {}
    wl_of = {}
    from dataflow_training.blocks.layouts import dsv32_dense_weight_layout, dsv32_moe_weight_layout
    for i in range(cfg.n_layers):
        wl = (dsv32_dense_weight_layout(dims) if dims.kinds[i] == "dense"
              else dsv32_moe_weight_layout(dims))
        wl_of[i] = wl
        buf = values[f"W_{i}"]
        before[i] = {
            f.name: torch_view(buf, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes).clone()
            for f in wl.fields if f.name.startswith(("w_idx", "idx_k_ln"))
        }
    from dataflow_training.data.segments import uniform_segments

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)},
    )
    for i in range(cfg.n_layers):
        rec = result.objects.get(f"W_{i}")
        slot = rec.backing or rec.fast
        for f in wl_of[i].fields:
            if not f.name.startswith(("w_idx", "idx_k_ln")):
                continue
            after = torch_view(slot.buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes)
            assert torch.equal(after, before[i][f.name]), (i, f.name)
    result.close()
    dry.close()
    for buf in values.values():
        backend.free(buf)


def test_dsv32_dense_warmup_model_step():
    """Dense warm-up gate (sparse_mode=False) — model-step matches
    golden; the MAIN MODEL (every non-indexer field, embed/head/router-
    bias included) is BIT-FROZEN across a real engine step; the indexer
    fields MOVE (full-prefix KL is live)."""
    import dataclasses

    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow_training.blocks.layouts import dsv32_dense_weight_layout, dsv32_moe_weight_layout
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    cfg = _tiny_cfg(sparse_mode=False)
    idx_only = ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w")
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL, reference_train_only=idx_only,
                     **family_gate_kwargs("dsv32")).assert_ok()

    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=13)
    idx_fields = ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w")
    before = {}
    wl_of = {}
    for i in range(cfg.n_layers):
        wl = (dsv32_dense_weight_layout(dims) if dims.kinds[i] == "dense"
              else dsv32_moe_weight_layout(dims))
        wl_of[i] = wl
        buf = values[f"W_{i}"]
        before[i] = {
            f.name: torch_view(buf, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes).clone()
            for f in wl.fields
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
    assert moved > 0, "no indexer field moved — KL not training"
    for obj, ref in (("W_embed", embed_before), ("W_head", head_before)):
        rec = result.objects.get(obj)
        slot = rec.backing or rec.fast
        got = torch_view(slot.buffer, (rec.size_bytes,), torch.uint8)
        assert torch.equal(got, ref), f"{obj} moved in warm-up"
    result.close()
    dry.close()
    for buf in values.values():
        backend.free(buf)


def test_dsv32_poison_on_free_changes_nothing():
    base = _run()
    poisoned = _run(engine_kwargs={"poison_on_free": True})
    _assert_same(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]  # not NaN


def test_dsv32_interleaving_stress_changes_nothing():
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
