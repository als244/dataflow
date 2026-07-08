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
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.training.testing.gradcheck import check_model_step, rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu


def _tiny_cfg(**over):
    from dataflow.training.dsv32 import ShapedDsv32Config

    return replace(ShapedDsv32Config.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow.training.dsv32 import dims_of_dsv32

    return dims_of_dsv32(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------


def test_golden_dsv32_trains():
    from dataflow.models.dsv32_reference import GoldenDsv32
    from dataflow.tasks.layouts import head_weight_layout

    cfg = _tiny_cfg()
    dims = _tiny_dims(cfg)
    gen = torch.Generator().manual_seed(0)

    def packed(layout):
        flat = torch.zeros(layout.total_bytes, dtype=torch.uint8)
        fb = flat.view(torch.uint8)
        for f in layout.fields:
            n = int(torch.tensor(f.shape).prod())
            if f.dtype == "fp32":
                v = torch.zeros(n, dtype=torch.float32)  # the balance bias
                fb[f.offset_bytes:f.offset_bytes + f.nbytes] = v.view(torch.uint8)
                continue
            vals = (torch.randn(n, generator=gen) * 0.02).to(torch.bfloat16)
            if f.name.endswith("_norm_w"):
                vals = torch.ones(n, dtype=torch.bfloat16)
            fb[f.offset_bytes:f.offset_bytes + f.nbytes] = vals.view(torch.uint8)
        return flat

    def table():
        n = dims.vocab_size * dims.d_model
        return (torch.randn(n, generator=gen) * 0.02).to(torch.bfloat16).view(torch.uint8)

    golden = GoldenDsv32.from_packed_bytes(
        dims, cfg.n_layers, table(),
        [packed(golden_layout) for golden_layout in
         (GoldenDsv32(dims=dims, n_layers=cfg.n_layers).block_layout(i)
          for i in range(cfg.n_layers))],
        packed(head_weight_layout(dims)),
    )
    toks = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen).cuda()
    tgts = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen).cuda()
    ce0, aux0 = golden.loss_terms(toks, tgts)
    assert torch.isfinite(aux0) and float(aux0.detach()) > 0.0
    losses = [golden.train_step(toks, tgts) for _ in range(3)]
    assert all(x == x for x in losses)
    assert abs(losses[0] - math.log(dims.vocab_size)) < 0.5
    assert losses[-1] < losses[0]
    # the balance bias moved (skewed routing at random init) but only by
    # multiples of the update speed
    moved = [b["w_router_bias"] for b in golden.w_blocks if "w_router_bias" in b]
    assert moved and any(m.abs().sum() > 0 for m in moved)
    for m in moved:
        steps = m / cfg.bias_update_speed
        assert torch.allclose(steps, steps.round(), atol=1e-4)


# --- lowering ----------------------------------------------------------------------


def test_dsv32_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program, simulate_program

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
    from dataflow.training.dsv32 import ShapedDsv32Config, lower_dsv32

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

    from dataflow.training.dsv32 import dims_of_dsv32, lower_dsv32

    cfg = _tiny_cfg()
    part = dataclasses.replace(dims_of_dsv32(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow.training.dsv32.moe_spec_of", return_value=part):
            lower_dsv32(cfg)


# --- full program through the real engine ------------------------------------------


_BIAS_ATOL = {
    "w_router_bias": 2.5e-3,  # 2.5x speed: +-2 steps + fp slack
    # LayerNorm bias: zero-init, sub-noise KL grads -> AdamW first-step
    # sign lottery (the dt_bias class; measured rel 0.4986 from sign
    # flips on ~1e-6 grads while every other field sat at 1e-3)
    "idx_k_ln_b": 2.5e-4,
}


def test_dsv32_model_step_vs_golden():
    check_model_step(_tiny_cfg(), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL).assert_ok()


def test_dsv32_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        field_atol=_BIAS_ATOL,
    ).assert_ok()


def test_dsv32_plan_invariance():
    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                          field_atol=_BIAS_ATOL)
    r2 = check_model_step(cfg, fast_memory_capacity=8 * 1024 * 1024, tol=3e-2,
                          field_atol=_BIAS_ATOL)
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=8 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
        field_atol=_BIAS_ATOL,
    )
    for r in (r1, r2, r3):
        r.assert_ok()


def test_dsv32_batch2_packed_sequences_vs_golden():
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL).assert_ok()


def test_dsv32_ga2_matches_golden():
    from dataflow.models.dsv32_reference import GoldenDsv32
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _tiny_cfg(grad_accum_rounds=2)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=16 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=3)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    golden = GoldenDsv32.from_packed_bytes(
        dims, cfg.n_layers, pinned("W_embed"),
        [pinned(f"W_{i}") for i in range(cfg.n_layers)], pinned("W_head"),
    )
    golden._pending_counts = []
    total = None
    for r in range(cfg.grad_accum_rounds):
        toks = torch_view(values[f"tokens_0_{r}"], (dims.tokens,), torch.int32).long().cuda()
        tgts = torch_view(values[f"targets_0_{r}"], (dims.tokens,), torch.int32).long().cuda()
        ce_r, aux_r = golden.loss_terms(toks, tgts)
        term = ce_r + aux_r
        total = term if total is None else total + term
    total.backward()
    golden.step_count = 1
    golden._opt_obj("embed", golden.w_embed)
    # bias rule on the STEP AGGREGATE counts (both rounds), mirroring the
    # runtime's dW accumulation; counts were captured per round in order
    n_moe = sum(1 for b in golden.w_blocks if "w_router_bias" in b)
    per_round = [golden._pending_counts[i::n_moe] for i in range(n_moe)] \
        if n_moe else []
    moe_i = 0
    for i, leaves in enumerate(golden.w_blocks):
        golden._opt_obj(f"block_{i}", leaves)
        if "w_router_bias" in leaves:
            agg = sum(per_round[moe_i])
            b = leaves["w_router_bias"]
            b.data.add_(torch.sign(agg.mean() - agg).to(b.dtype),
                        alpha=cfg.bias_update_speed)
            moe_i += 1
    golden._opt_obj("head", golden.w_head)

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
    )

    def worst_field_err(object_id):
        rec = result.objects.get(object_id)
        slot = rec.backing or rec.fast
        layout, leaves = golden.final_leaves(object_id)
        worst = 0.0
        for f in layout.fields:
            got = torch_view(slot.buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                             offset_bytes=f.offset_bytes)
            if f.name in _BIAS_ATOL:  # sign-lottery fields: absolute envelope
                d = (got.float().cpu() - leaves[f.name].float().cpu()).abs().max()
                assert float(d) <= _BIAS_ATOL[f.name], (f.name, float(d))
                continue
            worst = max(worst, rel_l2(got, leaves[f.name]))
        return worst

    assert worst_field_err("W_embed") < 3e-2
    for i in range(cfg.n_layers):
        assert worst_field_err(f"W_{i}") < 3e-2, f"W_{i}"
    assert worst_field_err("W_head") < 3e-2
    result.close()
    dry.close()
    for buf in values.values():
        backend.free(buf)


# --- engine gates -------------------------------------------------------------------


def _run(engine_kwargs=None, program=None, seed=7, resolver_wrapper=None):
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
    result = Engine(backend, **(engine_kwargs or {})).execute(
        prog, resolver=resolver,
        initial_buffers=values, pool_prewarm=dry.pool_demand,
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


def test_dsv32_multistep_matches_golden_and_loss_decreases():
    from dataflow.models.dsv32_reference import GoldenDsv32
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program
    from dataflow.training.train_loop import train

    STEPS = 3
    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=8 * 1024 * 1024)
    backend = CudaBackend()

    gen = torch.Generator().manual_seed(99)
    one_batch = (
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
    )
    batches = [one_batch] * STEPS

    snapshot = fam.initial_values(planned.program, cfg, backend, seed=5)

    def pinned(name):
        buf = snapshot[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    golden = GoldenDsv32.from_packed_bytes(
        dims, cfg.n_layers, pinned("W_embed"),
        [pinned(f"W_{i}") for i in range(cfg.n_layers)], pinned("W_head"),
    )
    golden_losses = [
        golden.train_step(t.long().cuda(), g.long().cuda()) for t, g in batches
    ]

    report = train(
        planned.program, cfg, backend,
        steps=STEPS, seed=5, token_stream=lambda s: batches[s],
    )
    for ours, ref in zip(report.losses, golden_losses):
        assert abs(ours - ref) / max(abs(ref), 1e-9) < 3e-2, (report.losses, golden_losses)
    assert report.losses[-1] < report.losses[0]
    assert all(n == 0 for n in report.step_slab_overflows[1:]), report.step_slab_overflows


def test_dsv32_frozen_indexer_ablation():
    """train_indexer=False (Shein ablation knob): model-step still matches
    golden AND the five indexer fields are BIT-FROZEN across the step
    (no gradients, no AdamW, not even weight decay)."""
    import dataclasses

    from dataflow.models.dsv32_reference import GoldenDsv32
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _tiny_cfg(train_indexer=False)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL).assert_ok()

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=11)
    before = {}
    wl_of = {}
    from dataflow.tasks.layouts import dsv32_dense_weight_layout, dsv32_moe_weight_layout
    for i in range(cfg.n_layers):
        wl = (dsv32_dense_weight_layout(dims) if dims.kind_of(i) == "dense"
              else dsv32_moe_weight_layout(dims))
        wl_of[i] = wl
        buf = values[f"W_{i}"]
        before[i] = {
            f.name: torch_view(buf, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes).clone()
            for f in wl.fields if f.name.startswith(("w_idx", "idx_k_ln"))
        }
    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
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
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow.tasks.layouts import dsv32_dense_weight_layout, dsv32_moe_weight_layout
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _tiny_cfg(sparse_mode=False)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL).assert_ok()

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=13)
    idx_fields = ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w")
    before = {}
    wl_of = {}
    for i in range(cfg.n_layers):
        wl = (dsv32_dense_weight_layout(dims) if dims.kind_of(i) == "dense"
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
    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
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
