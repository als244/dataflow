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
    from dataflow.training.models.glm52 import ShapedGlm52Config

    return replace(ShapedGlm52Config.tiny(), **over)


def _tiny_dims(cfg=None):
    from dataflow.training.models.glm52 import dims_of_glm52

    return dims_of_glm52(cfg if cfg is not None else _tiny_cfg())


# --- golden self-consistency -----------------------------------------------------


def test_golden_glm52_trains():
    from dataflow.models.glm52_reference import GoldenGlm52
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

    golden = GoldenGlm52.from_packed_bytes(
        dims, cfg.n_layers, table(),
        [packed(golden_layout) for golden_layout in
         (GoldenGlm52(dims=dims, n_layers=cfg.n_layers).block_layout(i)
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


def test_glm52_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program, simulate_program

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
    from dataflow.training.models.glm52 import ShapedGlm52Config, lower_glm52

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

    from dataflow.training.models.glm52 import dims_of_glm52, lower_glm52

    cfg = _tiny_cfg()
    part = dataclasses.replace(dims_of_glm52(cfg).moe, expert_ids=(0, 1, 2))
    with pytest.raises(NotImplementedError):
        with mock.patch("dataflow.training.models.glm52.moe_spec_of", return_value=part):
            lower_glm52(cfg)


# --- full program through the real engine ------------------------------------------


_BIAS_ATOL = {
    "w_router_bias": 2.5e-3,  # 2.5x speed: +-2 steps + fp slack
    # LayerNorm bias: zero-init, sub-noise KL grads -> AdamW first-step
    # sign lottery (the dt_bias class; measured rel 0.4986 from sign
    # flips on ~1e-6 grads while every other field sat at 1e-3)
    "idx_k_ln_b": 2.5e-4,
}


def test_glm52_model_step_vs_golden():
    check_model_step(_tiny_cfg(), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL).assert_ok()


def test_glm52_aux_zero_model_step_vs_golden():
    check_model_step(
        _tiny_cfg(aux_coef=0.0), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        field_atol=_BIAS_ATOL,
    ).assert_ok()


def test_glm52_plan_invariance():
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


def test_glm52_batch2_packed_sequences_vs_golden():
    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL).assert_ok()


def test_glm52_ga2_matches_golden():
    from dataflow.models.glm52_reference import GoldenGlm52
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

    golden = GoldenGlm52.from_packed_bytes(
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


def test_glm52_fixed_seed_bitwise_deterministic():
    a = _run()
    b = _run()
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert torch.equal(a[k], b[k]), k


def test_glm52_measured_costs_replan_still_golden():
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


def test_glm52_multistep_matches_golden_and_loss_decreases():
    from dataflow.models.glm52_reference import GoldenGlm52
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

    golden = GoldenGlm52.from_packed_bytes(
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



def test_glm52_leader_follower_pair_ladder():
    """THE IndexShare math gate, isolated at block level: a gml LEADER +
    gmf FOLLOWER chain vs golden autograd of the two-block compose with
    L_multi = (KL(p_leader||sigma) + KL(p_follower||sigma)) / 2. Runs the
    runtime bwds in reverse order (follower creates dM with its target;
    leader consumes and chains sigma - (p_own + dM)/2 through its indexer
    weights) and compares EVERY gradient."""
    from dataflow.models.glm52_reference import GoldenGlm52
    from dataflow.tasks.modules.dsa_reference import dsa_mask_from_idx
    from dataflow.tasks.models.glm52_blocks import (
        Glm52MfBlockBwd,
        Glm52MfBlockFwd,
        Glm52MlBlockBwd,
        Glm52MlBlockFwd,
    )
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME
    from dataflow.tasks.kernels import KernelCtx, resolve_kernels
    from dataflow.tasks.layouts import glm52_meta_layout, grad_layout

    # pattern chosen so the gml leader at layer 1 serves EXACTLY one
    # follower (group {1, 2}, N=2) — the chain below is the whole group
    cfg = _tiny_cfg(indexer_types=("full", "full", "shared", "full", "full", "shared"))
    dims = _tiny_dims(cfg)
    kernels = resolve_kernels()
    kctx = KernelCtx()
    ld_fwd, ld_bwd = Glm52MlBlockFwd(dims, kernels), Glm52MlBlockBwd(dims, kernels)
    f_fwd, f_bwd = Glm52MfBlockFwd(dims, kernels), Glm52MfBlockBwd(dims, kernels)

    gen = torch.Generator(device="cuda").manual_seed(77)

    def mk_weights(wl):
        w = {}
        for f in wl.fields:
            n = int(torch.tensor(f.shape).prod())
            dt = TORCH_DTYPE_BY_NAME[f.dtype]
            if f.name.endswith("_norm_w") or f.name == "idx_k_ln_w":
                w[f.name] = torch.ones(f.shape, device="cuda", dtype=dt)
            elif f.name in ("w_router_bias", "idx_k_ln_b"):
                w[f.name] = torch.zeros(f.shape, device="cuda", dtype=dt)
            else:
                w[f.name] = (torch.randn(n, generator=gen, device="cuda") * 0.06
                             ).to(dt).view(f.shape)
        return w

    w_ld = mk_weights(ld_fwd.wl)
    w_f = mk_weights(f_fwd.wl)
    x = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5
         ).to(torch.bfloat16)
    dy2 = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5
           ).to(torch.bfloat16)

    def mk_ctx(cl):
        return {f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype],
                                    device="cuda") for f in cl.fields}

    def mk_meta(kind):
        m_l = glm52_meta_layout(dims, kind)
        return {f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype],
                                    device="cuda") for f in m_l.fields}

    meta_ld = mk_meta("gml")
    meta_f = mk_meta("gmf")
    a1, a2 = mk_ctx(ld_fwd.cl), mk_ctx(f_fwd.cl)
    y1 = torch.empty_like(x)
    y2 = torch.empty_like(x)
    ld_fwd._forward(kctx, x, w_ld, y1, a1, extras={"meta": dict(meta_ld)})
    f_fwd._forward(kctx, y1, w_f, y2, a2,
                   extras={"meta": dict(meta_f),
                           "shared_idx": meta_ld["dsa_idx"]})

    gl_ld = grad_layout(ld_fwd.wl, dims.dtypes)
    gl_f = grad_layout(f_fwd.wl, dims.dtypes)
    dw_ld = {f.name: torch.zeros(f.shape, device="cuda", dtype=TORCH_DTYPE_BY_NAME[f.dtype])
             for f in gl_ld.fields}
    dw_f = {f.name: torch.zeros(f.shape, device="cuda", dtype=TORCH_DTYPE_BY_NAME[f.dtype])
            for f in gl_f.fields}
    dm = torch.empty(dims.tokens, dims.index_topk, dtype=torch.float32, device="cuda")
    dx1 = torch.empty_like(x)   # grad into y1 from the follower
    dx0 = torch.empty_like(x)
    # reverse order: follower bwd creates dM, leader bwd consumes it
    f_bwd._backward(kctx, dy2, a2, y1, w_f, dx1, dw_f, accum=False,
                    meta={"meta": meta_f, "shared_idx": meta_ld["dsa_idx"],
                          "_dm_view": dm, "_dm_create": True, "_kl_n": 2})
    ld_bwd._backward(kctx, dx1, a1, x, w_ld, dx0, dw_ld, accum=False,
                     meta={"meta": meta_ld, "_dm_view": dm, "_kl_n": 2})

    # ---- golden compose ----
    leaves_ld = {n: (t_.detach().clone().requires_grad_()
                     if n != "w_router_bias" else t_) for n, t_ in w_ld.items()}
    leaves_f = {n: (t_.detach().clone().requires_grad_()
                    if n != "w_router_bias" else t_) for n, t_ in w_f.items()}
    x_ref = x.clone().requires_grad_()
    golden = GoldenGlm52(dims=dims, n_layers=cfg.n_layers)
    golden._layer_ptr = 1          # gml leader sits at layer 1 in tiny
    golden._group_scores = None
    golden._group_mask = None
    # pin the runtime's selections/routings
    from dataflow.tasks.modules.dsa_reference import dsa_index_scores_reference
    from dataflow.tasks.modules.mla_reference import mla_qkv_reference
    from dataflow.tasks import ops as _ops

    h1_ref = _ops.rmsnorm_reference(x_ref, leaves_ld["attn_norm_w"])
    q_lora_ref, *_ = mla_qkv_reference(h1_ref, leaves_ld, dims)
    scores = dsa_index_scores_reference(h1_ref.detach(), q_lora_ref.detach(),
                                        leaves_ld, dims)
    mask = dsa_mask_from_idx(meta_ld["dsa_idx"].long(), dims, dims.tokens)
    golden._group_scores, golden._group_mask = scores, mask
    golden._pin_mask = True

    def golden_block(x_in, leaves, layer, route_ids):
        golden._layer_ptr = layer
        # reuse the pinned group state for both members
        gs, gm = golden._group_scores, golden._group_mask
        y_ref, aux = golden.block_forward(x_in, leaves, route_ids=route_ids)
        if golden._layer_ptr == layer + 1 and dims.role_of(layer) == "full":
            # block_forward recomputed scores/mask from its own graph for
            # the leader; RE-PIN the mask-based state to the runtime's
            golden._group_scores = golden._group_scores if gs is None else golden._group_scores
        return y_ref, aux

    # simpler: call block_forward directly with pinned state — the leader
    # call would overwrite the pinned mask with its own selection, so pin
    # AFTER by monkey-adjusting: run leader with its natural computation
    # but force the mask to the runtime's selection
    import dataflow.models.glm52_reference as GR

    orig_topk = GR.dsa_topk_reference
    try:
        GR.dsa_topk_reference = lambda s, k: meta_ld["dsa_idx"].long()
        golden._layer_ptr = 1
        y1_ref, aux1 = golden.block_forward(
            x_ref, leaves_ld, route_ids=meta_ld["route_ids"])
        y2_ref, aux2 = golden.block_forward(
            y1_ref, leaves_f, route_ids=meta_f["route_ids"])
    finally:
        GR.dsa_topk_reference = orig_topk
    (aux1 + aux2).backward(retain_graph=True)
    y2_ref.backward(dy2)

    errors = {"fwd:y1": rel_l2(y1, y1_ref.detach()), "fwd:y2": rel_l2(y2, y2_ref.detach()),
              "bwd:dx0": rel_l2(dx0, x_ref.grad)}
    for name, dw in (("ld", dw_ld), ("f", dw_f)):
        leaves = leaves_ld if name == "ld" else leaves_f
        for fname, g in dw.items():
            if fname == "w_router_bias":
                continue
            errors[f"{name}:d{fname}"] = rel_l2(g, leaves[fname].grad)
    bad = {k: round(v, 4) for k, v in errors.items() if v > 4e-2}
    assert not bad, bad


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
    from dataflow.training.families import resolve_family

    cfg = _tiny_cfg(train_indexer=False)
    fam = resolve_family(cfg)
    prog = fam.lower(cfg)
    assert not [o for o in prog.initial_objects if o.id.startswith("dM_")]
    assert not [oid for task in prog.task_by_id().values()
                for oid in (task.outputs and [o.id for o in task.outputs] or [])
                if str(oid).startswith("dM_")]
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL).assert_ok()


def test_glm52_dense_warmup_model_step():
    """Dense warm-up gate (sparse_mode=False) — model-step matches golden.
    IndexShare twist over dsv32's warm-up: followers deposit FULL-PREFIX
    rows into the group dM and the leader trains on the group centroid
    (p_own + dM)/N — the tiny config's N=3 group [1,2,3] and N=2 group
    [4,5] pin the averaging. Main wgrads are SKIPPED in the engine (dW
    zeroed); the frozen optimizer makes that invisible to param/state
    comparisons, which is exactly the point."""
    cfg = _tiny_cfg(sparse_mode=False)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     field_atol=_BIAS_ATOL).assert_ok()


def test_glm52_dense_warmup_freeze_and_movement():
    """Across a REAL engine warm-up step: every non-indexer field —
    embed/head/router-bias included, and EVERY follower field — is
    BIT-FROZEN; leader indexer fields move (full-prefix group KL live)."""
    import dataclasses

    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view
    from dataflow.tasks.layouts import (
        dsv3_moe_weight_layout,
        dsv32_dense_weight_layout,
        dsv32_moe_weight_layout,
    )
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    cfg = _tiny_cfg(sparse_mode=False)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=13)
    idx_fields = ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w")
    wl_of = {}
    for i in range(cfg.n_layers):
        kind = dims.kind_of(i)
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
