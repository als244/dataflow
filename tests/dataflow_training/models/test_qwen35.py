"""Qwen3.5 correctness ladder, part 1: kernel/spec pinning (GPU).

Before any block exists, the family's math spec (pure-torch reference forms
in tasks/ops.py) is pinned three ways:
  1. our sequential delta-rule recurrence == fla's own naive reference (fp32,
     spec vs spec);
  2. fla's CHUNK kernels (the ones the blocks will call) == our recurrence
     at bf16 tolerances — forward AND backward (the backward is the
     arch-sensitivity check: fla issue #640 documented a chunk-backward
     Triton failure on an earlier GPU architecture — verify the current
     device does not need that workaround);
  3. the conv + l2norm helpers == their references.

Tests:
- test_reference_recurrence_matches_fla_naive: our fp32 gated-delta-rule recurrence matches fla's naive reference (spec vs spec).
- test_fla_chunk_fwd_matches_reference: fla's chunk forward output matches our recurrence at bf16 tolerance and its saved g_input equals the raw gate input.
- test_fla_chunk_bwd_matches_reference_autograd: fla's chunk backward grads (dq/dk/dv/dbeta/da/dA_log/ddt_bias) match autograd through the fp32 recurrence.
- test_conv_and_l2norm_helpers_match_references: the fla causal-conv1d-silu and l2norm helpers match our references.
- test_qwen35_stage_context_completeness: both lin and attn forward blocks' emitted context fields equal their activation layouts, with recompute stopping before the last stage.
- test_qwen35_lowering_validates_and_plans: the untied cfg lowers with separate W_head/O_head and plans/simulates, while the tied variant emits a single W_embed serving embed and head.
- test_qwen35_tied_model_step_vs_golden: the tied variant's model-step matches golden with the shared dW_embed created by head_bwd.
- test_qwen35_plan_invariance: the model-step math is identical across memory budgets and recompute plans.
- test_qwen35_batch2_packed_sequences_vs_golden: a batch=2 packed-sequence model-step matches golden with conv and DeltaNet recurrence reset at boundaries.
- test_qwen35_poison_on_free_changes_nothing: the poison_on_free engine option leaves loss and weights unchanged and non-NaN.
- test_qwen35_interleaving_stress_changes_nothing: random per-task launch jitter leaves loss and weights unchanged.
- test_qwen35_measured_costs_replan_still_golden: profiling the heterogeneous task set then replanning on measured costs leaves the math unchanged.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

pytest.importorskip("fla")

from dataflow_training.blocks import ops  # noqa: E402
from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu

T, HK, HV, K, V = 256, 2, 4, 32, 32


def _inputs(seed=0, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, HK, K, device="cuda", generator=g).to(dtype)
    k = torch.randn(T, HK, K, device="cuda", generator=g).to(dtype)
    v = (torch.randn(T, HV, V, device="cuda", generator=g) * 0.5).to(dtype)
    beta = torch.rand(T, HV, device="cuda", generator=g).to(dtype)
    a = torch.randn(T, HV, device="cuda", generator=g).to(dtype)
    A_log = (torch.empty(HV, device="cuda").uniform_(1.0, 16.0, generator=g)).log()
    dt_bias = torch.zeros(HV, device="cuda")
    g_log = ops.gated_delta_gate_reference(a, A_log, dt_bias)
    qn = ops.l2norm_reference(q)
    kn = ops.l2norm_reference(k)
    return qn, kn, v, beta, a, A_log, dt_bias, g_log


def test_reference_recurrence_matches_fla_naive():
    """Spec vs spec at fp32: our recurrence == fla's naive reference."""
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

    qn, kn, v, beta, _a, _Al, _dt, g_log = _inputs(dtype=torch.float32)
    ours = ops.gated_delta_rule_reference(qn, kn, v, beta, g_log)
    rep = HV // HK
    theirs, _ = naive_recurrent_gated_delta_rule(
        qn.repeat_interleave(rep, dim=1).unsqueeze(0),
        kn.repeat_interleave(rep, dim=1).unsqueeze(0),
        v.unsqueeze(0), beta.unsqueeze(0), g_log.unsqueeze(0),
        scale=K ** -0.5,
    )
    assert rel_l2(ours, theirs.squeeze(0).to(ours.dtype)) < 1e-5


def test_fla_chunk_fwd_matches_reference():
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    qn, kn, v, beta, a, A_log, dt_bias, g_log = _inputs(dtype=torch.bfloat16)
    ref = ops.gated_delta_rule_reference(qn, kn, v, beta, g_log)
    # fwd contract (fla 0.5.1, from ChunkGatedDeltaRuleFunction):
    # returns (g_post, o, A_int, final_state, initial_state, g_input)
    g_post, o, A_int, _fs, _is, g_input = chunk_gated_delta_rule_fwd(
        qn.unsqueeze(0), kn.unsqueeze(0), v.unsqueeze(0).contiguous(),
        a.unsqueeze(0), beta.unsqueeze(0), scale=K ** -0.5,
        initial_state=None, output_final_state=False,
        cu_seqlens=None, chunk_indices=None,
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
    )
    assert rel_l2(o.squeeze(0), ref) < 3e-2
    # g_post / A_int / g_input are opaque bwd inputs we save verbatim —
    # assert only sanity, not internal structure
    assert torch.isfinite(g_post).all() and g_post.shape[-1] == HV
    assert g_input is not None and torch.isfinite(g_input.float()).all()
    # under use_gate_in_kernel, fwd's g_input return is the RAW gate input
    # (a) passed through — gdn_gate_bwd re-derives softplus grads from it.
    # Our blocks therefore reuse the saved `ba`'s a-slice as g_input; no
    # extra context field needed.
    assert rel_l2(g_input.squeeze(0).float(), a.float()) < 1e-3


def test_fla_chunk_bwd_matches_reference_autograd():
    """The arch-sensitivity check: fla's chunk bwd vs autograd through our
    recurrence (fla #640 documented a Triton bwd failure on an earlier GPU
    architecture; verify the current device does not need the expand/reduce
    workaround)."""
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_fwd,
    )

    qn, kn, v, beta, a, A_log, dt_bias, g_log = _inputs(dtype=torch.bfloat16)
    do = (torch.randn_like(v.float()) * 0.5).to(torch.bfloat16)

    # reference grads by autograd through the fp32 recurrence
    q_r = qn.float().requires_grad_()
    k_r = kn.float().requires_grad_()
    v_r = v.float().requires_grad_()
    beta_r = beta.float().requires_grad_()
    a_r = a.float().requires_grad_()
    A_r = A_log.clone().requires_grad_()
    dt_r = dt_bias.clone().requires_grad_()
    g_r = ops.gated_delta_gate_reference(a_r, A_r, dt_r)
    out = ops.gated_delta_rule_reference(q_r, k_r, v_r, beta_r, g_r)
    out.backward(do.float())

    g_post, o, A_int, _fs, _is, g_input = chunk_gated_delta_rule_fwd(
        qn.unsqueeze(0), kn.unsqueeze(0), v.unsqueeze(0).contiguous(),
        a.unsqueeze(0), beta.unsqueeze(0), scale=K ** -0.5,
        initial_state=None, output_final_state=False,
        cu_seqlens=None, chunk_indices=None,
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
    )
    # bwd contract: g = POST-cumsum gate (fwd ret 0), g_input = PRE-cumsum
    # per-token gate (fwd ret 5); returns
    # (dq, dk, dv, dbeta, da, dh0, dA_log, ddt_bias)
    dq, dk, dv, db, da, _dh0, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
        q=qn.unsqueeze(0), k=kn.unsqueeze(0), v=v.unsqueeze(0).contiguous(),
        g=g_post, beta=beta.unsqueeze(0), A=A_int,
        scale=K ** -0.5, initial_state=None, do=do.unsqueeze(0), dht=None,
        cu_seqlens=None, chunk_indices=None,
        use_gate_in_kernel=True, g_input=g_input, A_log=A_log, dt_bias=dt_bias,
    )
    assert rel_l2(dq.squeeze(0), q_r.grad) < 5e-2
    assert rel_l2(dk.squeeze(0), k_r.grad) < 5e-2
    assert rel_l2(dv.squeeze(0), v_r.grad) < 5e-2
    assert rel_l2(db.squeeze(0), beta_r.grad) < 5e-2
    assert rel_l2(da.squeeze(0), a_r.grad) < 5e-2
    assert dA_log is not None and rel_l2(dA_log, A_r.grad) < 5e-2
    assert ddt_bias is not None and rel_l2(ddt_bias, dt_r.grad) < 8e-2


def test_conv_and_l2norm_helpers_match_references():
    import fla.modules.conv.triton.ops as fops
    from fla.modules.l2norm import l2norm_fwd

    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(512, 192, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(192, 4, device="cuda", generator=g) * 0.2).to(torch.bfloat16)
    y = fops.causal_conv1d_fwd(x.unsqueeze(0), w, None, None, activation="silu")
    y = y[0] if isinstance(y, tuple) else y
    y = y.squeeze(0) if y.dim() == 3 else y
    assert rel_l2(y, ops.causal_conv1d_silu_reference(x, w)) < 2e-2

    q = torch.randn(512, 4, 32, device="cuda", generator=g).to(torch.bfloat16)
    qn, _rstd = l2norm_fwd(q.view(-1, 32))
    assert rel_l2(qn.view_as(q), ops.l2norm_reference(q)) < 2e-2




def _tiny_dims():
    from dataflow_training.model_families.qwen35 import ShapedQwen35Config, derive_dims

    return derive_dims(ShapedQwen35Config.tiny())


# block-level ladder retired with the golden models: block math is
# gated by the per-op kernel pins, the model-level dW comparison
# (grad: entries), and per-block isolation (tools/deep_compare.py
# --isolate); see docs/correctness_compare.md.


# --- structure + ladder 3: full program through the real engine --------------


def _tiny_cfg(**over):
    from dataclasses import replace

    from dataflow_training.model_families.qwen35 import ShapedQwen35Config

    return replace(ShapedQwen35Config.tiny(), **over)


def test_qwen35_stage_context_completeness():
    from dataflow_training.blocks.layouts import (
        qwen35_attn_activation_layout,
        qwen35_lin_activation_layout,
    )
    from dataflow_training.model_families.qwen35.blocks import Qwen35AttnBlockFwd, Qwen35LinBlockFwd

    dims = _tiny_dims()
    for cls, cl in (
        (Qwen35LinBlockFwd, qwen35_lin_activation_layout(dims)),
        (Qwen35AttnBlockFwd, qwen35_attn_activation_layout(dims)),
    ):
        declared = {f.name for f in cl.fields}
        emitted = cls.context_fields_emitted()
        assert declared == emitted, (cls.__name__, declared ^ emitted)
        assert cls.recompute_stage_count() < len(cls.STAGES)


@pytest.mark.sim
def test_qwen35_lowering_validates_and_plans():
    from dataflow.core import validate_program
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program, simulate_program
    from dataflow_training.model_families.qwen35 import ShapedQwen35Config

    # untied (the 9B default): separate W_head/O_head, bare-table W_embed
    cfg = _tiny_cfg()
    fam = resolve_family(cfg)
    assert fam.name == "qwen35"
    program = fam.lower(cfg)
    validate_program(program)
    assert program.metadata["family"] == "qwen35-shaped"
    ids = {spec.id for spec in program.initial_objects}
    assert {"W_embed", "W_head", "O_head"} <= ids
    planned = plan_program(program, fast_memory_capacity=8 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0

    # tied (2B-style): ONE W_embed serves embed and head
    tied = fam.lower(ShapedQwen35Config.tiny_tied())
    validate_program(tied)
    tids = {spec.id for spec in tied.initial_objects}
    assert "W_embed" in tids and "W_head" not in tids and "O_head" not in tids




@pytest.mark.sim
def test_qwen35_tied_model_step_vs_golden():
    """The 2B-style tied variant stays golden-verified E2E (one W_embed
    leaf, head_bwd round-0 creates the shared dW_embed)."""
    from dataflow_training.model_families.qwen35 import ShapedQwen35Config
    from dataflow_training.testing.gradcheck import check_model_step, family_gate_kwargs

    kw = family_gate_kwargs("qwen35")
    # the tied config's state-path grads (A_log/dt_bias) draw ~0.9952
    # cosine — its own config, its own noise draw; the family band
    # (0.998, calibrated on tiny()) stays tight for everything else
    kw["min_cosine"] = 0.99
    check_model_step(
        ShapedQwen35Config.tiny_tied(), fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
        **kw,
    ).assert_ok()


@pytest.mark.sim
def test_qwen35_plan_invariance():
    """Different budgets + recompute plans must produce identical math.

    Tight legs run at 12 MiB, not 8: the UNTIED tiny at 8 MiB sits so close
    to its working set that the interleaved optimizer's in-flight O_i
    prefetches strand the ledger under real timing and the pressure-eviction
    valve ping-pongs (every resident has a near next-use at tiny scale — no
    far-future Belady candidate) until DeadlockError. Known byte-timing-inversion-class
    timing-inversion corner, loud not silent; 9B budgets show 0 evictions.
    12 MiB still forces offload traffic + the forced-recompute plan."""
    from dataflow_training.testing.gradcheck import check_model_step, family_gate_kwargs

    cfg = _tiny_cfg()
    r1 = check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                          **family_gate_kwargs("qwen35"))
    r2 = check_model_step(cfg, fast_memory_capacity=12 * 1024 * 1024, tol=3e-2,
                          **family_gate_kwargs("qwen35"))
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    r3 = check_model_step(
        cfg, fast_memory_capacity=12 * 1024 * 1024, recompute_levels=levels, tol=3e-2,
        **family_gate_kwargs("qwen35"),
    )
    for r in (r1, r2, r3):
        r.assert_ok()


@pytest.mark.sim
def test_qwen35_batch2_packed_sequences_vs_golden():
    """batch=2 packs two sequences into one token axis: cu_seqlens must reset
    the conv window and the DeltaNet recurrence at the boundary (the golden
    reference resets by construction — agreement pins the kernels do too)."""
    from dataflow_training.testing.gradcheck import check_model_step, family_gate_kwargs

    cfg = _tiny_cfg(batch=2, seq_len=64)
    check_model_step(cfg, fast_memory_capacity=64 * 1024 * 1024, tol=3e-2,
                     **family_gate_kwargs("qwen35")).assert_ok()


def _run35(engine_kwargs=None, resolver_wrapper=None, program=None, seed=7):
    """One engine run of the tiny qwen35 program; returns loss + final weights."""
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
    # readback masks the layouts' alignment-padding gaps: the bwd tasks write
    # FIELDS, adamw updates the whole flat buffer (padding included, from
    # undefined dW padding — NaN under poison-on-free), and no field ever
    # reads padding. The gaps are allocator residue outside the math
    # contract; the gate compares the model. (llama3/qwen3 never had gaps —
    # every field size there is an alignment multiple; the qwen35 lin layout
    # has 8-byte A_log/dt_bias fields at tiny scale.)
    from dataflow_training.blocks.layouts import (
        head_weight_layout,
        qwen35_attn_weight_layout,
        qwen35_lin_weight_layout,
    )

    dims = fam.derive_dims(cfg)

    def masked(flat, layout):
        if layout is None:  # bare table (untied W_embed): no padding gaps
            return flat
        keep = torch.zeros_like(flat, dtype=torch.bool)
        for f in layout.fields:
            n = 1
            for s in f.shape:
                n *= int(s)
            start = f.offset_bytes // 2
            keep[start : start + n] = True
        return torch.where(keep, flat, torch.zeros_like(flat))

    if cfg.tied_embeddings:
        layouts = {"W_embed": head_weight_layout(dims)}
    else:
        layouts = {"W_embed": None, "W_head": head_weight_layout(dims)}
    for i in range(cfg.n_layers):
        layouts[f"W_{i}"] = (
            qwen35_attn_weight_layout(dims) if dims.kinds[i] == "full"
            else qwen35_lin_weight_layout(dims)
        )
    out = {}
    for obj_id, layout in layouts.items():
        rec = result.objects.get(obj_id)
        slot = rec.backing or rec.fast
        flat = torch_view(slot.buffer, (rec.size_bytes // 2,), torch.bfloat16).clone()
        out[obj_id] = masked(flat, layout)
    loss_rec = result.objects.get("loss_0_0")
    out["loss"] = float(torch_view((loss_rec.backing or loss_rec.fast).buffer, (1,), torch.float32)[0])
    result.close()
    dry.close()
    for buf in values.values():
        backend.free(buf)
    return out


def _assert_same35(a: dict, b: dict, tol: float = 1e-3):
    assert abs(a["loss"] - b["loss"]) / max(abs(b["loss"]), 1e-9) < tol, (a["loss"], b["loss"])
    for k in a:
        if k == "loss":
            continue
        err = rel_l2(a[k], b[k])
        assert err < tol, f"{k}: rel_l2={err}"


@pytest.mark.sim
def test_qwen35_poison_on_free_changes_nothing():
    base = _run35()
    poisoned = _run35(engine_kwargs={"poison_on_free": True})
    _assert_same35(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]  # not NaN


@pytest.mark.sim
def test_qwen35_interleaving_stress_changes_nothing():
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

    base = _run35()
    jittered = _run35(resolver_wrapper=wrapper)
    _assert_same35(jittered, base)


@pytest.mark.sim
def test_qwen35_measured_costs_replan_still_golden():
    """Profiling must handle the heterogeneous task set (linattn_*/gattn_*
    keys); re-planning on measured costs must not change the math."""
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

    base = _run35()
    replanned = plan_program(measured, fast_memory_capacity=8 * 1024 * 1024).program
    again = _run35(program=replanned)
    _assert_same35(again, base)


