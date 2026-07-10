"""LBL-mode battery (aux refactor): the retained-inputs opt-in vs the
per-round default, plus the per-step counts machinery they share.

EQUAL where math says equal:
  - dW_router at ga=1 (f_round == f_global there) across the two code paths
    (dlogits injection vs the deferred contraction);
  - the noaux bias vector across ga partitionings (counts are scale-free
    and selections are per-token) — bit-exact;
  - the step counts across ga partitionings of the SAME tokens.
DIFFERENT where math says different (the silent-no-op guards):
  - the two modes' routers at ga>1 (per-round f's vs f_global);
  - the upstream weights at ga=1 (per-round injection back-propagates
    through h2; any deferred scheme necessarily cannot).
GA-INVARIANCE, isolated: under SGD (linear), the aux-induced router delta
  W(alpha) - W(0) equals -lr * g_aux exactly, with the CE part cancelling —
  retained's delta is ga-invariant, per-round's demonstrably is not. (The
  naive full-weight engine(ga4)==engine(ga1) comparison is ill-posed: the
  engine SUMS per-round mean losses, so the CE gradient itself scales
  with the round count, and AdamW is nonlinear on top.)
And: expert_counts_overall is monotone across steps; a recompute-everything
  plan produces the SAME counts as save-all (recompute has no Aux edge —
  the double-count negative test).
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.fake import FakeBackend  # noqa: E402
from dataflow.runtime.engine import uniform_segments  # noqa: E402
from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view  # noqa: E402
from dataflow.tasks.modules.moe.spec import moe_aux_layout  # noqa: E402
from dataflow.training.families import resolve_family  # noqa: E402
from dataflow.training.models.dsv3 import ShapedDsv3Config  # noqa: E402
from dataflow.training.models.olmoe import ShapedOlmoeConfig  # noqa: E402
from dataflow.training.planning import plan_program  # noqa: E402
from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402

# one step's tokens, partitioned two ways: ga=1 x (b=4, s=32) == ga=4 x (b=1, s=32)
T_STEP = 128
SEQ = 32


def olmoe_cfg(ga: int, *, aux: float, retained: bool, opt="sgd"):
    return replace(
        ShapedOlmoeConfig.tiny(), seq_len=SEQ, batch=T_STEP // (SEQ * ga),
        grad_accum_rounds=ga, aux_coef=aux, lbl_retained_inputs=retained,
        opt_policy=opt,
    )


def run_step(cfg, seed=7, *, steps_in_program=1, recompute_levels=None,
             pin_dw0=False):
    """One engine execution from a seeded init with the step tokens WRITTEN
    EXPLICITLY (the same master arrays partitioned across rounds), so
    different ga configs consume identical tokens. Returns
    {object_id: {field: cpu tensor}} for the persistent objects (plus
    dW_0_0's fields when ``pin_dw0`` — gradient-level observation without
    optimizer/bf16-weight-quantum masking)."""
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    program = fam.lower(replace(cfg, num_steps=steps_in_program),
                        recompute_levels=recompute_levels)
    if pin_dw0:
        program = replace(program, final_locations={
            **program.final_locations, "dW_0_0": "backing",
            "dW_0_1": "backing"})
    planned = plan_program(program, fast_memory_capacity=96 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=seed)
    master = torch.Generator().manual_seed(99)
    for s in range(steps_in_program):
        tok = torch.randint(0, cfg.vocab_size, (T_STEP,), generator=master,
                            dtype=torch.int32)
        tgt = torch.randint(0, cfg.vocab_size, (T_STEP,), generator=master,
                            dtype=torch.int32)
        per_round = T_STEP // cfg.grad_accum_rounds
        for r in range(cfg.grad_accum_rounds):
            lo = r * per_round
            torch_view(values[f"tokens_{s}_{r}"], (per_round,),
                       torch.int32).copy_(tok[lo:lo + per_round])
            torch_view(values[f"targets_{s}_{r}"], (per_round,),
                       torch.int32).copy_(tgt[lo:lo + per_round])
    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program),
                  "step": 0},
    )
    out: dict = {}
    fl = None
    from dataflow.training.models import dsv3 as dsv3_mod
    from dataflow.training.models import olmoe as olmoe_mod
    mod = olmoe_mod if type(cfg).__name__ == "ShapedOlmoeConfig" else dsv3_mod
    _, fl = mod.family_layouts(cfg)
    if pin_dw0:
        from dataflow.tasks.layouts import grad_layout
        for li in (0, 1):
            gl = grad_layout(fl.layers[li].weights, fam.dims_of(cfg).dtypes,
                             layer=li,
                             opt_policy=getattr(fam.dims_of(cfg),
                                                "opt_policy", None))
            grec = result.objects.get(f"dW_0_{li}")
            gslot = grec.backing or grec.fast
            out[f"dW_0_{li}"] = {
                f.name: torch_view(gslot.buffer, f.shape,
                                   TORCH_DTYPE_BY_NAME[f.dtype],
                                   offset_bytes=f.offset_bytes).clone().cpu()
                for f in gl.fields}
    for i in range(cfg.n_layers):
        rec = result.objects.get(f"W_{i}")
        slot = rec.backing or rec.fast
        out[f"W_{i}"] = {
            f.name: torch_view(slot.buffer, f.shape,
                               TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes).clone().cpu()
            for f in fl.layers[i].weights.fields}
        if fl.layers[i].aux is not None:
            arec = result.objects.get(f"Aux_{i}")
            aslot = arec.backing or arec.fast
            out[f"Aux_{i}"] = {
                f.name: torch_view(aslot.buffer, f.shape,
                                   TORCH_DTYPE_BY_NAME[f.dtype],
                                   offset_bytes=f.offset_bytes).clone().cpu()
                for f in moe_aux_layout(fam.dims_of(cfg), fam.dims_of(cfg).moe).fields}
    result.close()
    for buf in values.values():
        backend.free(buf)
    return out


def test_retained_structure_and_default_stability():
    fam = resolve_family(olmoe_cfg(4, aux=0.02, retained=True))
    program = fam.lower(olmoe_cfg(4, aux=0.02, retained=True))
    by_id = {t.id: t for t in program.tasks}
    last_bwd = by_id["block_bwd_0_3_0"]
    packs = [i for i in last_bwd.inputs if i.startswith("AuxTemp_")]
    assert sorted(packs) == [f"AuxTemp_0_{r}_0" for r in range(4)], packs
    early_bwd = by_id["block_bwd_0_1_0"]
    assert [i for i in early_bwd.inputs if i.startswith("AuxTemp_")] == \
        ["AuxTemp_0_1_0"]
    # knob off: no cross-round inputs, no retained fields (hash tripwire
    # pins the bytes; this pins the reason)
    off = resolve_family(olmoe_cfg(4, aux=0.02, retained=False)).lower(
        olmoe_cfg(4, aux=0.02, retained=False))
    off_last = {t.id: t for t in off.tasks}["block_bwd_0_3_0"]
    assert [i for i in off_last.inputs if i.startswith("AuxTemp_")] == \
        ["AuxTemp_0_3_0"]


@pytest.mark.gpu
def test_modes_agree_on_router_at_ga1_and_differ_upstream():
    """At ga=1 f_round == f_global: the two code paths must produce the same
    router (within cross-path rounding), while the UPSTREAM weights differ
    (per-round back-propagates the aux term through h2; retained cannot)."""
    per_round = run_step(olmoe_cfg(1, aux=25.0, retained=False), pin_dw0=True)
    retained = run_step(olmoe_cfg(1, aux=25.0, retained=True), pin_dw0=True)
    for i in (0, 1):
        d_router = rel_l2(per_round[f"W_{i}"]["w_router"],
                          retained[f"W_{i}"]["w_router"])
        assert d_router < 5e-3, (i, d_router)
    # the upstream evidence lives at the GRADIENT level (bf16 weights at
    # tiny lr round the alpha-scale effect away): per-round's injection
    # reaches layer 0's dW through layer 1's dx; retained's cannot.
    d_up = max(rel_l2(per_round["dW_0_0"][n], retained["dW_0_0"][n])
               for n in ("wq", "w13_experts", "ffn_norm_w"))
    assert d_up > 0.0, "upstream gradients identical -> retained knob inert?"
    # the TOP aux layer's router GRADIENT is the clean ga=1 equality (its
    # dy carries no aux from above; layer 0's total router grad also
    # differs via layer 1's upstream term, by design)
    assert rel_l2(per_round["dW_0_1"]["w_router"],
                  retained["dW_0_1"]["w_router"]) < 5e-3


@pytest.mark.gpu
def test_modes_differ_on_router_at_ga4():
    """The silent-no-op guard: with 4 rounds the per-round f's differ from
    f_global, so the two modes' routers must NOT coincide."""
    per_round = run_step(olmoe_cfg(4, aux=25.0, retained=False))
    retained = run_step(olmoe_cfg(4, aux=25.0, retained=True))
    gaps = [rel_l2(per_round[f"W_{i}"]["w_router"],
                   retained[f"W_{i}"]["w_router"]) for i in (0, 1)]
    assert max(gaps) > 1e-6, gaps


@pytest.mark.gpu
def test_retained_router_delta_is_ga_invariant_per_round_is_not():
    """Under SGD the aux-induced router delta isolates -lr*g_aux exactly.
    Retained mode's delta is invariant to the round partitioning of the
    same step tokens; the per-round default's is not."""
    def router_delta(ga, retained):
        on = run_step(olmoe_cfg(ga, aux=25.0, retained=retained))
        off = run_step(olmoe_cfg(ga, aux=0.0, retained=retained))
        return {i: on[f"W_{i}"]["w_router"].float() -
                   off[f"W_{i}"]["w_router"].float() for i in (0, 1)}

    ret1, ret4 = router_delta(1, True), router_delta(4, True)
    per1, per4 = router_delta(1, False), router_delta(4, False)
    for i in (0, 1):
        ret_gap = rel_l2(ret1[i], ret4[i])
        per_gap = rel_l2(per1[i], per4[i])
        assert ret_gap < 5e-2, (i, ret_gap)
        assert per_gap > 2 * ret_gap, (i, per_gap, ret_gap)


@pytest.mark.gpu
def test_counts_ga_invariant_and_sum_exact():
    """Same tokens, different partitioning: per-token selections are
    identical, so the step counts agree and always sum to T_STEP * top_k."""
    c1 = run_step(olmoe_cfg(1, aux=0.0, retained=False))
    c4 = run_step(olmoe_cfg(4, aux=0.0, retained=False))
    for i in (0, 1):
        a1 = c1[f"Aux_{i}"]["expert_counts_current_step"]
        a4 = c4[f"Aux_{i}"]["expert_counts_current_step"]
        assert int(a1.sum()) == T_STEP * 2, a1
        assert torch.equal(a1, a4), (i, a1, a4)
        assert torch.equal(c1[f"Aux_{i}"]["expert_counts_overall"].int(), a1)


@pytest.mark.gpu
def test_counts_overall_monotone_across_steps():
    two = run_step(olmoe_cfg(4, aux=0.0, retained=False), steps_in_program=2)
    for i in (0, 1):
        cur = two[f"Aux_{i}"]["expert_counts_current_step"]
        overall = two[f"Aux_{i}"]["expert_counts_overall"]
        assert int(cur.sum()) == T_STEP * 2
        assert int(overall.sum()) == 2 * T_STEP * 2      # both steps
        assert bool((overall.int() >= cur).all())


@pytest.mark.gpu
def test_recompute_never_double_counts():
    cfg = olmoe_cfg(4, aux=0.0, retained=False)
    fam = resolve_family(cfg)
    program = fam.lower(cfg)
    levels = {o: 1 for t in program.tasks for o in [x.id for x in t.outputs]
              if o.startswith("A_")}
    save_all = run_step(cfg)
    rc_all = run_step(cfg, recompute_levels=levels)
    for i in (0, 1):
        sa = save_all[f"Aux_{i}"]["expert_counts_current_step"]
        rc = rc_all[f"Aux_{i}"]["expert_counts_current_step"]
        # DOUBLE counting would inflate the sum (each recompute re-adding
        # T*K per round) — the sum must stay EXACTLY one step's assignments.
        assert int(rc.sum()) == T_STEP * 2, rc
        # across two DIFFERENT plans the no-ctx forward variant rounds
        # differently, so near-tie selections may flip a token or two —
        # bounded flip noise, never systematic drift
        assert int((sa - rc).abs().sum()) <= 8, (i, sa, rc)


@pytest.mark.gpu
def test_noaux_bias_ga_invariant_bit_exact():
    """Counts are scale-free and per-token: the sign rule gives a BIT-EQUAL
    fp32 bias whichever way the step's tokens are partitioned."""
    def dsv3_cfg(ga):
        return replace(ShapedDsv3Config.tiny(), seq_len=SEQ,
                       batch=T_STEP // (SEQ * ga), grad_accum_rounds=ga,
                       aux_coef=0.0, bias_update_speed=1e-3, opt_policy="sgd")

    b1 = run_step(dsv3_cfg(1))
    b4 = run_step(dsv3_cfg(4))
    moe_layers = [i for i in range(3) if "w_router_bias" in b1.get(f"W_{i}", {})]
    assert moe_layers, "no moe layers found"
    for i in moe_layers:
        assert torch.equal(b1[f"W_{i}"]["w_router_bias"],
                           b4[f"W_{i}"]["w_router_bias"]), i
        assert bool(b1[f"W_{i}"]["w_router_bias"].abs().gt(0).any()), i
