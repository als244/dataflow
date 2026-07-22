"""FreezePlan: spec composer, derivation, surgery, and E2E parity.

The freeze feature's contract (docs/notes/handling_frozen_plan.md):
frozen params are SPECIFIED once (the optimizer policy — the freeze()
composer is its front door) and every structural consequence DERIVES:
dW/O shrink to trainable fields, fully-frozen layers lose their
backward tasks when nothing below trains (truncation) or keep a
dgrad-only pass-through (guards-first), A saves only where a backward
will read it, and the dy chain stops at the deepest trainable depth.

E2E gates run the REAL engine against the isolated reference twin.
Fully-frozen weight objects deliberately carry NO optimizer task, so
check_model_step's gradient extraction (one optimizer task per weight
object) cannot run those configs; the freeze gates instead use the
local assert_frozen_model_step runner: the twin steps only the
mirrored trainable allowlist, certifying loss + trainable-param
parity while frozen params are checked BIT-IDENTICAL to the shared
init on both legs (the freeze contract itself). Partial-field freezes
keep every optimizer task and still ride check_model_step.

Tests:
- test_no_freeze_derives_none: no freeze and partial field/pair freezes derive no FreezePlan (dW shrinks via layouts with no surgery).
- test_truncated_prefix_plan: freezing layer 0 plus the embedding derives a truncated-then-train plan that drops the boundary backward, dy, recv_dy, and ctx below it and keeps the head trainable.
- test_passthrough_plan: freezing layer 0 with a trainable embedding derives a passthrough plan that keeps dgrad emission, dy production, and ctx save so grads reach the embedding.
- test_all_frozen_ce_rejected: constructing a FreezePlan with nothing trainable raises ValueError.
- test_composer_semantics: the freeze() composer marks layer, field, pair, and embed targets frozen while other fields follow the base policy rules.
- test_plan_repr_compact: the FreezePlan repr shows a compact regime-and-objective summary.
- test_model_step_truncated_prefix: with layer 0 and embed frozen the layer-0 backward, its A, its dW, dy_embed, and embed_bwd are absent and the engine step matches the frozen-aware twin.
- test_model_step_passthrough: with layer 0 frozen but the embedding trainable, layer 0 runs dgrad-only (no dW_0), dW_embed exists, and the engine matches the twin.
- test_model_step_partial_fields: freezing wq/wk fleetwide needs no surgery, shrinks dW/O to the trainable fields, skips the frozen wgrads, and still matches the golden.
- test_model_step_truncated_ga2: two-round grad accumulation over the truncated program matches a twin summing the same rounds, with per-round loss, trainable params, and frozen-bit identity all checked.
- test_model_step_pair_freeze: freezing different fields on different layers gives per-layer dW sizes matching the policy and the engine matches the golden.
- test_all_families_truncated_prefix_lowers: every model family lowers a layer-0+embed truncated program (layer-0 backward, A, and embed_bwd gone) that still validates.
- test_model_step_truncated_olmoe: truncating a frozen layer 0 (router and experts) plus the embedding in a MoE family still matches the twin driving the same load-balancing channel through the engine.
- test_train_indexer_unified_into_policy: train_indexer=False removes the five indexer fields from the grad layout and the lowered dW_0_0 size equals the shrunken layout.
- test_model_step_frozen_head: freezing the LM head keeps head_loss but drops dW_head/O_head and the head wgrad GEMM, and the engine matches the twin with head plus final_norm frozen.
- test_bench_default_stream_semantics: the bench default data stream yields shifted next-token targets with last-position wrap, rounds unique within a step but repeated across steps, and decoupled mode's independent-yet-fixed targets.
- test_initial_values_refill_identity: refilling initial_values via into= is byte-identical to a fresh seeded init.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch  # noqa: F401

from dataflow_training.blocks.optim import freeze
from dataflow_training.lowering.freeze_plan import FreezePlan, derive_freeze_plan
from dataflow_training.model_families.llama3 import (
    ShapedLlamaConfig,
    derive_dims,
    lower_llama3,
)
from dataflow_training.testing.gradcheck import check_model_step

FIELDS = ("attn_norm_w", "wq", "wk", "wv", "wo", "ffn_norm_w",
          "w1", "w3", "w2")


def _plan(cfg):
    d = derive_dims(cfg)
    return derive_freeze_plan(
        d, cfg.n_layers, lambda i: FIELDS,
        tied_embeddings=bool(getattr(cfg, "tied_embeddings", False)))


def _tiny(**over):
    return dataclasses.replace(ShapedLlamaConfig.tiny(), **over)


# ---------------------------------------------------------------- analyzer


def make_default_stream_probe(*, vocab: int, tokens: int, seq_len: int,
                               rounds: int, seed: int,
                               decoupled: bool = False):
    """Test seam: the bench default_stream's data contract, minus the
    engine. Returns (stream, seq_len); see test_bench_default_stream_
    semantics."""
    import torch as _torch

    gen = _torch.Generator().manual_seed(seed + 1)
    fixed: dict[int, tuple] = {}

    def stream(k: int):
        r = k % max(1, rounds)
        if r not in fixed:
            toks = _torch.randint(0, vocab, (tokens,), generator=gen,
                                  dtype=_torch.int32)
            if decoupled:
                tgts = _torch.randint(0, vocab, (tokens,), generator=gen,
                                      dtype=_torch.int32)
            else:
                tgts = (toks.view(-1, seq_len).roll(-1, dims=1)
                        .reshape(-1).contiguous())
            fixed[r] = (toks, tgts)
        return fixed[r]

    return stream, seq_len


def test_no_freeze_derives_none():
    assert _plan(_tiny()) is None
    # partial freezes are structurally invisible too: dW shrinks via
    # layouts, no surgery — the byte-identity fast path
    assert _plan(_tiny(opt_policy=freeze(fields=("wq",)))) is None
    assert _plan(_tiny(opt_policy=freeze(pairs=(("wo", 1),)))) is None


def test_truncated_prefix_plan():
    plan = _plan(_tiny(opt_policy=freeze(layers=(0,), embed=True)))
    assert plan.regimes == ("truncated", "train")
    assert plan.emit_bwd == (False, True)
    assert plan.produce_dy == (False, False)   # nothing below layer 1 trains
    assert plan.recv_dy == (False, True)
    assert plan.save_ctx == (False, True)
    assert not plan.embed_trainable and plan.head_trainable


def test_passthrough_plan():
    plan = _plan(_tiny(opt_policy=freeze(layers=(0,))))
    assert plan.regimes == ("passthrough", "train")
    assert plan.emit_bwd == (True, True)       # dgrads must reach embed
    assert plan.produce_dy == (True, True)   # embed trains below layer 1
    assert plan.save_ctx == (True, True)
    assert plan.embed_trainable


def test_all_frozen_ce_rejected():
    with pytest.raises(ValueError, match="nothing"):
        FreezePlan(n_layers=1, regimes=("truncated",), emit_bwd=(False,),
                   recv_dy=(False,), produce_dy=(False,),
                   save_ctx=(False,), embed_trainable=False,
                   head_trainable=False)


def test_composer_semantics():
    pol = freeze(base="muon", layers=(0,), fields=("wo",),
                 pairs=(("w1", 1),), embed=True)
    assert pol.for_field("wq", 0, (8, 8)) == "frozen"      # layer freeze
    assert pol.for_field("wo", 1, (8, 8)) == "frozen"      # field freeze
    assert pol.for_field("w1", 1, (8, 8)) == "frozen"      # pair freeze
    assert pol.for_field("embed.w", None, (16, 8)) == "frozen"
    assert pol.for_field("wq", 1, (8, 8)) == "muon"        # base rules
    assert pol.for_field("attn_norm_w", 1, (8,)) == "adamw"


def test_plan_repr_compact():
    plan = _plan(_tiny(opt_policy=freeze(layers=(0,), embed=True)))
    r = repr(plan)
    assert "truncated=0" in r and "train=1" in r and "obj=ce" in r


# ---------------------------------------------------------------- E2E (GPU)

_CAP = 64 * 1024 * 1024


def assert_frozen_model_step(cfg, *, frozen_prefixes, tol=3e-2, seed=0):
    """Freeze-aware E2E gate: one REAL-engine step vs the isolated twin
    stepping only the params OUTSIDE ``frozen_prefixes`` (twin-name
    prefixes — the mirror of the engine's freeze policy). Certifies the
    CE loss and every trainable param in twin-name space, and pins the
    freeze contract itself: frozen params must be BIT-IDENTICAL to the
    shared init on BOTH legs (a wrong allowlist fails loudly here).

    Exists because check_model_step's gradient extraction resolves one
    optimizer task per weight object; fully-frozen objects have none."""
    import torch

    from dataflow_training.model_families import bridges
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow_training.data.segments import uniform_segments
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import (
        EngineFinalBytes,
        reference_model_step,
        rel_l2,
    )

    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=_CAP)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=seed)

    twin = bridges.build_reference_model(cfg)
    trainable = tuple(name for name, par in twin.named_parameters()
                      if not name.startswith(frozen_prefixes))
    n_params = sum(1 for name, par in twin.named_parameters())
    assert trainable and len(trainable) < n_params, frozen_prefixes
    twin_loss, twin, twin_states, init_state, twin_counts = (
        reference_model_step(cfg, values, train_only=trainable, model=twin))

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program,
        resolver=fam.build_resolver(dims),
        initial_buffers=values,
        pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)},
    )
    loss_buf = result.objects.get("loss_0_0").backing.buffer
    run_loss = float(torch_view(loss_buf, (1,), torch.float32)[0])
    assert abs(run_loss - twin_loss) / max(abs(twin_loss), 1e-6) < tol

    engine_state = bridges.to_reference_state_dict(
        cfg, EngineFinalBytes(result))
    twin_state = dict(twin.state_dict())
    for name, engine_tensor in engine_state.items():
        engine_tensor = engine_tensor.cpu()
        if name in init_state and name.startswith(frozen_prefixes):
            init = init_state[name].cpu()
            assert torch.equal(engine_tensor, init), f"engine moved {name}"
            assert torch.equal(twin_state[name].cpu(), init), \
                f"twin moved {name}"
            continue
        err = rel_l2(engine_tensor, twin_state[name])
        assert err < tol, (name, err)
    result.close()
    dry.close()


@pytest.mark.gpu
def test_model_step_truncated_prefix():
    """Layer 0 + embedding frozen: layer 0 has NO backward, NO A, NO
    dW/O; no dy below the head->layer-1 edge; no embed_bwd. Engine
    matches the twin stepping the same trainable set."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    cfg = _tiny(opt_policy=freeze(layers=(0,), embed=True))
    prog = lower_llama3(cfg)
    ids = set(prog.task_by_id())
    sizes = prog.object_sizes()
    assert "block_bwd_0_0_0" not in ids
    assert "A_0_0_0" not in sizes and "dW_0_0" not in sizes
    # the boundary backward keeps its dy output (positional contract);
    # it is consumer-less and disposable. dy_embed's PRODUCER (layer 0's
    # backward) is gone, so it does not exist at all.
    assert "dy_0_0_0" in sizes and "dy_embed_0_0" not in sizes
    assert not any("dy_0_0_0" in tt.inputs
                   for tt in prog.task_by_id().values())
    assert not any(t.startswith("embed_bwd") for t in ids)
    assert_frozen_model_step(cfg, frozen_prefixes=("embed.", "blocks.0."))


@pytest.mark.gpu
def test_model_step_passthrough():
    """Layer 0 frozen, embedding trainable: layer 0's backward runs
    dgrad-only (dw=None tolerated; wgrads skip via the freeze-aware
    acc), dy reaches embed_bwd, dW_embed exists."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    cfg = _tiny(opt_policy=freeze(layers=(0,)))
    prog = lower_llama3(cfg)
    sizes = prog.object_sizes()
    assert "dW_0_0" not in sizes and "dW_embed_0" in sizes
    assert_frozen_model_step(cfg, frozen_prefixes=("blocks.0.",))


@pytest.mark.gpu
def test_model_step_partial_fields():
    """wq/wk frozen fleet-wide: no surgery (plan None), dW/O shrink to
    the trainable fields, the frozen fields' wgrads never compute
    (absent from dw -> acc skips), and every trainable field still
    matches the golden."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    cfg = _tiny(opt_policy=freeze(fields=("wq", "wk")))
    check_model_step(cfg, fast_memory_capacity=_CAP, tol=3e-2).assert_ok()


@pytest.mark.gpu
def test_model_step_truncated_ga2():
    """Grad accumulation across the truncated program: create/accumulate
    rounds on the shrunken dW set. The twin accumulates the SAME two
    rounds (one backward on the summed per-round means) and steps only
    the trainable params; per-round CE, trainable params, and
    frozen-bit-identity all gate."""
    import torch
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")

    from dataflow_training.model_families import bridges
    from dataflow_training.run.driver import adamw_field_step
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow_training.data.segments import uniform_segments
    from dataflow_training.blocks.base_blocks import AdamWHyper
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import EngineFinalBytes, rel_l2

    cfg = _tiny(grad_accum_rounds=2,
                opt_policy=freeze(layers=(0,), embed=True))
    frozen = ("embed.", "blocks.0.")
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    planned = plan_program(fam.lower(cfg), fast_memory_capacity=_CAP)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=0)

    twin = bridges.build_reference_model(cfg)
    bridges.load_reference_init(twin, cfg, dims,
                                bridges.get_bytes_from_values(values))
    init_state = {k: v.detach().clone() for k, v in twin.state_dict().items()}
    twin.train()
    rows = dims.max_tokens // cfg.seq_len
    twin_losses = []
    total = None
    for r in range(cfg.grad_accum_rounds):
        toks = torch_view(values[f"tokens_0_{r}"], (dims.max_tokens,),
                          torch.int32).long().cuda().view(rows, cfg.seq_len)
        tgts = torch_view(values[f"targets_0_{r}"], (dims.max_tokens,),
                          torch.int32).long().cuda().view(rows, cfg.seq_len)
        loss_r = twin.loss(toks, tgts)
        twin_losses.append(float(loss_r.detach()))
        total = loss_r if total is None else total + loss_r
    total.backward()
    hp = AdamWHyper()
    for name, par in twin.named_parameters():
        if par.grad is None or name.startswith(frozen):
            continue
        m = torch.zeros_like(par)
        v = torch.zeros_like(par)
        adamw_field_step(par.data, par.grad, m, v, lr=hp.lr,
                         beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                         weight_decay=hp.weight_decay, step=1)

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)},
    )
    for r, twin_loss in enumerate(twin_losses):
        loss_buf = result.objects.get(f"loss_0_{r}").backing.buffer
        run_loss = float(torch_view(loss_buf, (1,), torch.float32)[0])
        assert abs(run_loss - twin_loss) / max(abs(twin_loss), 1e-6) < 3e-2

    engine_state = bridges.to_reference_state_dict(
        cfg, EngineFinalBytes(result))
    twin_state = dict(twin.state_dict())
    for name, engine_tensor in engine_state.items():
        engine_tensor = engine_tensor.cpu()
        if name.startswith(frozen):
            init = init_state[name].cpu()
            assert torch.equal(engine_tensor, init), f"engine moved {name}"
            assert torch.equal(twin_state[name].cpu(), init), \
                f"twin moved {name}"
            continue
        err = rel_l2(engine_tensor, twin_state[name])
        assert err < 3e-2, (name, err)
    result.close()
    dry.close()


@pytest.mark.gpu
def test_model_step_pair_freeze():
    """(field, layer)-pair axis, end to end: different fields frozen on
    different layers -> per-layer dW layouts differ exactly per policy;
    engine matches the policy-dispatched golden."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    from dataflow_training.blocks.layouts import grad_layout, weight_layout

    cfg = _tiny(opt_policy=freeze(pairs=(("wo", 0), ("w1", 1))))
    prog = lower_llama3(cfg)
    sizes = prog.object_sizes()
    d = derive_dims(cfg)
    for i in (0, 1):
        want = grad_layout(weight_layout(d, layer=i), d.dtypes, layer=i,
                           opt_policy=d.opt_policy).total_bytes
        assert sizes[f"dW_0_{i}"] == want
    assert sizes["dW_0_0"] != sizes["dW_0_1"]   # wo and w1 differ in bytes
    check_model_step(cfg, fast_memory_capacity=_CAP, tol=3e-2).assert_ok()


def test_all_families_truncated_prefix_lowers():
    """Every family derives FreezePlans in its builder: layer 0 + embed
    frozen -> layer 0's backward, its A, and embed_bwd are gone; the
    program still validates. (Engine semantics are covered by the llama3
    and olmoe E2E gates — this pins the structural wiring fleet-wide.)"""
    from dataflow.core.validate import validate_program
    from dataflow_training.model_families.families import family

    for fname in ("qwen3", "qwen35", "qwen35moe", "qwen3moe", "olmoe",
                  "dsv3", "dsv32", "glm52"):
        fam = family(fname)
        cfg = dataclasses.replace(fam.config_type.tiny(),
                                  opt_policy=freeze(layers=(0,), embed=True))
        prog = fam.lower(cfg)
        validate_program(prog)
        ids = set(prog.task_by_id())
        sizes = prog.object_sizes()
        assert "block_bwd_0_0_0" not in ids, fname
        assert "A_0_0_0" not in sizes, fname
        assert not any(t.startswith("embed_bwd") for t in ids), fname


@pytest.mark.gpu
def test_model_step_truncated_olmoe():
    """Truncation through a MoE family's engine path: layer 0 (router,
    experts and all) + embedding frozen — the MoE tail's guarded direct
    dw writes and the aux injection above the boundary must still match
    the twin (which drives the same round-global LBL channel)."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    from dataflow_training.model_families.olmoe import ShapedOlmoeConfig

    cfg = dataclasses.replace(ShapedOlmoeConfig.tiny(),
                              opt_policy=freeze(layers=(0,), embed=True))
    assert_frozen_model_step(cfg, frozen_prefixes=("embed.", "blocks.0."))


def test_train_indexer_unified_into_policy():
    """train_indexer=False is now a freeze-policy composition: the five
    indexer fields vanish from dW/O layouts (before: present, zeroed,
    and skipped by a resolver special-case). The family ablation gates
    prove step parity; this pins the storage consequence."""
    from dataflow_training.blocks.layouts import (
        dsv32_dense_weight_layout,
        grad_layout,
    )
    from dataflow_training.model_families.dsv32 import (
        ShapedDsv32Config,
        derive_dims as derive_dims_dsv32,
        lower_dsv32,
    )

    cfg = dataclasses.replace(ShapedDsv32Config.tiny(), train_indexer=False)
    dims = derive_dims_dsv32(cfg)
    gl = grad_layout(dsv32_dense_weight_layout(dims), dims.dtypes, layer=0,
                     opt_policy=dims.opt_policy)
    names = {f.name for f in gl.fields}
    assert not names & {"w_idx_q", "w_idx_k", "idx_k_ln_w",
                        "idx_k_ln_b", "w_idx_w"}
    prog = lower_dsv32(cfg)
    assert prog.object_sizes()["dW_0_0"] == gl.total_bytes


@pytest.mark.gpu
def test_model_step_frozen_head():
    """Frozen LM head: head_loss still runs (CE + dy_last), but dW_head/
    O_head vanish and the head wgrad GEMM is skipped inside the chunk
    loop (found by --freeze-head bench smoke: the launch used to index
    the dW output positionally)."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 triton kernels need compute capability >= (8, 0)")
    cfg = _tiny(opt_policy=freeze(head=True))
    prog = lower_llama3(cfg)
    ids = set(prog.task_by_id())
    sizes = prog.object_sizes()
    assert any(t_.startswith("head_loss") for t_ in ids)
    assert "dW_head_0" not in sizes and "O_head" not in sizes
    assert not any(t_.startswith("optimizer_head") for t_ in ids)
    # W_head packs lm_head AND final_norm — both frozen with the head
    assert_frozen_model_step(cfg,
                             frozen_prefixes=("lm_head.", "final_norm."))


def test_bench_default_stream_semantics():
    """The bench data contract: shifted targets (next-token within each
    sequence, last position wraps), rounds unique within a step, the
    SAME data repeated across steps (memorization signal)."""
    import torch



    stream, seq = make_default_stream_probe(vocab=97, tokens=32, seq_len=8,
                                             rounds=2, seed=3)
    t0_r0, y0_r0 = stream(0)
    t0_r1, y0_r1 = stream(1)
    t1_r0, y1_r0 = stream(2)   # next step, round 0
    assert not torch.equal(t0_r0, t0_r1)          # unique within step
    assert torch.equal(t0_r0, t1_r0)              # repeated across steps
    assert torch.equal(y0_r0.view(-1, seq)[:, :-1],
                       t0_r0.view(-1, seq)[:, 1:])   # shifted
    assert torch.equal(y0_r0.view(-1, seq)[:, -1],
                       t0_r0.view(-1, seq)[:, 0])    # wrap

    # decoupled mode: targets independent of tokens, still fixed
    # across steps, still unique per round
    dstream, _ = make_default_stream_probe(vocab=97, tokens=32, seq_len=8,
                                            rounds=2, seed=3, decoupled=True)
    dt0, dy0 = dstream(0)
    dt1, dy1 = dstream(2)
    assert torch.equal(dt0, dt1) and torch.equal(dy0, dy1)  # repeat
    assert not torch.equal(dy0.view(-1, 8)[:, :-1],
                           dt0.view(-1, 8)[:, 1:])          # NOT shifted


@pytest.mark.gpu
def test_initial_values_refill_identity():
    """into= refill is byte-identical to a fresh init (deterministic
    seeded fill), so bench's from-init discipline (refill before every
    measured train()) truly restarts the model."""
    import torch

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family

    cfg = _tiny()
    fam = resolve_family(cfg)
    prog = lower_llama3(cfg)
    backend = CudaBackend()
    fresh = fam.initial_values(prog, cfg, backend, seed=7)
    dirty = fam.initial_values(prog, cfg, backend, seed=7)
    for k, buf in dirty.items():   # simulate training: scribble weights
        torch_view(buf, (buf.size_bytes,), torch.uint8).random_(0, 255)
    refilled = fam.initial_values(prog, cfg, backend, seed=7, into=dirty)
    assert refilled is dirty
    for k in fresh:
        a = torch_view(fresh[k], (fresh[k].size_bytes,), torch.uint8)
        b = torch_view(dirty[k], (dirty[k].size_bytes,), torch.uint8)
        assert torch.equal(a, b), k
