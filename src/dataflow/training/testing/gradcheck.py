"""General gradient-checking helpers (the plan's correctness-ladder tooling).

- `check_block_backward(dims)`: our hand-written block backward vs autograd
  through the pinned reference block forward — every packed dW field and
  dx. Also verifies recompute-path equivalence (recomputed context ≡
  saved context) and gradient-accumulation semantics (accum=True adds).
- `check_model_step(...)`: a full annotated program executed by the real
  Engine vs the ISOLATED reference twin — loss, final params, gradients
  in dW space, and MoE counts parity (see docs/correctness_compare.md).

All comparisons use relative L2 error (robust for bf16) with per-tensor
reporting; failures name the offending field.
"""
from __future__ import annotations

from dataclasses import dataclass

import dataclasses

import torch

from dataflow.tasks.base_blocks import AdamWHyper  # noqa: F401


def cos_sim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Flat fp64 cosine similarity — the direction-agreement check
    paired with rel_l2 magnitude checks in every per-tensor
    engine-vs-reference comparison. Healthy implementations sit
    > 0.995; zero-vectors count as aligned (nothing to disagree on)."""
    a = actual.detach().double().reshape(-1)
    b = expected.detach().double().reshape(-1).to(a.device)
    na = float(a.norm())
    nb = float(b.norm())
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a @ b) / (na * nb))


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float().cpu().reshape(-1)
    e = expected.detach().float().cpu().reshape(-1)
    denom = e.norm().item()
    if denom == 0.0:
        # moved-vs-zero is the LOUDEST possible signal, not a small
        # absolute norm (a dark training channel hid behind the old
        # |a| return: engine-trained fields vs a zero twin read as ~lr)
        return float("inf") if a.norm().item() > 0.0 else 0.0
    return (a - e).norm().item() / denom


@dataclass
class CheckReport:
    errors: dict[str, float]
    tol: float
    # per-tensor direction agreement (cos_sim); empty for scalar-only
    # checks. Enveloped sub-noise fields are excluded by the harness —
    # a sign-lottery vector has no meaningful direction.
    cosines: dict[str, float] = dataclasses.field(default_factory=dict)
    min_cosine: float = 0.995
    # check_model_step(capture=True): twin-space {"engine","twin","init"}
    # state dicts (cpu fp32) for control-pair / offline analysis
    states: dict | None = None
    # per-PREFIX tolerance overrides (e.g. {"grad:": 0.2}): entries whose
    # name starts with a key gate against that tolerance instead of
    # ``tol`` — gradient-space bands are calibrated per family (flip
    # amplification, docs/correctness_compare.md) while param/loss
    # entries stay tight
    tol_by_prefix: dict | None = None

    def tol_for(self, name: str) -> float:
        if self.tol_by_prefix:
            for prefix, t in self.tol_by_prefix.items():
                if name.startswith(prefix):
                    return t
        return self.tol

    @property
    def ok(self) -> bool:
        if any(v > self.tol_for(k) for k, v in self.errors.items()):
            return False
        return all(c >= self.min_cosine for c in self.cosines.values())

    def worst(self) -> tuple[str, float]:
        name = max(self.errors, key=self.errors.get)  # type: ignore[arg-type]
        return name, self.errors[name]

    def assert_ok(self) -> None:
        if not self.ok:
            bad = {k: round(v, 5) for k, v in self.errors.items()
                   if v > self.tol_for(k)}
            low = {k: round(c, 5) for k, c in self.cosines.items()
                   if c < self.min_cosine}
            raise AssertionError(
                f"gradcheck failed (tol={self.tol}, "
                f"min_cos={self.min_cosine}): over-tol {bad}; "
                f"cosine under {low}")


def _random_block_state(dims, wl, seed: int):
    """Per-field random weights at each field's OWN storage dtype."""
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME

    gen = torch.Generator(device="cuda").manual_seed(seed)
    views = {}
    for f in wl.fields:
        n = 1
        for d in f.shape:
            n *= int(d)
        dt = TORCH_DTYPE_BY_NAME[f.dtype]
        if f.name.endswith("_norm_w"):
            views[f.name] = torch.ones(f.shape, device="cuda", dtype=dt)
        else:
            views[f.name] = (
                torch.randn(n, generator=gen, device="cuda") * 0.02
            ).to(dt).view(f.shape)
    x = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    dy = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    return views, x, dy


def _ctx_tensors(cl) -> dict[str, torch.Tensor]:
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME

    return {
        f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
        for f in cl.fields
    }


def check_block_backward(dims, *, family=None, seed: int = 0, tol: float = 3e-2) -> CheckReport:
    """Ladder level 2 for the transformer block (fwd/bwd/recompute/accum).

    ``family`` is a `dataflow.training.families.Family`; defaults to llama3
    (back-compat for existing callers passing bare LlamaDims)."""
    from dataflow.tasks.kernels import KernelCtx, resolve_kernels

    if family is None:
        from dataflow.training.families import family as _fam

        family = _fam("llama3")
    if family.block_fwd is None:
        raise TypeError(
            f"family {family.name!r} is heterogeneous (no generic gradcheck "
            "bundle); its per-kind block ladders live in its own test module"
        )

    from dataflow.tasks.ops import Segments

    w, x, dy = _random_block_state(dims, family.weight_layout(dims), seed)
    kernels = resolve_kernels()
    kctx = KernelCtx()
    fwd = family.block_fwd(dims, kernels)
    bwd = family.block_bwd(dims, kernels)

    # ONE materialized Segments handed to BOTH sides (bypassing launch, this
    # gate supplies the varlen metadata the engine prologue normally would)
    seg = Segments.of_dims(dims).on("cuda")

    # our forward with saved context
    a = _ctx_tensors(family.activation_layout(dims))
    y = torch.empty_like(x)
    fwd._forward(kctx, x, w, y, a, extras={"seg": seg})

    # recompute-path equivalence: the RECOMPUTE executable's truncated
    # forward must rebuild an identical context from the same x
    a2 = _ctx_tensors(family.activation_layout(dims))
    family.block_recompute(dims, kernels)._forward_context(kctx, x, w, a2, extras={"seg": seg})
    # the truncated recompute intentionally does NOT produce y (the block
    # output is never a backward dependency) — context equality is the claim
    errors = {f"recompute:{k}": rel_l2(a2[k], a[k]) for k in a}
    cosines: dict[str, float] = {}

    # our backward: per-field dW buffers at the policy's GRAD dtypes
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME
    from dataflow.tasks.layouts import grad_layout

    gl = grad_layout(family.weight_layout(dims), dims.dtypes)
    dw = {
        f.name: torch.zeros(f.shape, device="cuda", dtype=TORCH_DTYPE_BY_NAME[f.dtype])
        for f in gl.fields
    }
    dx = torch.empty_like(x)
    a["_seg"] = seg  # the launch merges this into `a`; here we do it directly
    bwd._backward(kctx, dy, a, x, w, dx, dw, accum=False)

    # autograd through the block-level reference form (per-field leaves,
    # composed from the pinned tasks.ops reference library)
    from dataflow.training.testing.block_forms import BLOCK_FORWARDS

    block_forward = BLOCK_FORWARDS[family.name]
    leaves = {n: t.detach().clone().requires_grad_() for n, t in w.items()}
    x_ref = x.clone().requires_grad_()
    y_ref = block_forward(dims, x_ref, leaves, seg)
    y_ref.backward(dy)

    errors["fwd:y"] = rel_l2(y, y_ref)
    cosines["fwd:y"] = cos_sim(y, y_ref)
    errors["bwd:dx"] = rel_l2(dx, x_ref.grad)
    cosines["bwd:dx"] = cos_sim(dx, x_ref.grad)
    for name in dw:
        errors[f"bwd:d{name}"] = rel_l2(dw[name], leaves[name].grad)
        cosines[f"bwd:d{name}"] = cos_sim(dw[name], leaves[name].grad)

    # accumulation semantics: running backward again with accum=True doubles
    bwd._backward(kctx, dy, a, x, w, dx, dw, accum=True)
    for name in dw:
        errors[f"accum:2x:{name}"] = rel_l2(dw[name], 2.0 * leaves[name].grad)

    return CheckReport(errors=errors, tol=tol, cosines=cosines)


# The gradient tier's learning rate: SGD's update is -lr*grad, but bf16
# weight storage rounds the applied update to the weight's quantum
# (ulp(0.02) ~ 6e-5). At the training default lr=1e-4 the update is ~1
# quantum and BOTH legs' updates collapse to rounding lottery; at 1e-2
# the update spans ~160 quanta and quantization contributes <~1%
# relative — update-space comparison then measures the GRADIENTS.
GRAD_TIER_LR = 1e-2


def reference_model_step(cfg, values, *, seq_lens=None,
                         train_only=None, optimizer="adamw", model=None,
                         hyper=None):
    """One exact-replica training step of the pure-torch twin from the
    ENGINE's init bytes: forward/backward (per-sequence when
    ``seq_lens`` packs the round — exact, attention never crosses
    sequences), then the optimizer replica (engine-default hyper,
    step 1) on every parameter. ``optimizer`` selects the replica:
    "adamw" (the default step) or "sgd" (p -= lr*g — the GRADIENT
    tier: the update IS the gradient, so update-space comparisons
    measure gradient parity without adamw's step-1 sign quantization
    amplifying sub-noise elements into coin flips).

    When ``cfg.aux_coef > 0`` and the twin declares
    ``AUX_FORM == "sequence_wise"``, each sequence's backward adds the
    UNSCALED ``aux_coef * load_balance_loss()`` term — exactly the
    engine's per-round seq-aux application (full alpha per round; the
    sequence-wise form decomposes over the per-sequence forwards).
    Twins with other forms must have aux zeroed by the CALLER on both
    legs (check_model_step does this) until a round-global drive
    exists. Returns (ce_loss, model, opt_states, init_state)."""
    from dataflow.pretrain import bridges
    from dataflow.pretrain.driver import adamw_field_step
    from dataflow.tasks.base_blocks import AdamWHyper
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    if model is None:
        model = bridges.build_reference_model(cfg)
    bridges.load_reference_init(model, cfg, dims,
                                bridges.get_bytes_from_values(values))
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    tokens = torch_view(values["tokens_0_0"], (dims.tokens,),
                        torch.int32).long().cuda()
    targets = (torch_view(values["targets_0_0"], (dims.tokens,),
                          torch.int32).long().cuda()
               if "targets_0_0" in values else tokens.clone())
    if seq_lens is None:
        seq = getattr(dims, "seq_len", None) or dims.tokens
        lens = tuple([seq] * (dims.tokens // seq))
    else:
        lens = tuple(int(x) for x in seq_lens)
    aux_coef = float(getattr(cfg, "aux_coef", 0.0) or 0.0)
    aux_form = getattr(model, "AUX_FORM", None)
    drive_aux_seq = aux_coef > 0.0 and aux_form == "sequence_wise"
    drive_aux_round = aux_coef > 0.0 and aux_form == "forward_global"
    drive_idx = (bool(getattr(cfg, "train_indexer", False))
                 and hasattr(model, "indexer_loss"))
    if drive_idx:
        model.enable_indexer_kl(True)
    step_valid = int((targets >= 0).sum())
    if getattr(model, "SUPPORTS_PACKED", False):
        # NATIVE VARLEN: the whole round is ONE packed forward — per-
        # sequence rope positions + block-diagonal attention inside the
        # twin (per-segment state resets for recurrent mixers). The aux
        # term of a packed forward IS the round term for both forms:
        # sequence_wise computes per segment internally; forward_global
        # over the packed tokens IS the engine's ROUND-global default.
        t = tokens.view(1, -1)
        g = targets.view(1, -1)
        ce = model.loss(t, g, seq_lens=lens)
        total = ce
        if aux_coef > 0.0 and (drive_aux_seq or drive_aux_round):
            # full alpha, never scaled by the CE denominator (the
            # engine's per-round application); loss CHANNEL stays CE
            total = total + aux_coef * model.load_balance_loss()
        if drive_idx:
            # DSA indexer KL, coefficient 1, unscaled; gradient reaches
            # only the indexer weights (both seams detached in the twin)
            total = total + model.indexer_loss()
        loss_total = float(ce.detach())
        total.backward()
    else:
        # fallback for twins not yet varlen-native (onboarding path):
        # per-sequence forwards + one deferred backward
        if drive_aux_round:
            model.reset_round_lbl()
        loss_total = 0.0
        total = None
        lo = 0
        for ln in lens:
            t = tokens[lo:lo + ln].view(1, ln)
            g = targets[lo:lo + ln].view(1, ln)
            valid = int((g >= 0).sum())
            ce = model.loss(t, g)
            seq_loss = ce * (valid / step_valid)
            if drive_aux_seq:
                seq_loss = seq_loss + aux_coef * model.load_balance_loss()
            if drive_idx:
                seq_loss = seq_loss + model.indexer_loss()
            total = seq_loss if total is None else total + seq_loss
            loss_total += float(ce.detach()) * (valid / step_valid)
            lo += ln
        if drive_aux_round:
            total = total + aux_coef * model.round_load_balance_loss()
        total.backward()
    hp = hyper if hyper is not None else AdamWHyper()
    states = {}
    for name, par in model.named_parameters():
        if par.grad is None:
            continue
        if train_only is not None and not any(
                name == t or name.endswith("." + t) or name.endswith(t)
                for t in train_only):
            continue        # frozen-phase configs: only listed params
                            # update (mirrors the engine's freeze plan)
        if optimizer == "sgd":
            par.data.add_(par.grad, alpha=-hp.lr)
            states[name] = ()
            continue
        m = torch.zeros_like(par)
        v = torch.zeros_like(par)
        adamw_field_step(par.data, par.grad, m, v, lr=hp.lr,
                         beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                         weight_decay=hp.weight_decay, step=1)
        states[name] = (m, v)
    step_counts = {name: m.step_counts.detach().clone()
                   for name, m in model.named_modules()
                   if getattr(m, "step_counts", None) is not None}
    speed = float(getattr(cfg, "bias_update_speed", 0.0) or 0.0)
    if speed > 0.0:
        # noaux balance rule, optimizer-time, step-aggregate counts —
        # the same module-walk the crosscheck runners use
        for module in model.modules():
            if hasattr(module, "apply_bias_update"):
                module.apply_bias_update(speed)
    return loss_total, model, states, init_state, step_counts


class EngineFinalBytes:
    """get_bytes over an engine run's post-step objects — feeds the
    bridge's to_reference_state_dict for twin-space comparison."""

    def __init__(self, run_result):
        self.result = run_result

    def __call__(self, object_id: str):
        from dataflow.tasks.interop import torch_view

        rec = self.result.objects.get(object_id)
        slot = rec.backing or rec.fast
        return torch_view(slot.buffer, (slot.buffer.size_bytes,),
                          torch.uint8)


class BlockRecorder(torch.nn.Module):
    """Composition wrapper capturing a twin block's output rows (and
    optionally swapping its input for the ENGINE's previous-block
    output — the per-block isolation instrument)."""

    def __init__(self, inner, swap_input=None, seq_offsets=None,
                 d_model=None):
        super().__init__()
        self.inner = inner
        self.swap_input = swap_input
        self.seq_offsets = seq_offsets
        self.d_model = d_model
        self.calls = 0
        self.outs = []

    def forward(self, x, *rest, **kw):
        if self.swap_input is not None:
            lo = self.seq_offsets[self.calls]
            hi = self.seq_offsets[self.calls + 1]
            x = self.swap_input[lo:hi].view(1, -1, self.d_model).to(x.dtype)
        self.calls += 1
        y = self.inner(x, *rest, **kw)
        keep = y[0] if isinstance(y, tuple) else y
        self.outs.append(keep.detach().float().cpu().reshape(
            -1, keep.shape[-1]))
        return y

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def isolated_block_compare(cfg, isolate, *, seed: int = 0,
                           fast_memory_capacity: int = 96 * 1024 * 1024,
                           hot_mult: float = 10.0) -> dict:
    """The per-block isolation instrument as a callable gate: feed the
    ENGINE's block-(N-1) output into the twin's block N (per sequence)
    and compare the block-N outputs row by row. Removes ALL upstream
    divergence, so per-block math is certified at the bf16 floor even
    for families whose MODEL-level comparison is saturated by discrete
    routing-flip cascades (docs/correctness_compare.md gotchas 4/8/9).
    ``isolate`` is a tuple of block indices swapped together — include
    a DSA follower's LEADER or the follower shows phantom broadband
    (gotcha 9). Returns {"rel", "cos", "row_median", "row_p90",
    "hot_rows", "rel_excl_hot"} for the LAST listed block."""
    import dataclasses as dc

    from dataflow.pretrain import bridges
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.engine import uniform_segments
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    program = fam.lower(cfg)
    keep = {o.id for t in program.tasks for o in t.outputs
            if o.id.startswith("y_")}
    program = dc.replace(
        program,
        final_locations={**dict(program.final_locations),
                         **{i: "fast" for i in keep}})
    planned = plan_program(program,
                           fast_memory_capacity=fast_memory_capacity)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=seed)

    twin = bridges.build_reference_model(cfg)
    bridges.load_reference_init(twin, cfg, dims,
                                bridges.get_bytes_from_values(values))
    twin.eval()

    run_args = {"segments": uniform_segments(dims, planned.program)}
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args=run_args)

    lens = tuple(getattr(dims, "seq_lens", None)
                 or (dims.seq_len,) * (dims.tokens // dims.seq_len))
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)

    recorders = {}
    for b in isolate:
        swap = None
        if b > 0:
            rec = result.objects.get(f"y_0_0_{b - 1}")
            slot = rec.fast or rec.backing
            swap = torch_view(slot.buffer, (dims.tokens, dims.d_model),
                              torch.bfloat16).clone()
        recorders[b] = BlockRecorder(twin.blocks[b], swap_input=swap,
                                     seq_offsets=offsets,
                                     d_model=dims.d_model)
        twin.blocks[b] = recorders[b]

    tokens = torch_view(values["tokens_0_0"], (dims.tokens,),
                        torch.int32).long().cuda()
    with torch.no_grad():
        lo = 0
        for ln in lens:
            twin(tokens[lo:lo + ln].view(1, ln))
            lo += ln

    last = isolate[-1]
    rec = result.objects.get(f"y_0_0_{last}")
    slot = rec.fast or rec.backing
    eng = torch_view(slot.buffer, (dims.tokens, dims.d_model),
                     torch.bfloat16).float().cpu()
    twn = torch.cat(recorders[last].outs)
    rowrel = (eng - twn).norm(dim=1) / twn.norm(dim=1).clamp_min(1e-12)
    med = float(rowrel.median())
    hot = (rowrel > max(hot_mult * med, 0.05)).nonzero().flatten().tolist()
    mask = torch.ones(dims.tokens, dtype=torch.bool)
    mask[hot] = False
    out = {
        "rel": rel_l2(eng, twn),
        "cos": cos_sim(eng, twn),
        "row_median": med,
        "row_p90": float(rowrel.quantile(0.9)),
        "hot_rows": hot,
        "rel_excl_hot": rel_l2(eng[mask], twn[mask]),
    }
    result.close()
    dry.close()
    from dataflow.tasks.interop import clear_view_cache

    clear_view_cache()
    for buf in values.values():
        backend.free(buf)
    return out


def engine_grad_state_dict(cfg, fam, dims, program, resolver, values,
                           result) -> dict:
    """Engine dW slabs -> twin-named gradient dict, with ZERO per-family
    code: each optimizer task's executable resolves its own
    (weight_layout, grad_layout, ns) — per-kind dispatch included — and
    the family bridge's to_reference_state_dict supplies the name map by
    reading FABRICATED weight-layout buffers whose field slots hold the
    gradient values (fields without gradient storage — frozen balance
    biases — are NaN-poisoned and stripped).

    Requires the run to have retained dW objects (final_locations)."""
    import math

    from dataflow.pretrain import bridges
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view

    w_sizes = {o.id: o.size_bytes for o in program.initial_objects}
    for oid, buf in values.items():
        w_sizes.setdefault(oid, buf.size_bytes)   # warmup programs may
        # not declare every frozen W as an initial object; the caller's
        # buffers know every size
    fabricated: dict[str, torch.Tensor] = {}
    for task in program.tasks:
        if not task.id.startswith("optimizer_"):
            continue
        w_id = next(i for i in task.inputs if i.startswith("W_"))
        dw_id = next(i for i in task.inputs if i.startswith("dW"))
        ex = resolver(task)
        wl, gl = ex._layouts(task, w_sizes[w_id])[:2]
        rec = result.objects.get(dw_id)
        slot = rec.fast or rec.backing
        raw = torch_view(slot.buffer, (slot.buffer.size_bytes,), torch.uint8)
        grads = gl.unpack_tensor(raw)
        gl_dtypes = {f.name: f.dtype for f in gl.fields}
        fake = torch.empty(wl.total_bytes, dtype=torch.uint8, device="cuda")
        views = wl.unpack_tensor(fake)
        for f in wl.fields:
            if f.name in grads:
                if gl_dtypes[f.name] != f.dtype:
                    raise AssertionError(
                        f"{dw_id}:{f.name} grad dtype "
                        f"{gl_dtypes[f.name]} != weight dtype {f.dtype} — "
                        f"the fabricated-buffer shim would quantize")
                views[f.name].copy_(grads[f.name])
            else:
                views[f.name].fill_(
                    float("nan")
                    if TORCH_DTYPE_BY_NAME[f.dtype].is_floating_point
                    else 0)
        fabricated[w_id] = fake

    def shim(object_id):
        made = fabricated.get(object_id)
        if made is not None:
            return made
        # a W object with NO optimizer task (fully frozen): no gradient
        # storage exists — hand the bridge a NaN-poisoned buffer of the
        # right size so every derived entry is stripped below (bf16 NaN
        # bytes read as NaN under every float dtype view)
        n = w_sizes[object_id]
        blank = torch.zeros(n, dtype=torch.uint8, device="cuda")
        blank[: n - n % 2].view(torch.bfloat16).fill_(float("nan"))
        fabricated[object_id] = blank
        return blank

    out = {}
    for name, t in bridges.to_reference_state_dict(cfg, shim).items():
        tf = t.float()
        if tf.isnan().all():
            continue                     # gradient-free field (frozen)
        out[name] = t
    return out


# Per-family gradient-space bands + flip budgets for ladder-3 gates,
# calibrated at ~2x the measured worst of the reference matrix
# (docs/correctness_compare.md "Reference healthy numbers"); the tier
# mechanisms (near-tie flips; recurrent-state amplification; two-chooser
# cascade) are documented there. Param/loss entries always gate at the
# tight defaults — these bands apply to grad:{name} entries only.
FAMILY_GRAD_GATE = {
    "llama3":    (3e-2, 0.9990, None),
    "qwen3":     (4e-2, 0.9990, None),
    "qwen35":    (1e-1, 0.9980, None),
    "qwen3moe":  (2e-1, 0.9900, 3),
    "olmoe":     (3e-1, 0.9750, 3),
    "dsv3":      (4.5e-1, 0.9500, 3),
    "dsv32":     (4.5e-1, 0.9500, 4),
    "glm52":     (7e-1, 0.8700, 8),
    "qwen35moe": (8e-1, 0.8900, 4),
}


def family_gate_kwargs(family_name: str) -> dict:
    """check_model_step kwargs for a family's calibrated gradient gate:
    {grad_tol, min_cosine, counts_budget}. Families absent from the
    table gate at the tight defaults."""
    if family_name not in FAMILY_GRAD_GATE:
        return {}
    grad_tol, min_cos, budget = FAMILY_GRAD_GATE[family_name]
    return {"grad_tol": grad_tol, "min_cosine": min_cos,
            "counts_budget": budget}


def field_atol_for(name: str, field_atol: dict | None):
    """Suffix-match a twin parameter/buffer name against the caller's
    atol table (keys are engine-field/short names, e.g.
    "w_router_bias")."""
    if not field_atol:
        return None
    for key, atol in field_atol.items():
        if name == key or name.endswith("." + key) or name.endswith(key):
            return atol
    return None


def check_model_step(
    cfg,
    *,
    fast_memory_capacity: int,
    recompute_levels: dict[str, int] | None = None,
    seed: int = 0,
    tol: float = 3e-2,
    field_atol: dict[str, float] | None = None,
    run_args: dict | None = None,
    reference_seq_lens: tuple[int, ...] | None = None,
    reference_train_only: tuple[str, ...] | None = None,
    optimizer: str = "adamw",
    capture: bool = False,
    min_cosine: float = 0.995,
    counts_budget: int | None = 3,
    grad_tol: float | None = None,
) -> CheckReport:
    """Ladder level 3: full annotated program through the REAL engine vs
    the ISOLATED pure-torch reference twin — loss, final params, AND
    one-step UPDATES (final - init), compared in twin-name space (the
    engine's final bytes are mapped through the family bridge's
    to_reference_state_dict, so every comparison key is a twin
    parameter/buffer name).

    The update-space entries (``upd:{name}``) are the sharp instrument:
    at one AdamW step from zero moments the update is ~= lr * sign(grad)
    elementwise, so final-param comparisons are dominated by the shared
    init (|p0| >> lr) and can hide gradient-semantics errors; the update
    comparison strips p0 and turns any sign disagreement into signal.

    ``reference_seq_lens`` overrides the twin's per-sequence packing;
    when None it follows the config's static ``seq_lens`` (the twin
    computes attention per independent row, so ragged rounds MUST reach
    it as separate sequences), else uniform rows.

    ``field_atol`` maps a name SUFFIX to an absolute elementwise
    tolerance replacing rel_l2/update checks for that entry — reserved
    for parameters whose one-step update is provably below cross-
    implementation kernel noise (a documented minimal example is the
    bar for adding one; see the varlen suite's envelope notes)."""
    from dataclasses import replace as dc_replace

    from dataflow.pretrain import bridges
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    hyper = AdamWHyper()
    if optimizer == "sgd":
        cfg = dc_replace(cfg, opt_policy="sgd")
        hyper = AdamWHyper(lr=GRAD_TIER_LR)
    twin_cfg = cfg
    if getattr(cfg, "sparse_mode", True) is False:
        # DSA dense warm-up: the twins are sparse-only BY DESIGN, and
        # dense attention IS the sparse path with the whole prefix
        # selected (index_topk >= T) — the KL live set then covers the
        # full prefix, the paper's warm-up objective. The ENGINE leg
        # keeps the true warm-up cfg; only the twin build maps.
        twin_cfg = dc_replace(cfg, sparse_mode=True, index_topk=1 << 30)
    twin = bridges.build_reference_model(twin_cfg)
    if (float(getattr(cfg, "aux_coef", 0.0) or 0.0) > 0.0
            and getattr(twin, "AUX_FORM", None) is None):
        # a twin that declares NO load-balance form cannot express the
        # engine's LBL — zero the channel on BOTH legs (symmetric) and
        # treat a missing declaration on an MoE family as a gap to fix
        cfg = dc_replace(cfg, aux_coef=0.0)

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    program = fam.lower(cfg, recompute_levels=recompute_levels)
    # retain the gradient slabs: dW ids in final_locations survive pool
    # recycling after the optimizer consumes them, enabling the direct
    # dW-vs-autograd comparison (the sharp parity instrument — final
    # params are init-dominated ~200x, update space is bf16-quantum
    # contaminated at training lr)
    dw_ids = {o.id for t in program.tasks for o in t.outputs
              if o.id.startswith("dW")}
    dw_ids.update(o.id for o in program.initial_objects
                  if o.id.startswith("dW"))
    program = dc_replace(
        program,
        final_locations={**dict(program.final_locations),
                         **{i: "fast" for i in dw_ids}})
    planned = plan_program(program, fast_memory_capacity=fast_memory_capacity)

    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=seed)

    if reference_seq_lens is None:
        reference_seq_lens = getattr(dims, "seq_lens", None)
    twin_loss, twin, _twin_states, init_state, twin_counts = (
        reference_model_step(
            twin_cfg, values, seq_lens=reference_seq_lens,
            train_only=reference_train_only, optimizer=optimizer,
            model=twin, hyper=hyper))
    if getattr(cfg, "sparse_mode", True) is False:
        # warm-up's loss CHANNEL is the multi-layer indexer objective
        # (main model frozen), not CE — compare like for like
        twin_loss = float(twin.indexer_loss())

    if run_args is None:
        from dataflow.runtime.engine import uniform_segments

        run_args = {"segments": uniform_segments(dims, planned.program)}

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program,
        resolver=fam.build_resolver(dims, hyper),
        initial_buffers=values,
        pool_prewarm=dry.pool_demand,
        run_args=run_args,
    )

    errors: dict[str, float] = {}
    cosines: dict[str, float] = {}
    loss_buf = result.objects.get("loss_0_0").backing.buffer  # type: ignore[union-attr]
    run_loss = float(torch_view(loss_buf, (1,), torch.float32)[0])
    errors["loss"] = abs(run_loss - twin_loss) / max(abs(twin_loss), 1e-6)

    engine_state = bridges.to_reference_state_dict(
        cfg, EngineFinalBytes(result))
    twin_state = dict(twin.state_dict())
    for name, engine_tensor in engine_state.items():
        twin_tensor = twin_state.get(name)
        if twin_tensor is None:
            errors[name] = float("inf")
            continue
        atol = field_atol_for(name, field_atol)
        if atol is not None:
            gap = float((engine_tensor.float().cpu()
                         - twin_tensor.float().cpu()).abs().max())
            errors[name] = 0.0 if gap <= atol else gap / atol
            continue
        errors[name] = rel_l2(engine_tensor, twin_tensor)
        cosines[name] = cos_sim(engine_tensor, twin_tensor)

    twin_grads = {n: par.grad for n, par in twin.named_parameters()
                  if par.grad is not None}
    engine_grads = engine_grad_state_dict(
        cfg, fam, dims, planned.program, fam.build_resolver(dims, hyper),
        values, result)
    for name, g_engine in engine_grads.items():
        g_twin = twin_grads.get(name)
        if g_twin is None or g_twin.shape != g_engine.shape:
            continue        # frozen/train_only params carry no twin grad
        atol = field_atol_for(name, field_atol)
        if atol is not None:
            continue        # enveloped fields: raw-gap gate above only
        errors["grad:" + name] = rel_l2(g_engine, g_twin)
        cosines["grad:" + name] = cos_sim(g_engine, g_twin)
    aux_ids = sorted((k for k in result.objects.records
                      if k.startswith("Aux_")),
                     key=lambda k: int(k.split("_")[1]))
    if counts_budget is not None and aux_ids and twin_counts \
            and len(aux_ids) == len(twin_counts):
        # MoE counts parity: totals must equal tokens*top_k EXACTLY on
        # both sides (a mismatch is a counting bug); per-expert deltas
        # bound the number of near-tie flipped tokens, gated by the
        # flip budget (sum|delta|/2 <= budget -> entry 0.0, else the
        # flip count itself, which no float tol admits)
        from dataflow.tasks.modules.moe.spec import moe_aux_layout

        layout = moe_aux_layout(dims, dims.moe)
        for (tname, tc), oid in zip(sorted(twin_counts.items()), aux_ids):
            rec = result.objects.get(oid)
            slot = rec.fast or rec.backing
            ec = layout.views(slot.buffer)[
                "expert_counts_current_step"].long().cpu()
            tc = tc.long().cpu()
            if int(ec.sum()) != int(tc.sum()):
                errors["counts:" + tname] = float("inf")
                continue
            flips = int((ec - tc).abs().sum()) // 2
            errors["counts:" + tname] = (0.0 if flips <= counts_budget
                                         else float(flips))

    captured = None
    if capture:
        captured = {
            "engine": {k: v.detach().float().cpu().clone()
                       for k, v in engine_state.items()},
            "twin": {k: v.detach().float().cpu().clone()
                     for k, v in twin_state.items()},
            "init": {k: v.detach().float().cpu().clone()
                     for k, v in init_state.items()},
        }
    result.close()
    dry.close()
    from dataflow.tasks.interop import clear_view_cache

    clear_view_cache()
    for buf in values.values():
        backend.free(buf)
    return CheckReport(errors=errors, tol=tol, cosines=cosines,
                       min_cosine=min_cosine, states=captured,
                       tol_by_prefix=({"grad:": grad_tol}
                                      if grad_tol is not None else None))

