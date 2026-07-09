"""General gradient-checking helpers (the plan's correctness-ladder tooling).

- `check_block_backward(dims)`: our hand-written block backward vs autograd
  through the golden block forward — every packed dW field and dx. Also
  verifies recompute-path equivalence (recomputed context ≡ saved context)
  and gradient-accumulation semantics (accum=True adds exactly).
- `check_model_step(...)`: a full annotated program executed by the real
  Engine vs the golden model — loss, every dW-driven update, final params
  and optimizer state.

All comparisons use relative L2 error (robust for bf16) with per-tensor
reporting; failures name the offending field.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.tasks.base_blocks import AdamWHyper  # noqa: F401


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float().cpu().reshape(-1)
    e = expected.detach().float().cpu().reshape(-1)
    denom = e.norm().item()
    if denom == 0.0:
        return a.norm().item()
    return (a - e).norm().item() / denom


@dataclass
class CheckReport:
    errors: dict[str, float]
    tol: float

    @property
    def ok(self) -> bool:
        return all(v <= self.tol for v in self.errors.values())

    def worst(self) -> tuple[str, float]:
        name = max(self.errors, key=self.errors.get)  # type: ignore[arg-type]
        return name, self.errors[name]

    def assert_ok(self) -> None:
        if not self.ok:
            bad = {k: round(v, 5) for k, v in self.errors.items() if v > self.tol}
            raise AssertionError(f"gradcheck failed (tol={self.tol}): {bad}")


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

    w, x, dy = _random_block_state(dims, family.weight_layout(dims), seed)
    kernels = resolve_kernels()
    kctx = KernelCtx()
    fwd = family.block_fwd(dims, kernels)
    bwd = family.block_bwd(dims, kernels)

    # our forward with saved context
    a = _ctx_tensors(family.context_layout(dims))
    y = torch.empty_like(x)
    fwd._forward(kctx, x, w, y, a)

    # recompute-path equivalence: the RECOMPUTE executable's truncated
    # forward must rebuild an identical context from the same x
    a2 = _ctx_tensors(family.context_layout(dims))
    family.block_recompute(dims, kernels)._forward_context(kctx, x, w, a2)
    # the truncated recompute intentionally does NOT produce y (the block
    # output is never a backward dependency) — context equality is the claim
    errors = {f"recompute:{k}": rel_l2(a2[k], a[k]) for k in a}

    # our backward: per-field dW buffers at the policy's GRAD dtypes
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME
    from dataflow.tasks.layouts import grad_layout

    gl = grad_layout(family.weight_layout(dims), dims.dtypes)
    dw = {
        f.name: torch.zeros(f.shape, device="cuda", dtype=TORCH_DTYPE_BY_NAME[f.dtype])
        for f in gl.fields
    }
    dx = torch.empty_like(x)
    bwd._backward(kctx, dy, a, x, w, dx, dw, accum=False)

    # golden: autograd through the reference block forward (per-field leaves)
    golden = family.golden()(dims=dims, n_layers=1)
    leaves = {n: t.detach().clone().requires_grad_() for n, t in w.items()}
    x_ref = x.clone().requires_grad_()
    y_ref = golden.block_forward(x_ref, leaves)
    y_ref.backward(dy)

    errors["fwd:y"] = rel_l2(y, y_ref)
    errors["bwd:dx"] = rel_l2(dx, x_ref.grad)
    for name in dw:
        errors[f"bwd:d{name}"] = rel_l2(dw[name], leaves[name].grad)

    # accumulation semantics: running backward again with accum=True doubles
    bwd._backward(kctx, dy, a, x, w, dx, dw, accum=True)
    for name in dw:
        errors[f"accum:2x:{name}"] = rel_l2(dw[name], 2.0 * leaves[name].grad)

    return CheckReport(errors=errors, tol=tol)


def check_model_step(
    cfg,
    *,
    fast_memory_capacity: int,
    recompute_levels: dict[str, int] | None = None,
    seed: int = 0,
    tol: float = 3e-2,
    field_atol: dict[str, float] | None = None,
    run_args: dict | None = None,
    golden_seq_lens: tuple[int, ...] | None = None,
) -> CheckReport:
    """Ladder level 3: full annotated program through the REAL engine vs the
    golden model — loss, final params, optimizer state.

    ``field_atol`` maps a FIELD name to an absolute elementwise tolerance
    that replaces the rel_l2 comparison for that field. For sub-noise
    sign-lottery parameters only: a one-AdamW-step update of a field whose
    true gradient sits below the bf16 kernel noise floor is +-lr *
    sign(noise) on BOTH sides (qwen3.5's dt_bias — the bf16-ULP-vs-AdamW
    caveat: zero-init params with sub-noise grads), so rel_l2 there measures a coin
    flip, not correctness; the envelope |a-b| <= atol (~2*lr) still catches
    garbage. Real-magnitude fields keep the strict relative gate."""
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    # packed mode: engine takes per-round lens via run_args
    # ("seq_lens"); the GOLDEN runs static semantics with
    # golden_seq_lens (same lens, cfg-level) — same bytes, the
    # equality-with-golden gate for the args-driven path
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    program = fam.lower(cfg, recompute_levels=recompute_levels)
    planned = plan_program(program, fast_memory_capacity=fast_memory_capacity)

    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=seed)

    # golden model + inputs from the same bytes
    def pinned(name: str) -> torch.Tensor:
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    # tied embeddings: one W_embed leaf serves embedding AND head (the golden
    # takes no head arg; the program has no W_head object)
    tied = bool(getattr(cfg, "tied_embeddings", False))
    leaves = [pinned("W_embed"), [pinned(f"W_{i}") for i in range(cfg.n_layers)]]
    if not tied:
        leaves.append(pinned("W_head"))
    if golden_seq_lens is not None:
        import dataclasses as _dc

        _gdims = fam.dims_of(_dc.replace(cfg, seq_lens=tuple(golden_seq_lens)))
    else:
        _gdims = dims
    golden = fam.golden().from_packed_bytes(_gdims, cfg.n_layers, *leaves)
    tokens = torch_view(values["tokens_0_0"], (dims.tokens,), torch.int32).long().cuda()
    targets = (torch_view(values["targets_0_0"], (dims.tokens,), torch.int32).long().cuda()
               if "targets_0_0" in values else tokens.clone())  # warm-up: no CE, unused
    golden_loss = golden.train_step(tokens.cuda(), targets.cuda())

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program,
        resolver=fam.build_resolver(dims),
        initial_buffers=values,
        pool_prewarm=dry.pool_demand,
        run_args=run_args,
    )

    errors: dict[str, float] = {}
    loss_buf = result.objects.get("loss_0_0").backing.buffer  # type: ignore[union-attr]
    run_loss = float(torch_view(loss_buf, (1,), torch.float32)[0])
    errors["loss"] = abs(run_loss - golden_loss) / max(abs(golden_loss), 1e-6)

    def compare_fields(object_id: str) -> float:
        """Worst per-field rel_l2 vs the golden leaves — dtype-true and
        padding-blind (mixed layouts have alignment gaps nobody owns)."""
        from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME

        rec = result.objects.get(object_id)
        slot = rec.backing or rec.fast
        layout, leaves = golden.final_leaves(object_id)
        worst = 0.0
        for f in layout.fields:
            rt = torch_view(
                slot.buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                offset_bytes=f.offset_bytes,
            )
            atol = (field_atol or {}).get(f.name)
            if atol is not None:
                gap = float(
                    (rt.float().cpu() - leaves[f.name].float().cpu()).abs().max()
                )
                worst = max(worst, 0.0 if gap <= atol else gap / atol)
            else:
                worst = max(worst, rel_l2(rt, leaves[f.name]))
        return worst

    errors["W_embed"] = compare_fields("W_embed")
    for i in range(cfg.n_layers):
        errors[f"W_{i}"] = compare_fields(f"W_{i}")
    if not tied:
        errors["W_head"] = compare_fields("W_head")
    result.close()  # release the engine-local slab/arena for the next check
    dry.close()
    from dataflow.tasks.interop import clear_view_cache

    clear_view_cache()
    for buf in values.values():
        backend.free(buf)
    return CheckReport(errors=errors, tol=tol)
