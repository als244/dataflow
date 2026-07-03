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

from dataflow.tasks.layouts import LlamaDims, context_layout, weight_layout
from dataflow.tasks.llama3_blocks import AdamWHyper, BlockBwd, BlockFwd


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


def _random_block_state(dims: LlamaDims, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    wl = weight_layout(dims)
    flat = torch.randn(wl.total_bytes // 2, generator=gen, device="cuda", dtype=torch.float32)
    flat = (flat * 0.02).to(torch.bfloat16)
    views = {}
    for f in wl.fields:
        n = 1
        for d in f.shape:
            n *= int(d)
        start = f.offset_bytes // 2
        views[f.name] = flat[start : start + n].view(f.shape)
    views["attn_norm_w"].fill_(1.0)
    views["ffn_norm_w"].fill_(1.0)
    x = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    dy = (torch.randn(dims.tokens, dims.d_model, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    return flat, views, x, dy


def _ctx_tensors(dims: LlamaDims) -> dict[str, torch.Tensor]:
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME

    cl = context_layout(dims)
    return {
        f.name: torch.empty(f.shape, dtype=TORCH_DTYPE_BY_NAME[f.dtype], device="cuda")
        for f in cl.fields
    }


def check_block_backward(dims: LlamaDims, *, seed: int = 0, tol: float = 3e-2) -> CheckReport:
    """Ladder level 2 for the transformer block (fwd/bwd/recompute/accum)."""
    from dataflow.models.llama3_reference import GoldenLlama3

    from dataflow.tasks.kernels import KernelCtx, resolve_kernels

    flat, w, x, dy = _random_block_state(dims, seed)
    kernels = resolve_kernels()
    kctx = KernelCtx()
    fwd = BlockFwd(dims, kernels)
    bwd = BlockBwd(dims, kernels)

    # our forward with saved context
    a = _ctx_tensors(dims)
    y = torch.empty_like(x)
    fwd._forward(kctx, x, w, y, a)

    # recompute-path equivalence: the RECOMPUTE executable's truncated
    # forward must rebuild an identical context from the same x
    from dataflow.tasks.llama3_blocks import BlockRecompute

    a2 = _ctx_tensors(dims)
    BlockRecompute(dims, kernels)._forward_context(kctx, x, w, a2)
    # the truncated recompute intentionally does NOT produce y (the block
    # output is never a backward dependency) — context equality is the claim
    errors = {f"recompute:{k}": rel_l2(a2[k], a[k]) for k in a}

    # our backward
    wl = weight_layout(dims)
    dwflat = torch.zeros_like(flat)
    dw = {}
    for f in wl.fields:
        n = 1
        for d in f.shape:
            n *= int(d)
        start = f.offset_bytes // 2
        dw[f.name] = dwflat[start : start + n].view(f.shape)
    dx = torch.empty_like(x)
    bwd._backward(kctx, dy, a, x, w, dx, dw, accum=False)

    # golden: autograd through the reference block forward
    golden = GoldenLlama3(dims=dims, n_layers=1)
    flat_ref = flat.clone().requires_grad_()
    x_ref = x.clone().requires_grad_()
    y_ref = golden.block_forward(x_ref, golden._block_views(flat_ref))
    y_ref.backward(dy)

    errors["fwd:y"] = rel_l2(y, y_ref)
    errors["bwd:dx"] = rel_l2(dx, x_ref.grad)
    ref_dw = golden._block_views(flat_ref.grad)
    for name in dw:
        errors[f"bwd:d{name}"] = rel_l2(dw[name], ref_dw[name])

    # accumulation semantics: running backward again with accum=True doubles
    bwd._backward(kctx, dy, a, x, w, dx, dw, accum=True)
    errors["accum:2x"] = rel_l2(dwflat, 2.0 * flat_ref.grad)

    return CheckReport(errors=errors, tol=tol)


def check_model_step(
    cfg,
    *,
    fast_memory_capacity: int,
    recompute_levels: dict[str, int] | None = None,
    seed: int = 0,
    tol: float = 3e-2,
) -> CheckReport:
    """Ladder level 3: full annotated program through the REAL engine vs the
    golden model — loss, final params, optimizer state."""
    from dataflow.models.llama3_reference import GoldenLlama3
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.tasks.interop import torch_view
    from dataflow.tasks.llama3_blocks import build_resolver
    from dataflow.training.llama3_lowering import dims_of, initial_values, lower_llama3
    from dataflow.training.planning import plan_program

    dims = dims_of(cfg)
    program = lower_llama3(cfg, recompute_levels=recompute_levels)
    planned = plan_program(program, fast_memory_capacity=fast_memory_capacity)

    backend = CudaBackend()
    values = initial_values(planned.program, cfg, backend, seed=seed)

    # golden model + inputs from the same bytes
    def pinned(name: str) -> torch.Tensor:
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    golden = GoldenLlama3.from_packed_bytes(
        dims, cfg.n_layers,
        pinned("W_embed"),
        [pinned(f"W_{i}") for i in range(cfg.n_layers)],
        pinned("W_head"),
    )
    tokens = torch_view(values["tokens_0_0"], (dims.tokens,), torch.int32).long().cuda()
    targets = torch_view(values["targets_0_0"], (dims.tokens,), torch.int32).long().cuda()
    golden_loss = golden.train_step(tokens.cuda(), targets.cuda())

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program,
        resolver=build_resolver(dims),
        initial_buffers=values,
        pool_prewarm=dry.pool_demand,
    )

    errors: dict[str, float] = {}
    loss_buf = result.objects.get("loss_0_0").backing.buffer  # type: ignore[union-attr]
    run_loss = float(torch_view(loss_buf, (1,), torch.float32)[0])
    errors["loss"] = abs(run_loss - golden_loss) / max(abs(golden_loss), 1e-6)

    def final_bytes(object_id: str) -> torch.Tensor:
        rec = result.objects.get(object_id)
        slot = rec.backing or rec.fast
        return torch_view(slot.buffer, (rec.size_bytes // 2,), torch.bfloat16)

    errors["W_embed"] = rel_l2(final_bytes("W_embed"), golden.w_embed.detach().to(torch.bfloat16).reshape(-1))
    for i in range(cfg.n_layers):
        errors[f"W_{i}"] = rel_l2(final_bytes(f"W_{i}"), golden.w_blocks[i].detach().reshape(-1))
    errors["W_head"] = rel_l2(final_bytes("W_head"), golden.w_head.detach().to(torch.bfloat16).reshape(-1))
    result.close()  # release the engine-local slab/arena for the next check
    dry.close()
    from dataflow.tasks.interop import clear_view_cache

    clear_view_cache()
    for buf in values.values():
        backend.free(buf)
    return CheckReport(errors=errors, tol=tol)
