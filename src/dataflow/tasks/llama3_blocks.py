"""Llama3 block executables: stateless task implementations over runtime buffers.

Buffer-order contract with the lowering (positional, per compute_block_key):

    embed_fwd        in (tokens, W_embed)              out (y,)
    block_fwd        in (x, W)                          out (y, A) | (y,) when recomputed
    block_recompute  in (x, W)                          out (A,)
    block_bwd        in (dy, A, x, W[, dW])             out (dx[, dW]) ; mutates dW when accumulating
    head_fwd         in (y_last, W_head)                out (logits,)
    loss_bwd         in (logits, targets)               out (dlogits, loss)
    head_bwd         in (dlogits, y_last, W_head[, dW]) out (dy_last[, dW]) ; mutates dW when accumulating
    embed_bwd        in (dy_embed, tokens[, dW])        out ([dW]) ; mutates dW when accumulating
    optimizer_*      in (W, dW, O) mutates (W, O)       block_params: step (1-based)

Every launch enqueues on ctx.stream via ExternalStream and never synchronizes.
Torch-allocator scratch inside a task is bounded and measured by the
profiling harness (the plan's measurement-authoritative workspace policy).

Elementwise/reduction ops dispatch through a pinned kernel registry set
(``dataflow.tasks.kernels``; fused Triton by default, eager fallback,
DATAFLOW_KERNELS=eager to force the baseline). GEMMs (cuBLAS), flash
attention and embed scatter/gather (aten) stay direct calls. The chosen
implementations are recorded via ``KernelSet.describe()`` — measured task
costs are measurements of a specific kernel set.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.core import TaskSpec
from dataflow.runtime.executable import TaskContext

from . import ops
from .interop import external_stream, torch_view
from .kernels import KernelCtx, KernelSet, resolve_kernels
from .layouts import LlamaDims, PackedLayout, adamw_state_layout, context_layout, weight_layout


@dataclass(frozen=True)
class AdamWHyper:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass(frozen=True)
class _Base:
    dims: LlamaDims
    kernels: KernelSet = None  # type: ignore[assignment]

    @property
    def wl(self) -> PackedLayout:
        return weight_layout(self.dims)

    @property
    def cl(self) -> PackedLayout:
        return context_layout(self.dims)

    def _stream_ctx(self, ctx: TaskContext):
        es = external_stream(ctx.stream)
        return es, KernelCtx(stream_handle=es.cuda_stream, torch_stream=es)

    def _in(self, ctx: TaskContext, i: int) -> object:
        return ctx.inputs[ctx.task.inputs[i]]

    def _out(self, ctx: TaskContext, i: int) -> object:
        return ctx.outputs[ctx.task.outputs[i].id]


@dataclass(frozen=True)
class EmbedFwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
            tokens = torch_view(self._in(ctx, 0), (d.tokens,), torch.int32)
            w = torch_view(self._in(ctx, 1), (d.vocab_size, d.d_model), torch.bfloat16)
            y = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            ops.embed_fwd(tokens, w, y)


@dataclass(frozen=True)
class BlockFwd(_Base):
    emit_context: bool = True

    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            x = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            w = self.wl.views(self._in(ctx, 1))
            emit_ctx = len(ctx.task.outputs) > 1 and self.emit_context
            if emit_ctx:
                y = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
                a = self.cl.views(self._out(ctx, 1))
            else:
                y = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
                a = None
            self._forward(kctx, x, w, y, a)

    def _forward(self, kctx, x, w, y, a) -> None:
        d = self.dims
        K = self.kernels
        h1 = torch.empty_like(x)
        rstd_attn = torch.empty(d.tokens, dtype=torch.float32, device=x.device)
        K.rmsnorm_fwd(kctx, x, w["attn_norm_w"], h1, rstd_attn)
        qm = h1 @ w["wq"]
        q = torch.empty_like(qm)
        K.rope_fwd(kctx, qm, q, d.seq_len, d.n_heads, d.head_dim, d.rope_base)
        km = h1 @ w["wk"]
        k = torch.empty_like(km)
        K.rope_fwd(kctx, km, k, d.seq_len, d.n_kv_heads, d.head_dim, d.rope_base)
        v = h1 @ w["wv"]
        attn_out, lse = ops.flash_fwd(q, k, v, d.n_heads, d.n_kv_heads, d.head_dim, d.seq_len)
        h_mid = x + attn_out @ w["wo"]
        h2 = torch.empty_like(h_mid)
        rstd_ffn = torch.empty(d.tokens, dtype=torch.float32, device=x.device)
        K.rmsnorm_fwd(kctx, h_mid, w["ffn_norm_w"], h2, rstd_ffn)
        x1 = h2 @ w["w1"]
        x3 = h2 @ w["w3"]
        s = torch.empty_like(x1)
        K.swiglu_fwd_out(kctx, x1, x3, s)
        y.copy_(h_mid + s @ w["w2"])
        if a is not None:
            a["rstd_attn"].copy_(rstd_attn)
            a["q"].copy_(q)
            a["k"].copy_(k)
            a["v"].copy_(v)
            a["lse"].copy_(lse)
            a["attn_out"].copy_(attn_out)
            a["h_mid"].copy_(h_mid)
            a["rstd_ffn"].copy_(rstd_ffn)
            a["x1"].copy_(x1)
            a["x3"].copy_(x3)


@dataclass(frozen=True)
class BlockRecompute(BlockFwd):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            x = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            w = self.wl.views(self._in(ctx, 1))
            a = self.cl.views(self._out(ctx, 0))
            self._forward_context(kctx, x, w, a)

    def _forward_context(self, kctx, x, w, a) -> None:
        """Rebuild ONLY the saved context. The backward never reads the
        block output y (the next block's x is saved separately) and rebuilds
        s from x1/x3 itself, so recompute stops at the w1/w3 GEMMs — the
        swiglu, down-projection GEMM, residual add, and y copy of the full
        forward are pure waste here (~15-20% of the task's FLOPs)."""
        d = self.dims
        K = self.kernels
        h1 = torch.empty_like(x)
        rstd_attn = torch.empty(d.tokens, dtype=torch.float32, device=x.device)
        K.rmsnorm_fwd(kctx, x, w["attn_norm_w"], h1, rstd_attn)
        qm = h1 @ w["wq"]
        q = torch.empty_like(qm)
        K.rope_fwd(kctx, qm, q, d.seq_len, d.n_heads, d.head_dim, d.rope_base)
        km = h1 @ w["wk"]
        k = torch.empty_like(km)
        K.rope_fwd(kctx, km, k, d.seq_len, d.n_kv_heads, d.head_dim, d.rope_base)
        v = h1 @ w["wv"]
        attn_out, lse = ops.flash_fwd(q, k, v, d.n_heads, d.n_kv_heads, d.head_dim, d.seq_len)
        h_mid = x + attn_out @ w["wo"]
        h2 = torch.empty_like(h_mid)
        rstd_ffn = torch.empty(d.tokens, dtype=torch.float32, device=x.device)
        K.rmsnorm_fwd(kctx, h_mid, w["ffn_norm_w"], h2, rstd_ffn)
        a["rstd_attn"].copy_(rstd_attn)
        a["q"].copy_(q)
        a["k"].copy_(k)
        a["v"].copy_(v)
        a["lse"].copy_(lse)
        a["attn_out"].copy_(attn_out)
        a["h_mid"].copy_(h_mid)
        a["rstd_ffn"].copy_(rstd_ffn)
        torch.matmul(h2, w["w1"], out=a["x1"])
        torch.matmul(h2, w["w3"], out=a["x3"])


@dataclass(frozen=True)
class BlockBwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            dy = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            a = self.cl.views(self._in(ctx, 1))
            x = torch_view(self._in(ctx, 2), (d.tokens, d.d_model), torch.bfloat16)
            w = self.wl.views(self._in(ctx, 3))
            accum = bool(ctx.task.mutates)
            if accum:
                dw = self.wl.views(ctx.mutates[ctx.task.mutates[0]])
                dx = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            else:
                dx = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
                dw = self.wl.views(self._out(ctx, 1))
            self._backward(kctx, dy, a, x, w, dx, dw, accum)

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum: bool) -> None:
        d = self.dims
        K = self.kernels

        def acc(name: str, value: torch.Tensor) -> None:
            if accum:
                dw[name].add_(value.to(dw[name].dtype))
            else:
                dw[name].copy_(value.to(dw[name].dtype))

        def norm_bwd(dyv, xv, rstd, wv):
            dxv = torch.empty_like(xv)
            dwv = torch.empty(d.d_model, dtype=torch.float32, device=xv.device)
            K.rmsnorm_bwd(kctx, dyv, xv, rstd, wv, dxv, dwv)
            return dxv, dwv

        # --- mlp ---
        h2 = torch.empty_like(a["h_mid"])
        K.rmsnorm_apply(kctx, a["h_mid"], a["rstd_ffn"], w["ffn_norm_w"], h2)
        s = torch.empty_like(a["x1"])
        K.swiglu_fwd_out(kctx, a["x1"], a["x3"], s)
        ds = dy @ w["w2"].T
        acc("w2", s.T @ dy)
        dx1 = torch.empty_like(a["x1"])
        dx3 = torch.empty_like(a["x3"])
        K.swiglu_bwd(kctx, ds, a["x1"], a["x3"], dx1, dx3)
        del s
        acc("w1", h2.T @ dx1)
        acc("w3", h2.T @ dx3)
        dh2 = dx1 @ w["w1"].T + dx3 @ w["w3"].T
        dh_mid_n, dffn_norm = norm_bwd(dh2, a["h_mid"], a["rstd_ffn"], w["ffn_norm_w"])
        acc("ffn_norm_w", dffn_norm)
        dh_mid = dy + dh_mid_n

        # --- attention ---
        d_attn = dh_mid @ w["wo"].T
        acc("wo", a["attn_out"].T @ dh_mid)
        dq, dk, dv = ops.flash_bwd(
            d_attn, a["q"], a["k"], a["v"], a["attn_out"], a["lse"],
            d.n_heads, d.n_kv_heads, d.head_dim, d.seq_len,
        )
        dq_r = torch.empty_like(dq)
        K.rope_bwd(kctx, dq, dq_r, d.seq_len, d.n_heads, d.head_dim, d.rope_base)
        dk_r = torch.empty_like(dk)
        K.rope_bwd(kctx, dk, dk_r, d.seq_len, d.n_kv_heads, d.head_dim, d.rope_base)
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        acc("wq", h1.T @ dq_r)
        acc("wk", h1.T @ dk_r)
        acc("wv", h1.T @ dv)
        dh1 = dq_r @ w["wq"].T + dk_r @ w["wk"].T + dv @ w["wv"].T
        dx_n, dattn_norm = norm_bwd(dh1, x, a["rstd_attn"], w["attn_norm_w"])
        acc("attn_norm_w", dattn_norm)
        dx_out.copy_(dh_mid + dx_n)


@dataclass(frozen=True)
class HeadFwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            y = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            wh = torch_view(self._in(ctx, 1), (d.vocab_size, d.d_model), torch.bfloat16)
            logits = torch_view(self._out(ctx, 0), (d.tokens, d.vocab_size), torch.bfloat16)
            yn = torch.empty_like(y)
            rstd = torch.empty(d.tokens, dtype=torch.float32, device=y.device)
            self.kernels.rmsnorm_noweight(kctx, y, yn, rstd)  # final model norm
            torch.matmul(yn, wh.T, out=logits)


@dataclass(frozen=True)
class LossBwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            logits = torch_view(self._in(ctx, 0), (d.tokens, d.vocab_size), torch.bfloat16)
            targets = torch_view(self._in(ctx, 1), (d.tokens,), torch.int32)
            dlogits = torch_view(self._out(ctx, 0), (d.tokens, d.vocab_size), torch.bfloat16)
            loss = torch_view(self._out(ctx, 1), (1,), torch.float32)
            self.kernels.ce_loss_fwd_bwd(kctx, logits, targets, loss, dlogits)


@dataclass(frozen=True)
class HeadBwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            K = self.kernels
            dlogits = torch_view(self._in(ctx, 0), (d.tokens, d.vocab_size), torch.bfloat16)
            y = torch_view(self._in(ctx, 1), (d.tokens, d.d_model), torch.bfloat16)
            wh = torch_view(self._in(ctx, 2), (d.vocab_size, d.d_model), torch.bfloat16)
            accum = bool(ctx.task.mutates)
            dy = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            # final-norm recompute (cheap: one reduce over y) + backward
            yn = torch.empty_like(y)
            rstd = torch.empty(d.tokens, dtype=torch.float32, device=y.device)
            K.rmsnorm_noweight(kctx, y, yn, rstd)
            dyn = dlogits @ wh
            ones = torch.ones(d.d_model, device=dyn.device, dtype=torch.bfloat16)
            dw_scratch = torch.empty(d.d_model, dtype=torch.float32, device=y.device)
            K.rmsnorm_bwd(kctx, dyn, y, rstd, ones, dy, dw_scratch)
            if accum:
                dwh = torch_view(ctx.mutates[ctx.task.mutates[0]], (d.vocab_size, d.d_model), torch.bfloat16)
                dwh.add_(dlogits.T @ yn)
            else:
                dwh = torch_view(self._out(ctx, 1), (d.vocab_size, d.d_model), torch.bfloat16)
                torch.matmul(dlogits.T, yn, out=dwh)  # write straight into dW


@dataclass(frozen=True)
class EmbedBwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
            dy = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            tokens = torch_view(self._in(ctx, 1), (d.tokens,), torch.int32)
            accum = bool(ctx.task.mutates)
            if accum:
                dwe = torch_view(ctx.mutates[ctx.task.mutates[0]], (d.vocab_size, d.d_model), torch.bfloat16)
            else:
                dwe = torch_view(self._out(ctx, 0), (d.vocab_size, d.d_model), torch.bfloat16)
            ops.embed_bwd_accum(tokens, dy, dwe, zero_first=not accum)


@dataclass(frozen=True)
class AdamWStep(_Base):
    hyper: AdamWHyper = AdamWHyper()

    def launch(self, ctx: TaskContext) -> None:
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            w_buf = ctx.mutates[ctx.task.mutates[0]]
            g_buf = ctx.inputs[ctx.task.inputs[1]]
            o_buf = ctx.mutates[ctx.task.mutates[1]]
            elems = w_buf.size_bytes // 2  # bf16 params
            w = torch_view(w_buf, (elems,), torch.bfloat16)
            g = torch_view(g_buf, (elems,), torch.bfloat16)
            ol = adamw_state_layout(elems)
            m = ol.view(o_buf, "m")
            v = ol.view(o_buf, "v")
            step = int(ctx.task.block_params.get("step", 0)) + 1
            hp = self.hyper
            self.kernels.adamw_step(
                kctx, w, g, m, v,
                lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                weight_decay=hp.weight_decay, step=step,
            )


def build_resolver(
    dims: LlamaDims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    """Executable resolver keyed by compute_block_key — planner-inserted
    recompute tasks bind automatically. ``kernels`` pins the op
    implementations for every executable this resolver hands out (default:
    best available); ``resolver.kernel_set.describe()`` is the provenance
    record to stamp into profiles and reports."""
    kernels = kernels if kernels is not None else resolve_kernels()
    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "block_fwd": BlockFwd(dims, kernels),
        "block_recompute": BlockRecompute(dims, kernels),
        "block_bwd": BlockBwd(dims, kernels),
        "head_fwd": HeadFwd(dims, kernels),
        "loss_bwd": LossBwd(dims, kernels),
        "head_bwd": HeadBwd(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(dims, kernels, hyper),
        "optimizer_embed": AdamWStep(dims, kernels, hyper),
        "optimizer_head": AdamWStep(dims, kernels, hyper),
    }

    def resolver(task: TaskSpec):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key {key!r} (task {task.id!r})")
        return table[key]

    resolver.kernel_set = kernels
    return resolver
