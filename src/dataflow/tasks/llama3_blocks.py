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
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.core import TaskSpec
from dataflow.runtime.executable import TaskContext

from . import ops
from .interop import external_stream, torch_view
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

    @property
    def wl(self) -> PackedLayout:
        return weight_layout(self.dims)

    @property
    def cl(self) -> PackedLayout:
        return context_layout(self.dims)

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
        with torch.cuda.stream(external_stream(ctx.stream)):
            x = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            w = self.wl.views(self._in(ctx, 1))
            emit_ctx = len(ctx.task.outputs) > 1 and self.emit_context
            if emit_ctx:
                y = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
                a = self.cl.views(self._out(ctx, 1))
            else:
                y = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
                a = None
            self._forward(x, w, y, a)

    def _forward(self, x, w, y, a) -> None:
        d = self.dims
        h1 = torch.empty_like(x)
        rstd_attn = torch.empty(d.tokens, dtype=torch.float32, device=x.device)
        ops.rmsnorm_fwd(x, w["attn_norm_w"], h1, rstd_attn)
        q = ops.rope_fwd(h1 @ w["wq"], d.seq_len, d.n_heads, d.head_dim, d.rope_base)
        k = ops.rope_fwd(h1 @ w["wk"], d.seq_len, d.n_kv_heads, d.head_dim, d.rope_base)
        v = h1 @ w["wv"]
        attn_out, lse = ops.flash_fwd(q, k, v, d.n_heads, d.n_kv_heads, d.head_dim)
        h_mid = x + attn_out @ w["wo"]
        h2 = torch.empty_like(h_mid)
        rstd_ffn = torch.empty(d.tokens, dtype=torch.float32, device=x.device)
        ops.rmsnorm_fwd(h_mid, w["ffn_norm_w"], h2, rstd_ffn)
        x1 = h2 @ w["w1"]
        x3 = h2 @ w["w3"]
        y.copy_(h_mid + ops.swiglu_fwd(x1, x3) @ w["w2"])
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
        with torch.cuda.stream(external_stream(ctx.stream)):
            x = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            w = self.wl.views(self._in(ctx, 1))
            a = self.cl.views(self._out(ctx, 0))
            y_scratch = torch.empty_like(x)
            self._forward(x, w, y_scratch, a)


@dataclass(frozen=True)
class BlockBwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
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
            self._backward(dy, a, x, w, dx, dw, accum)

    def _backward(self, dy, a, x, w, dx_out, dw, accum: bool) -> None:
        d = self.dims

        def acc(name: str, value: torch.Tensor) -> None:
            if accum:
                dw[name].add_(value.to(dw[name].dtype))
            else:
                dw[name].copy_(value.to(dw[name].dtype))

        # --- mlp ---
        h2 = ops.rmsnorm_apply(a["h_mid"], a["rstd_ffn"], w["ffn_norm_w"])
        s = ops.swiglu_fwd(a["x1"], a["x3"])
        ds = dy @ w["w2"].T
        acc("w2", s.T @ dy)
        dx1, dx3 = ops.swiglu_bwd(ds, a["x1"], a["x3"])
        acc("w1", h2.T @ dx1)
        acc("w3", h2.T @ dx3)
        dh2 = dx1 @ w["w1"].T + dx3 @ w["w3"].T
        dh_mid_n, dffn_norm = ops.rmsnorm_bwd(dh2, a["h_mid"], a["rstd_ffn"], w["ffn_norm_w"])
        acc("ffn_norm_w", dffn_norm)
        dh_mid = dy + dh_mid_n

        # --- attention ---
        d_attn = dh_mid @ w["wo"].T
        acc("wo", a["attn_out"].T @ dh_mid)
        dq, dk, dv = ops.flash_bwd(
            d_attn, a["q"], a["k"], a["v"], a["attn_out"], a["lse"],
            d.n_heads, d.n_kv_heads, d.head_dim,
        )
        dq = ops.rope_bwd(dq, d.seq_len, d.n_heads, d.head_dim, d.rope_base)
        dk = ops.rope_bwd(dk, d.seq_len, d.n_kv_heads, d.head_dim, d.rope_base)
        h1 = ops.rmsnorm_apply(x, a["rstd_attn"], w["attn_norm_w"])
        acc("wq", h1.T @ dq)
        acc("wk", h1.T @ dk)
        acc("wv", h1.T @ dv)
        dh1 = dq @ w["wq"].T + dk @ w["wk"].T + dv @ w["wv"].T
        dx_n, dattn_norm = ops.rmsnorm_bwd(dh1, x, a["rstd_attn"], w["attn_norm_w"])
        acc("attn_norm_w", dattn_norm)
        dx_out.copy_(dh_mid + dx_n)


@dataclass(frozen=True)
class HeadFwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
            y = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            wh = torch_view(self._in(ctx, 1), (d.vocab_size, d.d_model), torch.bfloat16)
            logits = torch_view(self._out(ctx, 0), (d.tokens, d.vocab_size), torch.bfloat16)
            torch.matmul(y, wh.T, out=logits)


@dataclass(frozen=True)
class LossBwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
            logits = torch_view(self._in(ctx, 0), (d.tokens, d.vocab_size), torch.bfloat16)
            targets = torch_view(self._in(ctx, 1), (d.tokens,), torch.int32)
            dlogits = torch_view(self._out(ctx, 0), (d.tokens, d.vocab_size), torch.bfloat16)
            loss = torch_view(self._out(ctx, 1), (1,), torch.float32)
            ops.ce_loss_fwd_bwd(logits, targets, loss, dlogits)


@dataclass(frozen=True)
class HeadBwd(_Base):
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
            dlogits = torch_view(self._in(ctx, 0), (d.tokens, d.vocab_size), torch.bfloat16)
            y = torch_view(self._in(ctx, 1), (d.tokens, d.d_model), torch.bfloat16)
            wh = torch_view(self._in(ctx, 2), (d.vocab_size, d.d_model), torch.bfloat16)
            accum = bool(ctx.task.mutates)
            dy = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            torch.matmul(dlogits, wh, out=dy)
            dwh_new = dlogits.T @ y
            if accum:
                dwh = torch_view(ctx.mutates[ctx.task.mutates[0]], (d.vocab_size, d.d_model), torch.bfloat16)
                dwh.add_(dwh_new)
            else:
                dwh = torch_view(self._out(ctx, 1), (d.vocab_size, d.d_model), torch.bfloat16)
                dwh.copy_(dwh_new)


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
        with torch.cuda.stream(external_stream(ctx.stream)):
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
            ops.adamw_step(
                w, g, m, v,
                lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                weight_decay=hp.weight_decay, step=step,
            )


def build_resolver(dims: LlamaDims, hyper: AdamWHyper = AdamWHyper()):
    """Executable resolver keyed by compute_block_key — planner-inserted
    recompute tasks bind automatically."""
    table = {
        "embed_fwd": EmbedFwd(dims),
        "block_fwd": BlockFwd(dims),
        "block_recompute": BlockRecompute(dims),
        "block_bwd": BlockBwd(dims),
        "head_fwd": HeadFwd(dims),
        "loss_bwd": LossBwd(dims),
        "head_bwd": HeadBwd(dims),
        "embed_bwd": EmbedBwd(dims),
        "optimizer_block": AdamWStep(dims, hyper),
        "optimizer_embed": AdamWStep(dims, hyper),
        "optimizer_head": AdamWStep(dims, hyper),
    }

    def resolver(task: TaskSpec):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key {key!r} (task {task.id!r})")
        return table[key]

    return resolver
