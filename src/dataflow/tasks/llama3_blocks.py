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
from .layouts import (
    LlamaDims,
    PackedLayout,
    context_layout,
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
    opt_state_layout,
    weight_layout,
)


@dataclass(frozen=True)
class AdamWHyper:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.95      # sgdm/muon (tasks/optim.py); unused by adamw
    muon_lr: float | None = None  # muon fields use this when set (else lr)
    # lr(step) schedule (tasks/optim.py LRSchedule). None (default) =
    # no scaling; LRSchedule() = constant (debug-consistent); use
    # LRSchedule("wsd", warmup_steps=W, total_steps=T) for training.
    schedule: object = None


@dataclass(frozen=True)
class _Base:
    dims: LlamaDims
    kernels: KernelSet = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        """Family/kind weight layout; subclasses override. ``layer`` selects
        a depth-dependent dtype sub-policy (None = base policy)."""
        return weight_layout(self.dims, layer=layer)

    @property
    def wl(self) -> PackedLayout:
        return self._weight_layout()

    @property
    def gl(self) -> PackedLayout:
        """dW layout: mirrors wl field-by-field at the policy's grad dtypes
        (identical to wl under the default all-bf16 policy)."""
        return grad_layout(self.wl, self.dims.dtypes)

    @staticmethod
    def layer_of(task) -> int | None:
        """Layer index of a block-scoped task, derived from its W_{i}
        object (inputs for fwd/recompute/bwd, mutates for the optimizer).
        None for loose tasks (embed/head/loss)."""
        for oid in tuple(task.inputs) + tuple(task.mutates):
            if oid.startswith("W_") and oid not in ("W_embed", "W_head"):
                return int(oid.split("_")[1])
        return None

    def wl_for(self, task) -> PackedLayout:
        return self._weight_layout(self.layer_of(task))

    def gl_for(self, task) -> PackedLayout:
        layer = self.layer_of(task)
        return grad_layout(self._weight_layout(layer), self.dims.dtypes, layer=layer)

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

    # --- shared backward closures ---------------------------------------------
    # Every family backward re-created these two closures verbatim; they are
    # the create-vs-accumulate grad writer and the rmsnorm backward step.

    def _acc_fn(self, dw, accum: bool):
        def acc(name: str, value: torch.Tensor) -> None:
            if accum:
                dw[name].add_(value.to(dw[name].dtype))
            else:
                dw[name].copy_(value.to(dw[name].dtype))

        return acc

    def _meta_state(self, ctx) -> dict | None:
        """Family hook: st entries for METADATA objects (M_{s}_{r}_{i} —
        never-recompute forward artifacts: routing packs, selections).
        None = family has no metadata. Implementations inspect
        ctx.task.compute_block_key to set meta_ready for recompute."""
        return None

    def _norm_bwd_fn(self, kctx):
        K = self.kernels

        def norm_bwd(dyv, xv, rstd, wv):
            dxv = torch.empty_like(xv)
            dwv = torch.empty(wv.numel(), dtype=torch.float32, device=xv.device)
            K.rmsnorm_bwd(kctx, dyv, xv, rstd, wv, dxv, dwv)
            return dxv, dwv

        return norm_bwd


@dataclass(frozen=True)
class EmbedFwd(_Base):
    """Token-embedding lookup: tokens + W_embed -> the first hidden
    state."""
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
            tokens = torch_view(self._in(ctx, 0), (d.tokens,), torch.int32)
            w = torch_view(self._in(ctx, 1), (d.vocab_size, d.d_model), torch.bfloat16)
            y = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            ops.embed_fwd(tokens, w, y)


@dataclass(frozen=True)
class BlockFwd(_Base):
    """Transformer-block forward: runs the STAGES list, writing saved
    context (and metadata) fields through to the ctx buffers."""
    emit_context: bool = True

    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            x = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            w = self.wl_for(ctx.task).views(self._in(ctx, 1))
            y = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            a = None
            if self.emit_context:
                # A located by id, not position: metadata families append
                # M_ outputs after it (or drop A entirely under recompute
                # planning while keeping the metadata)
                for j, o in enumerate(ctx.task.outputs[1:], start=1):
                    if o.id.startswith("A_"):
                        a = self.cl.views(self._out(ctx, j))
                        break
            self._forward(kctx, x, w, y, a, extras=self._meta_state(ctx))

    # --- staged forward -------------------------------------------------------
    #
    # The forward is authored as an ordered list of named STAGES, each
    # declaring which saved-context fields it emits. Everything downstream
    # derives from this single description:
    #   - full forward  = run every stage (+ the y epilogue)
    #   - recompute     = run stages up to the LAST context-emitting stage —
    #                     derived, never hand-written, so "work past the
    #                     boundary" (e.g. the down-projection GEMM) cannot
    #                     creep back in when blocks evolve
    # A stage receives (kctx, kernels, dims, st) where ``st`` is the shared
    # state dict; it reads/writes intermediates there and writes any emitted
    # fields into st["a"] when a context is attached.

    @staticmethod
    def _stage_attn_norm(kctx, K, d, st):
        x = st["x"]
        h1 = torch.empty_like(x)
        rstd = torch.empty(d.tokens, dtype=torch.float32, device=x.device)
        K.rmsnorm_fwd(kctx, x, st["w"]["attn_norm_w"], h1, rstd)
        st["h1"] = h1
        if st["a"] is not None:
            st["a"]["rstd_attn"].copy_(rstd)

    @staticmethod
    def _stage_qkv_rope(kctx, K, d, st):
        # write-through: rope/GEMM outputs land DIRECTLY in the ctx views
        # when a context is attached — no scratch double + no copy pass
        h1, w, a = st["h1"], st["w"], st["a"]
        qm = h1 @ w["wq"]
        q = a["q"] if a is not None else torch.empty_like(qm)
        pos = ops.positions_for(d.seq_spec, qm.shape[0], qm.device)
        K.rope_fwd(kctx, qm, q, pos, d.n_heads, d.head_dim, d.rope_base)
        km = h1 @ w["wk"]
        k = a["k"] if a is not None else torch.empty_like(km)
        K.rope_fwd(kctx, km, k, pos, d.n_kv_heads, d.head_dim, d.rope_base)
        if a is not None:
            v = a["v"]
            torch.matmul(h1, w["wv"], out=v)
        else:
            v = h1 @ w["wv"]
        st.pop("h1")
        st.update(q=q, k=k, v=v)

    @staticmethod
    def _stage_attn(kctx, K, d, st):
        attn_out, lse = ops.flash_fwd(
            st["q"], st["k"], st["v"], d.n_heads, d.n_kv_heads, d.head_dim, d.seq_spec
        )
        st.pop("q"), st.pop("k"), st.pop("v")
        st["attn_out"] = attn_out
        if st["a"] is not None:
            st["a"]["lse"].copy_(lse)
            st["a"]["attn_out"].copy_(attn_out)

    @staticmethod
    def _stage_resid1_norm2(kctx, K, d, st):
        a = st["a"]
        if a is not None:
            h_mid = a["h_mid"]
            torch.addmm(st["x"], st["attn_out"], st["w"]["wo"], out=h_mid)
        else:
            h_mid = st["x"] + st["attn_out"] @ st["w"]["wo"]
        h2 = torch.empty_like(h_mid)
        rstd = torch.empty(d.tokens, dtype=torch.float32, device=h_mid.device)
        K.rmsnorm_fwd(kctx, h_mid, st["w"]["ffn_norm_w"], h2, rstd)
        st.pop("attn_out")
        st.update(h_mid=h_mid, h2=h2)
        if a is not None:
            a["rstd_ffn"].copy_(rstd)

    @staticmethod
    def _stage_up_proj(kctx, K, d, st):
        h2, w, a = st["h2"], st["w"], st["a"]
        if a is not None:
            torch.matmul(h2, w["w1"], out=a["x1"])
            torch.matmul(h2, w["w3"], out=a["x3"])
            st["x1"], st["x3"] = a["x1"], a["x3"]
        else:
            st["x1"] = h2 @ w["w1"]
            st["x3"] = h2 @ w["w3"]
        st.pop("h2")

    @staticmethod
    def _stage_swiglu(kctx, K, d, st):
        s_out = torch.empty_like(st["x1"])
        K.swiglu_fwd_out(kctx, st["x1"], st["x3"], s_out)
        st.pop("x1"), st.pop("x3")
        st["s"] = s_out

    @staticmethod
    def _stage_down_resid(kctx, K, d, st):
        torch.addmm(st["h_mid"], st.pop("s"), st["w"]["w2"], out=st["y"])
        st.pop("h_mid")

    # (stage_name, fn, emitted context fields)
    STAGES = (
        ("attn_norm", _stage_attn_norm.__func__, ("rstd_attn",)),
        ("qkv_rope", _stage_qkv_rope.__func__, ("q", "k", "v")),
        ("attn", _stage_attn.__func__, ("lse", "attn_out")),
        ("resid1_norm2", _stage_resid1_norm2.__func__, ("h_mid", "rstd_ffn")),
        ("up_proj", _stage_up_proj.__func__, ("x1", "x3")),
        ("swiglu", _stage_swiglu.__func__, ()),
        ("down_resid", _stage_down_resid.__func__, ()),
    )

    @classmethod
    def recompute_stage_count(cls) -> int:
        """Derived recompute boundary: stages up to the LAST one that emits
        a context field. Everything after it exists only to produce y.
        Stage entries may carry an optional 4th element "meta" marking a
        METADATA-producing stage — skipped entirely when the M object is
        supplied (recompute repopulates ONLY the A objects)."""
        last = 0
        for i, entry in enumerate(cls.STAGES):
            if entry[2]:
                last = i
        return last + 1

    @classmethod
    def context_fields_emitted(cls) -> set:
        return {f for entry in cls.STAGES for f in entry[2]}

    def _run_stages(self, kctx, x, w, a, *, count: int, y=None,
                    extras=None) -> None:
        st = {"x": x, "w": w, "a": a, "y": y}
        if extras:
            st.update(extras)
        skip_meta = bool(st.get("meta_ready"))
        for entry in self.STAGES[:count]:
            if skip_meta and len(entry) > 3 and entry[3] == "meta":
                continue  # metadata is supplied, never recomputed
            entry[1](kctx, self.kernels, self.dims, st)

    def _forward(self, kctx, x, w, y, a, extras=None) -> None:
        self._run_stages(kctx, x, w, a, count=len(self.STAGES), y=y,
                         extras=extras)


@dataclass(frozen=True)
class BlockRecompute(BlockFwd):
    """Derived recompute: replays the forward stages through the last
    context-emitting one to repopulate A from the block input."""
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            x = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            w = self.wl_for(ctx.task).views(self._in(ctx, 1))
            a = self.cl.views(self._out(ctx, 0))
            self._forward_context(kctx, x, w, a, extras=self._meta_state(ctx))

    def _forward_context(self, kctx, x, w, a, extras=None) -> None:
        """DERIVED from the stage list: run through the last context-emitting
        stage and stop. The block output y is never a backward dependency,
        so the trailing stages (swiglu, down-projection, residual) are
        skipped by construction — not by hand-maintained duplication."""
        self._run_stages(kctx, x, w, a, count=self.recompute_stage_count(),
                         extras=extras)


def dense_mlp_tail_bwd(kctx, K, dy, h_mid, rstd_ffn, x1, x3, w, acc, norm_bwd):
    """Dense SwiGLU MLP-tail backward, shared by every dense family/kind
    (previously duplicated verbatim in four `_backward` methods).

    Consumes the saved tail context — the post-attention residual (llama's
    ``h_mid`` / qwen3.5's ``xo``), ``rstd_ffn`` and the up-projections
    ``x1``/``x3`` — plus the packed weight views; accumulates
    w1/w3/w2/ffn_norm_w through ``acc`` and returns the residual-stream
    gradient WITH the incoming ``dy`` added. Kernel-call order is
    byte-identical to the per-family copies it replaced.

    Scratch discipline (M5.2): del each (t, .) temporary at last use so the
    caching allocator recycles it within the task; additive joins run in
    place (addmm_/add_, the forward's epilogue convention).
    """
    h2 = torch.empty_like(h_mid)
    K.rmsnorm_apply(kctx, h_mid, rstd_ffn, w["ffn_norm_w"], h2)
    s = torch.empty_like(x1)
    K.swiglu_fwd_out(kctx, x1, x3, s)
    ds = dy @ w["w2"].T
    acc("w2", s.T @ dy)
    del s
    dx1 = torch.empty_like(x1)
    dx3 = torch.empty_like(x3)
    K.swiglu_bwd(kctx, ds, x1, x3, dx1, dx3)
    del ds
    acc("w1", h2.T @ dx1)
    acc("w3", h2.T @ dx3)
    del h2
    dh2 = dx1 @ w["w1"].T
    dh2.addmm_(dx3, w["w3"].T)
    del dx1, dx3
    dh_mid, dffn_norm = norm_bwd(dh2, h_mid, rstd_ffn, w["ffn_norm_w"])
    del dh2
    acc("ffn_norm_w", dffn_norm)
    dh_mid.add_(dy)
    return dh_mid


@dataclass(frozen=True)
class BlockBwd(_Base):
    """Transformer-block backward: MLP-tail then attention backward, per
    the family template; creates dW on round 0, accumulates after."""
    # ctx field holding the post-attention residual the MLP tail reads
    # (plain class attr, not a dataclass field)
    MLP_RESID_FIELD = "h_mid"

    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            dy = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            a = self.cl.views(self._in(ctx, 1))
            x = torch_view(self._in(ctx, 2), (d.tokens, d.d_model), torch.bfloat16)
            w = self.wl_for(ctx.task).views(self._in(ctx, 3))
            accum = bool(ctx.task.mutates) and ctx.task.mutates[0].startswith("dW_")
            if accum:
                dw = self.gl_for(ctx.task).views(ctx.mutates[ctx.task.mutates[0]])
                dx = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            else:
                dw = None
                for j, o in enumerate(ctx.task.outputs[1:], start=1):
                    if o.id.startswith("dW_"):
                        dw = self.gl_for(ctx.task).views(self._out(ctx, j))
                        break
                dx = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            meta = self._meta_state(ctx)
            if meta is None:
                self._backward(kctx, dy, a, x, w, dx, dw, accum)
            else:
                self._backward(kctx, dy, a, x, w, dx, dw, accum, meta=meta)

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum: bool) -> None:
        """Template: MLP tail (shared helper, swappable per family) then the
        family's attention part. Kernel-call order unchanged from the
        pre-split monolith. ``dw``/``accum`` reach ``_mlp_bwd`` beyond the
        ``acc`` closure because MoE tails hand their stacked expert fields
        to grouped wgrads directly (create-vs-accumulate inside the op)."""
        acc = self._acc_fn(dw, accum)
        norm_bwd = self._norm_bwd_fn(kctx)
        dh_mid = self._mlp_bwd(kctx, dy, a, w, dw, accum, acc, norm_bwd)
        self._attn_bwd(kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        return dense_mlp_tail_bwd(
            kctx, self.kernels, dy, a[self.MLP_RESID_FIELD], a["rstd_ffn"],
            a["x1"], a["x3"], w, acc, norm_bwd,
        )

    def _attn_bwd(self, kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out) -> None:
        d = self.dims
        K = self.kernels
        d_attn = dh_mid @ w["wo"].T
        acc("wo", a["attn_out"].T @ dh_mid)
        dq, dk, dv = ops.flash_bwd(
            d_attn, a["q"], a["k"], a["v"], a["attn_out"], a["lse"],
            d.n_heads, d.n_kv_heads, d.head_dim, d.seq_spec,
        )
        del d_attn
        dq_r = torch.empty_like(dq)
        pos = ops.positions_for(d.seq_spec, dq.shape[0], dq.device)
        K.rope_bwd(kctx, dq, dq_r, pos, d.n_heads, d.head_dim, d.rope_base)
        del dq
        dk_r = torch.empty_like(dk)
        K.rope_bwd(kctx, dk, dk_r, pos, d.n_kv_heads, d.head_dim, d.rope_base)
        del dk
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        acc("wq", h1.T @ dq_r)
        acc("wk", h1.T @ dk_r)
        acc("wv", h1.T @ dv)
        del h1
        dh1 = dq_r @ w["wq"].T
        dh1.addmm_(dk_r, w["wk"].T)
        dh1.addmm_(dv, w["wv"].T)
        del dq_r, dk_r, dv
        dx_n, dattn_norm = norm_bwd(dh1, x, a["rstd_attn"], w["attn_norm_w"])
        del dh1
        acc("attn_norm_w", dattn_norm)
        torch.add(dh_mid, dx_n, out=dx_out)


HEAD_CHUNK_SCRATCH_BYTES = 512 << 20  # per (chunk, vocab) bf16 buffer


def head_chunk_rows(vocab_size: int) -> int:
    """Token-chunk size for the fused head: bounds each internal
    (chunk, vocab) bf16 buffer (logits, dlogits) to
    ~HEAD_CHUNK_SCRATCH_BYTES. Deterministic in the dims, so profiled
    costs are stable."""
    rows = HEAD_CHUNK_SCRATCH_BYTES // (2 * vocab_size)
    return max(256, (rows // 256) * 256)


@dataclass(frozen=True)
class HeadLoss(_Base):
    """Fused final-norm + LM head + CE loss + head backward, micro-chunked
    over tokens (flextrain-inspired). HARD INVARIANT — the memory model's
    point: no (tokens, vocab) tensor is EVER materialized; logits/dlogits
    exist only as (chunk, vocab) scratch inside one loop iteration.

    Buffer contract: inputs (y_last, targets, W_head-or-tied-W_embed
    [, dW accum rounds]); outputs (dy_last, loss [, dW on round 0]).
    Chunking numerics: per-row CE math is exact (rows are independent;
    dlogits normalize by the FULL token count via total_rows); the head
    dW accumulates across chunks in its grad STORAGE dtype — the same
    convention as grad-accum rounds.
    """

    @property
    def hl(self) -> PackedLayout:
        return head_weight_layout(self.dims)

    @property
    def hgl(self) -> PackedLayout:
        return grad_layout(self.hl, self.dims.dtypes, ns="head")

    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            K = self.kernels
            y = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            targets = torch_view(self._in(ctx, 1), (d.tokens,), torch.int32)
            wh = self.hl.views(self._in(ctx, 2))
            accum = bool(ctx.task.mutates)
            dy = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            loss = torch_view(self._out(ctx, 1), (1,), torch.float32)
            dwh = self.hgl.views(
                ctx.mutates[ctx.task.mutates[0]] if accum else self._out(ctx, 2)
            )
            chunk = head_chunk_rows(d.vocab_size)
            loss_acc = torch.zeros(1, dtype=torch.float32, device=y.device)
            part = torch.empty(1, dtype=torch.float32, device=y.device)
            dnorm_acc = torch.zeros(d.d_model, dtype=torch.float32, device=y.device)
            dnorm_c = torch.empty(d.d_model, dtype=torch.float32, device=y.device)
            for lo in range(0, d.tokens, chunk):
                hi = min(lo + chunk, d.tokens)
                y_c = y[lo:hi]
                yn = torch.empty_like(y_c)
                rstd = torch.empty(hi - lo, dtype=torch.float32, device=y.device)
                K.rmsnorm_fwd(kctx, y_c, wh["final_norm_w"], yn, rstd)
                logits = yn @ wh["w"].T                      # (c, V) — chunk scratch
                dlogits = torch.empty_like(logits)
                K.ce_loss_fwd_bwd(
                    kctx, logits, targets[lo:hi], part, dlogits, total_rows=d.tokens,
                )
                loss_acc += part
                dw_c = dlogits.T @ yn                        # (V, d)
                if accum or lo > 0:
                    dwh["w"].add_(dw_c.to(dwh["w"].dtype))
                else:
                    dwh["w"].copy_(dw_c.to(dwh["w"].dtype))
                dyn = dlogits @ wh["w"]                      # (c, d)
                del logits, dlogits
                K.rmsnorm_bwd(kctx, dyn, y_c, rstd, wh["final_norm_w"], dy[lo:hi], dnorm_c)
                dnorm_acc += dnorm_c
                # del before the next iteration's allocs: Python rebinding
                # would otherwise keep the previous chunk's buffers live
                # while the new ones allocate (2x peak)
                del yn, rstd, dyn, dw_c
            if accum:
                dwh["final_norm_w"].add_(dnorm_acc.to(dwh["final_norm_w"].dtype))
            else:
                dwh["final_norm_w"].copy_(dnorm_acc.to(dwh["final_norm_w"].dtype))
            loss.copy_(loss_acc)


@dataclass(frozen=True)
class EmbedBwd(_Base):
    """Embedding backward: deterministic per-row scatter of dy into
    dW_embed (sorted single-writer, never atomics)."""
    @property
    def egl(self) -> PackedLayout:
        return grad_layout(embed_weight_layout(self.dims), self.dims.dtypes, ns="embed")

    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            dy = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            tokens = torch_view(self._in(ctx, 1), (d.tokens,), torch.int32)
            accum = bool(ctx.task.mutates)
            buf = ctx.mutates[ctx.task.mutates[0]] if accum else self._out(ctx, 0)
            dwe = self.egl.view(buf, "w")
            self.kernels.embed_bwd_accum(kctx, tokens, dy, dwe,
                                         zero_first=not accum)


@dataclass(frozen=True)
class AdamWStep(_Base):  # name kept for resolver back-compat; see OptimizerStep alias
    """Per-FIELD optimizer step over one packed weight object —
    dispatches each field's rule (adamw/sgd/sgdm/muon/custom) and
    hyperparameters through the config's optimizer policy
    (tasks/optim.py); AdamW remains the all-fields default.

    Each field updates through its own w/g/m/v views at the dtype policy's
    storage dtypes (fp32 in registers regardless — the kernels are
    dtype-generic); padding gaps are never touched. ``kind`` selects the
    llama-shaped default layout ("block" | "embed" | "head");
    ``layout_for(dims, task, w_size_bytes) -> (PackedLayout, ns)`` overrides
    it for families whose optimizer key spans several layouts (qwen3's own
    block layout, qwen3.5's per-kind blocks / tied embed). The chosen
    layout's total_bytes must equal the W buffer size — a mismatch is a
    loud error, never a silent misview.
    """

    hyper: AdamWHyper = AdamWHyper()
    kind: str = "block"
    layout_for: object = None
    # per-field update overrides: {field_name: fn(kctx, kernels, w_view,
    # g_view)} — the field SKIPS AdamW math entirely (no m/v read, no
    # decay). First customer: DeepSeek-V3's non-gradient router bias,
    # whose dW slot carries expert counts and whose update is the
    # balance sign rule (tasks/moe/stages.moe_bias_update).
    update_specials: object = None

    def _layouts(self, task, w_size: int):
        d = self.dims
        layer = self.layer_of(task)
        if self.layout_for is not None:
            wl_, ns = self.layout_for(d, task, w_size)
        elif self.kind == "embed":
            wl_, ns = embed_weight_layout(d), "embed"
        elif self.kind == "head":
            wl_, ns = head_weight_layout(d), "head"
        else:
            wl_, ns = weight_layout(d, layer=layer), None
        if wl_.total_bytes != w_size:
            raise ValueError(
                f"optimizer layout mismatch for {task.id!r}: layout "
                f"{wl_.total_bytes} bytes vs W buffer {w_size} bytes"
            )
        p = d.dtypes
        op = getattr(d, "opt_policy", None)
        return (wl_, grad_layout(wl_, p, ns=ns, layer=layer),
                opt_state_layout(wl_, p, ns=ns, layer=layer, opt_policy=op),
                ns)

    def launch(self, ctx: TaskContext) -> None:
        from dataclasses import replace as dc_replace

        from .optim import OPTIMIZERS, hyper_for, resolve_opt_policy

        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            w_buf = ctx.mutates[ctx.task.mutates[0]]
            g_buf = ctx.inputs[ctx.task.inputs[1]]
            # a fully-stateless assignment (all-sgd layer) has NO O
            # object — the lowering scrubbed it (apply_exact_sizes)
            o_buf = (ctx.mutates[ctx.task.mutates[1]]
                     if len(ctx.task.mutates) > 1 else None)
            wl_, gl_, ol_, ns = self._layouts(ctx.task, w_buf.size_bytes)
            step = int(ctx.task.block_params.get("step", 0)) + 1
            op = resolve_opt_policy(getattr(self.dims, "opt_policy", None))
            layer = self.layer_of(ctx.task)
            key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
            # lr schedule: pure function of the step index; scales lr
            # AND muon_lr, applied AFTER per-field hyper overrides
            sched = self.hyper.schedule
            sched_scale = sched.scale(step) if sched is not None else 1.0
            for f in wl_.fields:
                if self.update_specials is not None and f.name in self.update_specials:
                    # highest-priority per-field override (noaux bias
                    # rule, frozen fields) — skips policy AND state
                    self.update_specials[f.name](
                        kctx, self.kernels,
                        wl_.view(w_buf, f.name).view(-1),
                        gl_.view(g_buf, f.name).view(-1),
                    )
                    continue
                opt = OPTIMIZERS[op.for_field(key(f.name), layer, f.shape)]
                hp = hyper_for(op, key(f.name), layer, self.hyper)
                if sched_scale != 1.0:
                    hp = dc_replace(
                        hp, lr=hp.lr * sched_scale,
                        muon_lr=(hp.muon_lr * sched_scale
                                 if hp.muon_lr else hp.muon_lr))
                if opt.slots and o_buf is None:
                    raise ValueError(
                        f"{ctx.task.id}: field {f.name!r} wants "
                        f"{opt.name!r} state but the task has no O object")
                states = {slot: ol_.view(o_buf, f"{slot}_{f.name}").view(-1)
                          for slot in opt.slots}
                opt.step(kctx, self.kernels, hp, step,
                         wl_.view(w_buf, f.name).view(-1),
                         gl_.view(g_buf, f.name).view(-1),
                         states, f.shape)



OptimizerStep = AdamWStep  # the general per-field-policy optimizer executable


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
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(dims, kernels, hyper, kind="block"),
        "optimizer_embed": AdamWStep(dims, kernels, hyper, kind="embed"),
        "optimizer_head": AdamWStep(dims, kernels, hyper, kind="head"),
    }

    def resolver(task: TaskSpec):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key {key!r} (task {task.id!r})")
        return table[key]

    resolver.kernel_set = kernels
    return resolver
