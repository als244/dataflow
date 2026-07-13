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

from .. import ops
from ..interop import external_stream, torch_view
from ..kernels import KernelCtx, KernelSet, resolve_kernels
from ..layouts import (
    LlamaDims,
    PackedLayout,
    activation_layout,
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
    opt_state_layout,
    weight_layout,
)


from dataflow.tasks.base_blocks import (
    AdamWHyper,
    AdamWStep,
    GradReduceStep,
    EmbedBwd,
    EmbedFwd,
    HeadLoss,
    _Base,
)


def tp_group_handle(ctx, task):
    """The task's tensor-parallel GroupHandle, or None (standalone /
    warm-up run before the group exists — rank-local semantics, like
    a dp_group artifact run standalone)."""
    tp = task.block_params.get("tp")
    if tp is None:
        return None
    return (getattr(ctx, "groups", None) or {}).get(tp["group"])


def tp_allreduce_inplace(gh, es, tensor) -> None:
    """Producer-contract in-place allreduce on the group stream:
    es -> gh edge, sum over the tp group, gh -> es edge. gh None
    leaves the PARTIAL in place (standalone semantics)."""
    if gh is None:
        return
    produced = torch.cuda.Event()
    produced.record(es)
    gh.stream.wait_event(produced)
    gh.allreduce(tensor)
    summed = torch.cuda.Event()
    summed.record(gh.stream)
    es.wait_event(summed)


def tp_stage_down_resid(kctx, K, d, st):
    """Tensor-parallel down-projection: the shard's PARTIAL product
    allreduces over the tp group BEFORE the (replicated) residual
    joins — adding h_mid first would double it across ranks.

    Partials cross the wire in FP32. Rounding each shard's partial to
    bf16 before the sum injects relative error wherever the shards
    CANCEL — invisible at toy scale, but at 1B geometry it compounded
    into a diverging loss curve once the lr peaked (T4 attempt logs).
    fp32 partials keep ONE bf16 rounding total, matching the plain
    path's fused fp32-epilogue semantics."""
    partial = st.pop("s").float() @ st["w"]["w2"].float()
    tp_allreduce_inplace(st["tp_gh"], st["tp_es"], partial)
    partial.add_(st.pop("h_mid"))      # residual joins in fp32
    st["y"].copy_(partial)             # the ONE bf16 rounding


@dataclass(frozen=True)
class BlockFwd(_Base):
    """Transformer-block forward: runs the STAGES list, writing saved
    context (and metadata) fields through to the ctx buffers."""
    emit_context: bool = True

    # stage implementations swapped in under a tensor-parallel block
    # param (everything else is shard-transparent: up_proj gemms take
    # their width from the SLICED weight views, swiglu is elementwise)
    TP_STAGE_OVERRIDES = {"down_resid": tp_stage_down_resid}

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
                        a = self.cl_for(ctx.task).views(self._out(ctx, j))
                        break
            extras = self._aux_temp_state(ctx) or {}
            extras["seg"] = self._attn_meta(ctx)
            aux_counts = self._aux_counts_state(ctx)
            if aux_counts is not None:
                extras["aux_counts"] = aux_counts
            if ctx.task.block_params.get("tp") is not None:
                extras["tp_gh"] = tp_group_handle(ctx, ctx.task)
                extras["tp_es"] = es
            self._forward(kctx, x, w, y, a, extras=extras)

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
        pos = st["seg"].positions   # always varlen; run_args prologue
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
        # ALWAYS varlen: ONE launch for all segments (uniform batch is
        # equal-length segments); cu/max from the run_args prologue Segments
        seg = st["seg"]
        attn_out, lse = ops.flash_fwd(
            st["q"], st["k"], st["v"], d.n_heads, d.n_kv_heads,
            d.head_dim, cu_seqlens=seg.cu, max_seqlen=seg.max_len)
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
        if a is not None and "x1" in a:
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
        Stage entries may carry an optional 4th element "aux_temp" marking a
        METADATA-producing stage — skipped entirely when the M object is
        supplied (recompute repopulates ONLY the A objects)."""
        last = 0
        for i, entry in enumerate(cls.STAGES):
            if entry[2]:
                last = i
        return last + 1

    def recompute_stage_count_present(self) -> int:
        """Layout-aware boundary: the last stage emitting a PRESENT
        context field. Under trimmed layouts (dense warm-up saves only
        the objective's inputs) this shortens past post-attention
        stages whose emits were trimmed away; with full layouts it
        equals recompute_stage_count()."""
        present = {f.name for f in self.cl.fields}
        last = 0
        for i, entry in enumerate(self.STAGES):
            if entry[2] and (set(entry[2]) & present):
                last = i
        return last + 1

    @classmethod
    def context_fields_emitted(cls) -> set:
        return {f for entry in cls.STAGES for f in entry[2]}

    def stage_impls(self, st) -> tuple:
        """The stage list for this invocation: tensor-parallel runs
        (st carries tp_gh/tp_es) swap in the TP_STAGE_OVERRIDES
        entries by name; everything else is the plain STAGES list."""
        if "tp_es" not in st:
            return self.STAGES
        out = []
        for entry in self.STAGES:
            override = self.TP_STAGE_OVERRIDES.get(entry[0])
            if override is None:
                out.append(entry)
            else:
                out.append((entry[0], override) + tuple(entry[2:]))
        return tuple(out)

    def _run_stages(self, kctx, x, w, a, *, count: int, y=None,
                    extras=None) -> None:
        st = {"x": x, "w": w, "a": a, "y": y}
        if extras:
            st.update(extras)
        skip_aux_temp = bool(st.get("aux_temp_ready"))
        for entry in self.stage_impls(st)[:count]:
            if skip_aux_temp and len(entry) > 3 and entry[3] == "aux_temp":
                continue  # the aux pack is supplied, never recomputed
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
            a = self.cl_for(ctx.task).views(self._out(ctx, 0))
            extras = self._aux_temp_state(ctx) or {}
            extras["seg"] = self._attn_meta(ctx)
            # no tp extras: recompute stops at the last context-emitting
            # stage (up_proj), which is shard-transparent — the sliced
            # weight/context views carry the width; no collective runs
            self._forward_context(kctx, x, w, a, extras=extras)

    def _forward_context(self, kctx, x, w, a, extras=None) -> None:
        """DERIVED from the stage list: run through the last context-emitting
        stage and stop. The block output y is never a backward dependency,
        so the trailing stages (swiglu, down-projection, residual) are
        skipped by construction — not by hand-maintained duplication."""
        self._run_stages(kctx, x, w, a,
                         count=self.recompute_stage_count_present(),
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

    Scratch discipline: del each (t, .) temporary at last use so the
    caching allocator recycles it within the task; additive joins run in
    place (addmm_/add_, the forward's epilogue convention).
    """
    h2 = torch.empty_like(h_mid)
    K.rmsnorm_apply(kctx, h_mid, rstd_ffn, w["ffn_norm_w"], h2)
    s = torch.empty_like(x1)
    K.swiglu_fwd_out(kctx, x1, x3, s)
    ds = dy @ w["w2"].T
    if acc.wanted("w2"):
        acc("w2", s.T @ dy)
    del s
    dx1 = torch.empty_like(x1)
    dx3 = torch.empty_like(x3)
    K.swiglu_bwd(kctx, ds, x1, x3, dx1, dx3)
    del ds
    if acc.wanted("w1"):
        acc("w1", h2.T @ dx1)
    if acc.wanted("w3"):
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


def tp_mlp_tail_bwd(kctx, K, dy, h_mid, rstd_ffn, x1, x3, w, acc,
                    norm_bwd, gh, es):
    """Tensor-parallel twin of dense_mlp_tail_bwd: identical math on
    the SHARD-width views (x1/x3 and the w1/w3/w2 views arrive
    sliced), with ONE collective — the residual-stream gradient dh2
    is a partial sum over d_ff shards and allreduces before the
    (replicated, full-width) norm backward. dW writes are the local
    shard grads; the ffn_norm grad comes from the summed dh2, so it
    is a REPLICA on every rank (the optimizer's replica mode owns
    re-syncing replicated fields)."""
    h2 = torch.empty_like(h_mid)
    K.rmsnorm_apply(kctx, h_mid, rstd_ffn, w["ffn_norm_w"], h2)
    s = torch.empty_like(x1)
    K.swiglu_fwd_out(kctx, x1, x3, s)
    ds = dy @ w["w2"].T
    if acc.wanted("w2"):
        acc("w2", s.T @ dy)
    del s
    dx1 = torch.empty_like(x1)
    dx3 = torch.empty_like(x3)
    K.swiglu_bwd(kctx, ds, x1, x3, dx1, dx3)
    del ds
    if acc.wanted("w1"):
        acc("w1", h2.T @ dx1)
    if acc.wanted("w3"):
        acc("w3", h2.T @ dx3)
    del h2
    # fp32 partials across the wire: bf16-rounding each shard's
    # partial before the sum injects cancellation error (see
    # tp_stage_down_resid) — one bf16 rounding AFTER the sum
    dh2 = dx1.float() @ w["w1"].T.float()
    dh2.addmm_(dx3.float(), w["w3"].T.float())
    del dx1, dx3
    tp_allreduce_inplace(gh, es, dh2)   # partial over shards -> full
    dh2 = dh2.to(torch.bfloat16)
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
            a = self.cl_for(ctx.task).views(self._in(ctx, 1))
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
            a = {**a, "_seg": self._attn_meta(ctx)}
            aux_counts = self._aux_counts_state(ctx)
            if aux_counts is not None:
                a["aux_counts"] = aux_counts
            if ctx.task.block_params.get("tp") is not None:
                a["tp_gh"] = tp_group_handle(ctx, ctx.task)
                a["tp_es"] = es
            aux_temp = self._aux_temp_state(ctx)
            if aux_temp is None:
                self._backward(kctx, dy, a, x, w, dx, dw, accum)
            else:
                self._backward(kctx, dy, a, x, w, dx, dw, accum, aux_temp=aux_temp)

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum,
                  aux_temp=None) -> None:
        """Template: MLP tail then the family's attention part. Attention
        reads the run_args prologue metadata (cu/pos/max) that the LAUNCH
        merged into ``a`` under _pk_* keys (symmetric with the forward's
        ``st``). Direct-invocation callers (unit tests bypassing launch)
        get the uniform default here."""
        acc = self._acc_fn(dw, accum)
        norm_bwd = self._norm_bwd_fn(kctx)
        dh_mid = self._mlp_bwd(kctx, dy, a, w, dw, accum, acc, norm_bwd)
        self._attn_bwd(kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        if "tp_es" in a:
            return tp_mlp_tail_bwd(
                kctx, self.kernels, dy, a[self.MLP_RESID_FIELD],
                a["rstd_ffn"], a["x1"], a["x3"], w, acc, norm_bwd,
                gh=a["tp_gh"], es=a["tp_es"],
            )
        return dense_mlp_tail_bwd(
            kctx, self.kernels, dy, a[self.MLP_RESID_FIELD], a["rstd_ffn"],
            a["x1"], a["x3"], w, acc, norm_bwd,
        )

    def _attn_bwd(self, kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out) -> None:
        d = self.dims
        K = self.kernels
        seg = a["_seg"]
        d_attn = dh_mid @ w["wo"].T
        if acc.wanted("wo"):
            acc("wo", a["attn_out"].T @ dh_mid)
        dq, dk, dv = ops.flash_bwd(
            d_attn, a["q"], a["k"], a["v"], a["attn_out"], a["lse"],
            d.n_heads, d.n_kv_heads, d.head_dim,
            cu_seqlens=seg.cu, max_seqlen=seg.max_len)
        del d_attn
        dq_r = torch.empty_like(dq)
        pos = seg.positions          # always varlen; run_args prologue
        K.rope_bwd(kctx, dq, dq_r, pos, d.n_heads, d.head_dim, d.rope_base)
        del dq
        dk_r = torch.empty_like(dk)
        K.rope_bwd(kctx, dk, dk_r, pos, d.n_kv_heads, d.head_dim, d.rope_base)
        del dk
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        if acc.wanted("wq"):
            acc("wq", h1.T @ dq_r)
        if acc.wanted("wk"):
            acc("wk", h1.T @ dk_r)
        if acc.wanted("wv"):
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
        "grad_reduce_block": GradReduceStep(dims, kernels, kind="block"),
        "grad_reduce_embed": GradReduceStep(dims, kernels, kind="embed"),
        "grad_reduce_head": GradReduceStep(dims, kernels, kind="head"),
    }

    def resolver(task: TaskSpec):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key {key!r} (task {task.id!r})")
        return table[key]

    resolver.kernel_set = kernels
    return resolver
