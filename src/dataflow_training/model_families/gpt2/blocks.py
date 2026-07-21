"""GPT-2 block executables: the llama3 staged templates with LayerNorm
statistics (mean AND rstd), a fused c_attn QKV projection, biases on every
linear, GELU-tanh, and a two-table embed object ([wte | wpe] — the learned
positions gather ``Segments.positions``, so packed varlen rounds are
position-correct by construction; LayerNorm itself is PER-TOKEN over
d_model and needs no sequence awareness).

Buffer-order contract, layouts, and staging discipline are the family
template's (llama3/blocks.py); bias adds ride the GEMM epilogues
(``torch.addmm`` broadcast) and bias grads are fp32 column sums.
GEMMs are direct cuBLAS calls here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.runtime.executable import TaskContext
from dataflow.runtime.interop import torch_view

from ...blocks import ops
from ...blocks.base_blocks import (AdamWHyper, AdamWStep, HeadLoss,
                                   RoundPrologue, _Base, _fill_tokens)
from ...blocks.layouts import (
    Gpt2Dims,
    PackedLayout,
    gpt2_activation_layout,
    gpt2_embed_layout,
    gpt2_head_layout,
    gpt2_weight_layout,
    grad_layout,
)
from ...kernels import KernelSet, resolve_kernels
from ..llama3.blocks import BlockBwd, BlockFwd, BlockRecompute


@dataclass(frozen=True)
class Gpt2BlockFwd(BlockFwd):
    dims: Gpt2Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return gpt2_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return gpt2_activation_layout(self.dims)

    # --- stages (see BlockFwd for the authoring contract) ---------------------

    @staticmethod
    def _stage_ln1(kctx, K, d, st):
        x, w = st["x"], st["w"]
        h1 = torch.empty_like(x)
        mean = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        K.layernorm_fwd(kctx, x, w["attn_norm_w"], w.get("attn_norm_b"),
                        h1, mean, rstd)
        st["h1"] = h1
        if st["a"] is not None:
            st["a"]["mean_attn"].copy_(mean)
            st["a"]["rstd_attn"].copy_(rstd)

    @staticmethod
    def _stage_qkv(kctx, K, d, st):
        # linear-triple conversion pending (exemplar: llama3) — bias
        # epilogues (addmm broadcast) need a bias-aware lane first
        h1, w, a = st.pop("h1"), st["w"], st["a"]
        if "b_qkv" in w:
            qkv = torch.addmm(w["b_qkv"], h1, w["w_qkv"])
        else:
            qkv = torch.matmul(h1, w["w_qkv"])
        dm = d.d_model
        if a is not None:
            # write-through: the contiguous per-table saves ARE the ctx views
            a["q"].copy_(qkv[:, :dm])
            a["k"].copy_(qkv[:, dm:2 * dm])
            a["v"].copy_(qkv[:, 2 * dm:])
            st.update(q=a["q"], k=a["k"], v=a["v"])
        else:
            st.update(q=qkv[:, :dm].contiguous(),
                      k=qkv[:, dm:2 * dm].contiguous(),
                      v=qkv[:, 2 * dm:].contiguous())

    @staticmethod
    def _stage_attn(kctx, K, d, st):
        seg = st["seg"]
        attn_out, lse = ops.flash_fwd(
            st["q"], st["k"], st["v"], d.n_heads, d.n_heads,
            d.head_dim, cu_seqlens=seg.cu, max_seqlen=seg.max_len)
        st.pop("q"), st.pop("k"), st.pop("v")
        st["attn_out"] = attn_out
        if st["a"] is not None:
            st["a"]["lse"].copy_(lse)
            st["a"]["attn_out"].copy_(attn_out)

    @staticmethod
    def _stage_resid1_ln2(kctx, K, d, st):
        a, w = st["a"], st["w"]
        out = a["h_mid"] if a is not None else None
        if "b_o" in w:
            h_mid = torch.addmm(w["b_o"], st["attn_out"], w["wo"], out=out)
        else:
            h_mid = torch.matmul(st["attn_out"], w["wo"], out=out)
        h_mid.add_(st["x"])
        h2 = torch.empty_like(h_mid)
        mean = torch.empty(h_mid.shape[0], dtype=torch.float32, device=h_mid.device)
        rstd = torch.empty(h_mid.shape[0], dtype=torch.float32, device=h_mid.device)
        K.layernorm_fwd(kctx, h_mid, w["ffn_norm_w"], w.get("ffn_norm_b"),
                        h2, mean, rstd)
        st.pop("attn_out")
        st.update(h_mid=h_mid, h2=h2)
        if a is not None:
            a["mean_ffn"].copy_(mean)
            a["rstd_ffn"].copy_(rstd)

    @staticmethod
    def _stage_fc_gelu(kctx, K, d, st):
        h2, w, a = st.pop("h2"), st["w"], st["a"]
        out = a["x_fc"] if a is not None else None
        if "b_fc" in w:
            x_fc = torch.addmm(w["b_fc"], h2, w["w_fc"], out=out)
        else:
            x_fc = torch.matmul(h2, w["w_fc"], out=out)
        g = torch.empty_like(x_fc)
        K.gelu_fwd_out(kctx, x_fc, g)
        st["g"] = g

    @staticmethod
    def _stage_proj_resid(kctx, K, d, st):
        w = st["w"]
        if "b_proj" in w:
            torch.addmm(w["b_proj"], st.pop("g"), w["w_proj"], out=st["y"])
        else:
            torch.matmul(st.pop("g"), w["w_proj"], out=st["y"])
        st["y"].add_(st.pop("h_mid"))

    STAGES = (
        ("ln1", _stage_ln1.__func__, ("mean_attn", "rstd_attn")),
        ("qkv", _stage_qkv.__func__, ("q", "k", "v")),
        ("attn", _stage_attn.__func__, ("lse", "attn_out")),
        ("resid1_ln2", _stage_resid1_ln2.__func__,
         ("h_mid", "mean_ffn", "rstd_ffn")),
        ("fc_gelu", _stage_fc_gelu.__func__, ("x_fc",)),
        ("proj_resid", _stage_proj_resid.__func__, ()),
    )


@dataclass(frozen=True)
class Gpt2BlockRecompute(BlockRecompute):
    dims: Gpt2Dims = None  # type: ignore[assignment]

    STAGES = Gpt2BlockFwd.STAGES

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return gpt2_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return gpt2_activation_layout(self.dims)


@dataclass(frozen=True)
class Gpt2BlockBwd(BlockBwd):
    """MLP tail (GELU) then attention backward; the two LayerNorms
    backward through the fused layernorm_bwd kernel (dx, dw, db in one
    pass). Normed inputs h1/h2 are recomputed from the saved statistics."""

    dims: Gpt2Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return gpt2_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return gpt2_activation_layout(self.dims)

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum,
                  aux_temp=None) -> None:
        # linear-triple conversion pending (exemplar: llama3)
        d, K = self.dims, self.kernels
        acc = self._acc_fn(dw, accum)
        seg = a["_seg"]
        # ---- MLP tail --------------------------------------------------------
        h2 = torch.empty_like(a["h_mid"])
        K.layernorm_apply(kctx, a["h_mid"], a["mean_ffn"], a["rstd_ffn"],
                          w["ffn_norm_w"], w.get("ffn_norm_b"), h2)
        g = torch.empty_like(a["x_fc"])
        K.gelu_fwd_out(kctx, a["x_fc"], g)
        if acc.wanted("w_proj"):
            acc("w_proj", torch.matmul(g.transpose(0, 1), dy))
        if acc.wanted("b_proj"):
            acc("b_proj", dy.float().sum(0))
        del g
        dg = torch.matmul(dy, w["w_proj"].transpose(0, 1))
        dx_fc = torch.empty_like(a["x_fc"])
        K.gelu_bwd(kctx, dg, a["x_fc"], dx_fc)
        del dg
        if acc.wanted("w_fc"):
            acc("w_fc", torch.matmul(h2.transpose(0, 1), dx_fc))
        if acc.wanted("b_fc"):
            acc("b_fc", dx_fc.float().sum(0))
        del h2
        dh2 = torch.matmul(dx_fc, w["w_fc"].transpose(0, 1))
        del dx_fc
        dh_mid = torch.empty_like(a["h_mid"])
        dw_ffn = torch.empty(d.d_model, dtype=torch.float32, device=x.device)
        db_ffn = torch.empty(d.d_model, dtype=torch.float32, device=x.device)
        K.layernorm_bwd(kctx, dh2, a["h_mid"], a["mean_ffn"], a["rstd_ffn"],
                        w["ffn_norm_w"], dh_mid, dw_ffn, db_ffn)
        del dh2
        acc("ffn_norm_w", dw_ffn)
        acc("ffn_norm_b", db_ffn)
        dh_mid.add_(dy)
        # ---- attention -------------------------------------------------------
        d_attn = torch.matmul(dh_mid, w["wo"].transpose(0, 1))
        if acc.wanted("wo"):
            acc("wo", torch.matmul(a["attn_out"].transpose(0, 1), dh_mid))
        if acc.wanted("b_o"):
            acc("b_o", dh_mid.float().sum(0))
        dq, dk, dv = ops.flash_bwd(
            d_attn, a["q"], a["k"], a["v"], a["attn_out"], a["lse"],
            d.n_heads, d.n_heads, d.head_dim,
            cu_seqlens=seg.cu, max_seqlen=seg.max_len)
        del d_attn
        dqkv = torch.cat((dq, dk, dv), dim=1)
        del dq, dk, dv
        h1 = torch.empty_like(x)
        K.layernorm_apply(kctx, x, a["mean_attn"], a["rstd_attn"],
                          w["attn_norm_w"], w.get("attn_norm_b"), h1)
        if acc.wanted("w_qkv"):
            acc("w_qkv", torch.matmul(h1.transpose(0, 1), dqkv))
        if acc.wanted("b_qkv"):
            acc("b_qkv", dqkv.float().sum(0))
        del h1
        dh1 = torch.matmul(dqkv, w["w_qkv"].transpose(0, 1))
        del dqkv
        dx_n = torch.empty_like(x)
        dw_attn = torch.empty(d.d_model, dtype=torch.float32, device=x.device)
        db_attn = torch.empty(d.d_model, dtype=torch.float32, device=x.device)
        K.layernorm_bwd(kctx, dh1, x, a["mean_attn"], a["rstd_attn"],
                        w["attn_norm_w"], dx_n, dw_attn, db_attn)
        del dh1
        acc("attn_norm_w", dw_attn)
        acc("attn_norm_b", db_attn)
        torch.add(dh_mid, dx_n, out=dx_out)


def parse_object_round(oid: str) -> str:
    """Round suffix of a per-round object id (``y_embed_{s}_{r}`` /
    ``dy_embed_{s}_{r}`` / ``tokens_{s}_{r}``)."""
    return oid.rsplit("_", 1)[-1]


@dataclass(frozen=True)
class Gpt2EmbedFwd(_Base):
    """Two-table embedding: wte gather by token id PLUS wpe gather by
    per-sequence position (``Segments.positions`` — restarts at 0 each
    segment). Loudly rejects rounds whose longest segment exceeds the
    position table (host-side check on Segments.lengths; no device sync)."""

    dims: Gpt2Dims = None  # type: ignore[assignment]

    @property
    def el(self) -> PackedLayout:
        return gpt2_embed_layout(self.dims)

    def profile_fill(self, ctx: TaskContext) -> None:
        _fill_tokens(self, ctx)

    def launch(self, ctx: TaskContext) -> None:
        from dataflow.runtime.interop import external_stream

        from ...data.segments import resolve_segments

        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
            n = self.num_tokens(ctx)
            tokens = torch_view(self._in(ctx, 0), (d.max_tokens,), torch.int32)[:n]
            w = self.el.views(self._in(ctx, 1))
            y = torch_view(self._out(ctx, 0), (d.max_tokens, d.d_model), torch.bfloat16)[:n]
            r = parse_object_round(ctx.task.outputs[0].id)
            seg = resolve_segments(ctx, d, r)
            if seg.lengths and max(seg.lengths) > d.n_ctx:
                raise ValueError(
                    f"segment length {max(seg.lengths)} exceeds n_ctx "
                    f"{d.n_ctx} (learned positions cannot extend past the "
                    f"table)")
            ops.embed_fwd(tokens, w["w"], y)
            y.add_(torch.index_select(w["wpe"], 0, seg.positions))


@dataclass(frozen=True)
class Gpt2EmbedBwd(_Base):
    """Deterministic scatter of dy into BOTH tables: token rows (wte) and
    position rows (wpe) through the same sorted single-writer kernel."""

    dims: Gpt2Dims = None  # type: ignore[assignment]

    @property
    def egl(self) -> PackedLayout:
        return grad_layout(gpt2_embed_layout(self.dims), self.dims.dtypes,
                           ns="head" if self.dims.tied else "embed")

    def profile_fill(self, ctx: TaskContext) -> None:
        _fill_tokens(self, ctx)

    def launch(self, ctx: TaskContext) -> None:
        from dataflow.runtime.interop import external_stream

        from ...data.segments import resolve_segments

        d = self.dims
        es = external_stream(ctx.stream)
        from ...kernels import KernelCtx

        kctx = KernelCtx(stream_handle=es.cuda_stream, torch_stream=es)
        with torch.cuda.stream(es):
            n = self.num_tokens(ctx)
            dy = torch_view(self._in(ctx, 0), (d.max_tokens, d.d_model), torch.bfloat16)[:n]
            tokens = torch_view(self._in(ctx, 1), (d.max_tokens,), torch.int32)[:n]
            accum = bool(ctx.task.mutates)
            buf = ctx.mutates[ctx.task.mutates[0]] if accum else self._out(ctx, 0)
            views = self.egl.views(buf)
            r = parse_object_round(ctx.task.inputs[0])
            seg = resolve_segments(ctx, d, r)
            self.kernels.embed_bwd_accum(kctx, tokens, dy, views["w"],
                                         zero_first=not accum)
            self.kernels.embed_bwd_accum(kctx, seg.positions.int(), dy,
                                         views["wpe"], zero_first=not accum)


@dataclass(frozen=True)
class Gpt2HeadLoss(HeadLoss):
    """The fused final-norm + head + CE task with the final norm as
    LayerNorm (gain AND bias). Tied configs read/create the shared
    W_embed/dW_embed — the create path also ZEROES the wpe region so
    embed_bwd's accumulate lands on defined bytes."""

    dims: Gpt2Dims = None  # type: ignore[assignment]

    @property
    def hl(self) -> PackedLayout:
        d = self.dims
        return gpt2_embed_layout(d) if d.tied else gpt2_head_layout(d)

    def launch(self, ctx: TaskContext) -> None:
        from ...blocks.base_blocks import head_chunk_rows

        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            n = self.num_tokens(ctx)
            K = self.kernels
            y = torch_view(self._in(ctx, 0), (d.max_tokens, d.d_model), torch.bfloat16)[:n]
            targets = torch_view(self._in(ctx, 1), (d.max_tokens,), torch.int32)[:n]
            wh = self.hl.views(self._in(ctx, 2))
            accum = bool(ctx.task.mutates)
            dy = torch_view(self._out(ctx, 0), (d.max_tokens, d.d_model), torch.bfloat16)[:n]
            loss = torch_view(self._out(ctx, 1), (1,), torch.float32)
            if accum:
                dwh = self.hgl.views(ctx.mutates[ctx.task.mutates[0]])
            elif len(ctx.task.outputs) > 2:
                dwh = self.hgl.views(self._out(ctx, 2))
            else:
                dwh = None
            ra_h = ctx.run_args or {}
            vr = ra_h.get("valid_rows")
            norm_rows = n
            if vr is not None:
                if isinstance(vr, dict):
                    _r = (ctx.task.outputs[1].id.rsplit("_", 1)[-1]
                          if len(ctx.task.outputs) > 1 else "0")
                    if _r in vr:
                        norm_rows = int(vr[_r])
                else:
                    norm_rows = int(vr)
            if dwh is not None and not accum and "wpe" in dwh:
                # tied create: this task owns dW_embed's first write; the
                # position-table region is embed_bwd's to accumulate into
                dwh["wpe"].zero_()
            chunk = head_chunk_rows(d.vocab_size)
            hw = self.head_linear()
            loss_acc = torch.zeros(1, dtype=torch.float32, device=y.device)
            part = torch.empty(1, dtype=torch.float32, device=y.device)
            dnw_acc = torch.zeros(d.d_model, dtype=torch.float32, device=y.device)
            dnb_acc = torch.zeros(d.d_model, dtype=torch.float32, device=y.device)
            dnw_c = torch.empty(d.d_model, dtype=torch.float32, device=y.device)
            dnb_c = torch.empty(d.d_model, dtype=torch.float32, device=y.device)
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                y_c = y[lo:hi]
                yn = torch.empty_like(y_c)
                mean = torch.empty(hi - lo, dtype=torch.float32, device=y.device)
                rstd = torch.empty(hi - lo, dtype=torch.float32, device=y.device)
                K.layernorm_fwd(kctx, y_c, wh["final_norm_w"],
                                wh.get("final_norm_b"), yn, mean, rstd)
                logits = hw.fwd(kctx, yn, wh)                # (c, V) — chunk scratch
                dlogits = torch.empty_like(logits)
                K.ce_loss_fwd_bwd(
                    kctx, logits, targets[lo:hi], part, dlogits,
                    total_rows=norm_rows,
                )
                loss_acc += part
                if dwh is not None and "w" in dwh:
                    dw_c = hw.wgrad(kctx, yn, dlogits)       # (V, d)
                    if accum or lo > 0:
                        dwh["w"].add_(dw_c.to(dwh["w"].dtype))
                    else:
                        dwh["w"].copy_(dw_c.to(dwh["w"].dtype))
                    del dw_c
                dyn = hw.dgrad(kctx, dlogits, wh)            # (c, d)
                del logits, dlogits
                K.layernorm_bwd(kctx, dyn, y_c, mean, rstd,
                                wh["final_norm_w"], dy[lo:hi], dnw_c, dnb_c)
                dnw_acc += dnw_c
                dnb_acc += dnb_c
                del yn, mean, rstd, dyn
            for name, acc_t in (("final_norm_w", dnw_acc),
                                ("final_norm_b", dnb_acc)):
                if dwh is not None and name in dwh:
                    if accum:
                        dwh[name].add_(acc_t.to(dwh[name].dtype))
                    else:
                        dwh[name].copy_(acc_t.to(dwh[name].dtype))
            loss.copy_(loss_acc)


def gpt2_block_opt_layout(dims, task, w_size):
    return gpt2_weight_layout(dims, layer=_Base.parse_layer(task)), None


def gpt2_embed_opt_layout(dims, task, w_size):
    return gpt2_embed_layout(dims), ("head" if dims.tied else "embed")


def gpt2_head_opt_layout(dims, task, w_size):
    return gpt2_head_layout(dims), "head"


def build_gpt2_resolver(
    dims: Gpt2Dims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    """Executable resolver keyed by compute_block_key (see llama3's
    build_resolver for the contract)."""
    kernels = kernels if kernels is not None else resolve_kernels()
    table = {
        "prologue_round": RoundPrologue(dims, kernels),
        "embed_fwd": Gpt2EmbedFwd(dims, kernels),
        "block_fwd": Gpt2BlockFwd(dims, kernels),
        "block_recompute": Gpt2BlockRecompute(dims, kernels),
        "block_bwd": Gpt2BlockBwd(dims, kernels),
        "head_loss": Gpt2HeadLoss(dims, kernels),
        "embed_bwd": Gpt2EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(dims, kernels, hyper, kind="block",
                                     resolve_layout=gpt2_block_opt_layout),
        "optimizer_embed": AdamWStep(dims, kernels, hyper, kind="embed",
                                     resolve_layout=gpt2_embed_opt_layout),
        "optimizer_head": AdamWStep(dims, kernels, hyper, kind="head",
                                    resolve_layout=gpt2_head_opt_layout),
    }

    def resolver(task):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key {key!r} (task {task.id!r})")
        return table[key]

    resolver.kernel_set = kernels
    return resolver
