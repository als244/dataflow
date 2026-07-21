"""Qwen3.5-dense block executables: hybrid Gated-DeltaNet + gated attention.

Same buffer-order contract and staged authoring as llama3/qwen3; two block
kinds resolved by distinct compute_block_keys (``linattn_*`` / ``gattn_*``)
while the shared embed/head/loss/optimizer executables are reused verbatim
(the head reads the TIED ``W_embed`` object — same packed [table |
final_norm_w] layout).

Every fla call follows the contracts pinned by tests/dataflow_training/models/test_qwen35.py:
- ``chunk_gated_delta_rule_fwd`` returns
  ``(g_post, o, A_int, final_state, initial_state, g_input)``; the blocks
  save g_post/A_int/core_out and DISCARD g_input (it equals the raw ``a``
  slice of the saved ``ba``, which backward reuses).
- ``chunk_gated_delta_rule_bwd(g=g_post, g_input=a_raw, A=A_int, ...)``
  returns ``(dq, dk, dv, dbeta, da, dh0, dA_log, ddt_bias)``.
- ``l2norm_bwd(y, rstd, dy)`` consumes the OUTPUT + rstd.
- ``causal_conv1d_bwd(x, dy, ..., activation="silu")`` recomputes the silu
  internally and returns ``(dx, dweight, dbias, dresidual, dinit)``.
- Every tensor handed to an fla/conv Triton kernel must be CONTIGUOUS.
  The kernels index with contiguous strides; a strided column slice such
  as ``ba[:, HV:]`` is read with the wrong row stride and SILENTLY
  corrupts the gate gradients (dA_log off by ~4x rel — the ladder-2
  divergence). ``.contiguous()`` at each boundary is the contract.

The gated-RMSNorm runs through the ``gated_rmsnorm_fwd/bwd`` registry ops
(fla's fused Triton kernels by default, eager reference fallback; the bwd
consumes the saved ``rstd_gate`` and recomputes y for the out-projection
grad). The attention output gate is closed-form sigmoid math. cu_seqlens
resets the conv window and the recurrence at packed-sequence boundaries
whenever batch > 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.core import TaskSpec

from ...blocks import ops
from ...kernels import KernelSet, resolve_kernels
from ...blocks.layouts import (
    PackedLayout,
    Qwen35Dims,
    embed_weight_layout,
    head_weight_layout,
    qwen35_attn_activation_layout,
    qwen35_attn_weight_layout,
    qwen35_lin_activation_layout,
    qwen35_lin_weight_layout,
)
from ...blocks.base_blocks import AdamWHyper, AdamWStep, EmbedBwd, EmbedFwd, HeadLoss
from ..llama3.blocks import BlockBwd, BlockFwd, BlockRecompute


def _cu_seqlens(seg) -> tuple:
    """(cu_seqlens int64, chunk_indices) for packed rounds — from the round's
    Segments; (None, None) for a single sequence (fla's non-varlen path). cu
    is a device->device cast of the already-materialized ``seg.cu`` (int32) —
    no host->device build / hidden sync."""
    if len(seg.lengths) == 1:
        return None, None
    from fla.modules.conv.triton.ops import prepare_chunk_indices

    cu = seg.cu.to(torch.int64)
    return cu, prepare_chunk_indices(cu, 64)


# ---------------------------------------------------------------------------
# Gated DeltaNet block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen35LinBlockFwd(BlockFwd):
    dims: Qwen35Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen35_lin_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen35_lin_activation_layout(self.dims)

    @staticmethod
    def _stage_attn_norm(kctx, K, d, st):
        BlockFwd._stage_attn_norm(kctx, K, d, st)

    @staticmethod
    def _stage_proj(kctx, K, d, st):
        # write-through: the two projections land directly in the ctx views
        # (768 MB scratch double + copy pass saved at bs32 otherwise)
        # linear-triple conversion pending (exemplar: llama3)
        h1, w, a = st["h1"], st["w"], st["a"]
        if a is not None:
            qkvz, ba = a["qkvz"], a["ba"]
            torch.matmul(h1, w["w_qkvz"], out=qkvz)
            torch.matmul(h1, w["w_ba"], out=ba)
        else:
            qkvz = h1 @ w["w_qkvz"]
            ba = h1 @ w["w_ba"]
        st.pop("h1")
        st.update(qkvz=qkvz, ba=ba)

    @staticmethod
    def _stage_conv(kctx, K, d, st):
        conv_in = st["qkvz"][:, : d.conv_dim].contiguous()
        cu, _ci = _cu_seqlens(st["seg"])
        post = torch.empty_like(conv_in)
        K.causal_conv1d_silu_fwd(kctx, conv_in, st["w"]["w_conv"], post, cu)
        st["post_conv"] = post

    @staticmethod
    def _stage_heads_l2norm(kctx, K, d, st):
        from fla.modules.l2norm import l2norm_fwd

        post = st["post_conv"]
        t = post.shape[0]
        q = post[:, : d.key_dim].reshape(t * d.lin_k_heads, d.lin_k_head_dim)
        k = post[:, d.key_dim : 2 * d.key_dim].reshape(t * d.lin_k_heads, d.lin_k_head_dim)
        qn, _ = l2norm_fwd(q.contiguous())
        kn, _ = l2norm_fwd(k.contiguous())
        st["qn"] = qn.view(t, d.lin_k_heads, d.lin_k_head_dim)
        st["kn"] = kn.view(t, d.lin_k_heads, d.lin_k_head_dim)
        st["v_h"] = post[:, 2 * d.key_dim :].reshape(t, d.lin_v_heads, d.lin_v_head_dim)

    @staticmethod
    def _stage_fla(kctx, K, d, st):
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

        w = st["w"]
        b = st["ba"][:, : d.lin_v_heads]
        a_raw = st["ba"][:, d.lin_v_heads :].contiguous()
        beta = torch.sigmoid(b.float()).to(b.dtype)
        cu, ci = _cu_seqlens(st["seg"])
        g_post, o, A_int, _fs, _is, _gi = chunk_gated_delta_rule_fwd(
            st["qn"].unsqueeze(0), st["kn"].unsqueeze(0),
            st["v_h"].unsqueeze(0).contiguous(),
            a_raw.unsqueeze(0), beta.unsqueeze(0),
            scale=d.lin_k_head_dim ** -0.5,
            initial_state=None, output_final_state=False,
            cu_seqlens=cu, chunk_indices=ci,
            use_gate_in_kernel=True,
            A_log=w["A_log"].float(), dt_bias=w["dt_bias"].float(),
        )
        st.pop("qn"), st.pop("kn"), st.pop("v_h"), st.pop("post_conv"), st.pop("ba")
        st["core_out"] = o.squeeze(0).to(torch.bfloat16)
        if st["a"] is not None:
            st["a"]["g_post"].copy_(g_post.squeeze(0))
            st["a"]["A_int"].copy_(A_int.squeeze(0).to(torch.bfloat16))
            st["a"]["core_out"].copy_(st["core_out"])

    @staticmethod
    def _stage_norm_out(kctx, K, d, st):
        # linear-triple conversion pending (exemplar: llama3)
        t = st["x"].shape[0]
        rows = t * d.lin_v_heads
        o2 = st["core_out"].reshape(rows, d.lin_v_head_dim)
        z2 = st["qkvz"][:, d.conv_dim :].contiguous().view(rows, d.lin_v_head_dim)
        y2 = torch.empty_like(o2)
        rstd = torch.empty(rows, dtype=torch.float32, device=o2.device)
        K.gated_rmsnorm_fwd(kctx, o2, z2, st["w"]["lin_norm_w"], y2, rstd)
        a = st["a"]
        if a is not None:
            xo = a["xo"]
            torch.addmm(st["x"], y2.view(t, d.value_dim), st["w"]["w_out"], out=xo)
        else:
            xo = torch.addmm(st["x"], y2.view(t, d.value_dim), st["w"]["w_out"])
        st.pop("core_out"), st.pop("qkvz")
        st["h_mid"] = xo  # feeds the shared MLP-tail stages
        if a is not None:
            a["rstd_gate"].copy_(rstd)

    @staticmethod
    def _stage_ffn_norm(kctx, K, d, st):
        h_mid = st["h_mid"]
        h2 = torch.empty_like(h_mid)
        rstd = torch.empty(h_mid.shape[0], dtype=torch.float32, device=h_mid.device)
        K.rmsnorm_fwd(kctx, h_mid, st["w"]["ffn_norm_w"], h2, rstd)
        st["h2"] = h2
        if st["a"] is not None:
            st["a"]["rstd_ffn"].copy_(rstd)

    @staticmethod
    def _stage_up_proj(kctx, K, d, st):
        BlockFwd._stage_up_proj(kctx, K, d, st)

    @staticmethod
    def _stage_swiglu(kctx, K, d, st):
        BlockFwd._stage_swiglu(kctx, K, d, st)

    @staticmethod
    def _stage_down_resid(kctx, K, d, st):
        BlockFwd._stage_down_resid(kctx, K, d, st)

    STAGES = (
        ("attn_norm", _stage_attn_norm.__func__, ("rstd_attn",)),
        ("proj", _stage_proj.__func__, ("qkvz", "ba")),
        ("conv", _stage_conv.__func__, ()),
        ("heads_l2norm", _stage_heads_l2norm.__func__, ()),
        ("fla", _stage_fla.__func__, ("g_post", "A_int", "core_out")),
        ("norm_out", _stage_norm_out.__func__, ("rstd_gate", "xo")),
        ("ffn_norm", _stage_ffn_norm.__func__, ("rstd_ffn",)),
        ("up_proj", _stage_up_proj.__func__, ("x1", "x3")),
        ("swiglu", _stage_swiglu.__func__, ()),
        ("down_resid", _stage_down_resid.__func__, ()),
    )


@dataclass(frozen=True)
class Qwen35LinBlockRecompute(Qwen35LinBlockFwd, BlockRecompute):
    # standard derived-recompute pattern (the historical hand-rolled
    # launch predated BlockRecompute and was the fleet's only custom one)
    pass


@dataclass(frozen=True)
class Qwen35LinBlockBwd(BlockBwd):
    dims: Qwen35Dims = None  # type: ignore[assignment]

    MLP_RESID_FIELD = "xo"  # the post-attention residual field (plays h_mid)

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen35_lin_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen35_lin_activation_layout(self.dims)

    def _attn_bwd(self, kctx, dxo, a, x, w, acc, norm_bwd, dx_out) -> None:
        # DeltaNet part; the shared dense MLP tail already ran via
        # BlockBwd._backward's template (_mlp_bwd, xo plays h_mid).
        # linear-triple conversion pending (exemplar: llama3)
        # Scratch discipline (bs32 measured 9.5 GiB of task-internal
        # torch scratch): every (t, .) temporary is del'd at its LAST use so
        # the caching allocator can recycle it within the task, and additive
        # joins run in place (addmm_/add_ — same epilogue convention the
        # forward already uses). Kernel-call order is unchanged.
        from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd

        d = self.dims
        K = self.kernels
        t = dxo.shape[0]

        # --- gated norm + out projection (fused kernel; y recomputed for
        # free by the bwd and reused for the out-projection weight grad) ---
        rows = t * d.lin_v_heads
        z2 = a["qkvz"][:, d.conv_dim :].contiguous().view(rows, d.lin_v_head_dim)
        o2 = a["core_out"].reshape(rows, d.lin_v_head_dim)
        do_normed = (dxo @ w["w_out"].T).contiguous().view(rows, d.lin_v_head_dim)
        dcore = torch.empty_like(o2)
        dz2 = torch.empty_like(z2)
        dwn = torch.empty(d.lin_v_head_dim, dtype=torch.float32, device=o2.device)
        y2 = torch.empty_like(o2)
        K.gated_rmsnorm_bwd(
            kctx, do_normed, o2, z2, w["lin_norm_w"], a["rstd_gate"],
            dcore, dz2, dwn, y2,
        )
        del do_normed, z2
        if acc.wanted("w_out"):
            acc("w_out", y2.view(t, d.value_dim).T @ dxo)
        del y2
        acc("lin_norm_w", dwn)
        dz = dz2.view(t, d.lin_v_heads, d.lin_v_head_dim)

        # --- recompute conv/l2norm inputs from saved qkvz ---
        conv_in = a["qkvz"][:, : d.conv_dim].contiguous()
        cu, ci = _cu_seqlens(a["_seg"])
        post = torch.empty_like(conv_in)
        K.causal_conv1d_silu_fwd(kctx, conv_in, w["w_conv"], post, cu)
        q2 = post[:, : d.key_dim].reshape(t * d.lin_k_heads, d.lin_k_head_dim).contiguous()
        qn, q_rstd = l2norm_fwd(q2)
        del q2
        k2 = post[:, d.key_dim : 2 * d.key_dim].reshape(t * d.lin_k_heads, d.lin_k_head_dim).contiguous()
        kn, k_rstd = l2norm_fwd(k2)
        del k2
        v_h = post[:, 2 * d.key_dim :].reshape(t, d.lin_v_heads, d.lin_v_head_dim)

        b = a["ba"][:, : d.lin_v_heads]
        a_raw = a["ba"][:, d.lin_v_heads :].contiguous()
        beta = torch.sigmoid(b.float()).to(b.dtype)

        # --- fla chunk bwd (pinned contract) ---
        dq, dk, dv, db, da, _dh0, dA_log, ddt = chunk_gated_delta_rule_bwd(
            q=qn.view(t, d.lin_k_heads, d.lin_k_head_dim).unsqueeze(0),
            k=kn.view(t, d.lin_k_heads, d.lin_k_head_dim).unsqueeze(0),
            v=v_h.unsqueeze(0).contiguous(),
            g=a["g_post"].unsqueeze(0),
            beta=beta.unsqueeze(0),
            A=a["A_int"].unsqueeze(0),
            scale=d.lin_k_head_dim ** -0.5,
            initial_state=None,
            do=dcore.reshape(t, d.lin_v_heads, d.lin_v_head_dim).unsqueeze(0).contiguous(),
            dht=None,
            cu_seqlens=cu, chunk_indices=ci,
            use_gate_in_kernel=True, g_input=a_raw.unsqueeze(0),
            A_log=w["A_log"].float(), dt_bias=w["dt_bias"].float(),
        )
        del post, v_h, dcore
        acc("A_log", dA_log)
        acc("dt_bias", ddt)

        # --- l2norm bwd (takes OUTPUT + rstd) ---
        dq_pre = l2norm_bwd(qn, q_rstd, dq.squeeze(0).reshape(t * d.lin_k_heads, d.lin_k_head_dim))
        del qn, dq
        dk_pre = l2norm_bwd(kn, k_rstd, dk.squeeze(0).reshape(t * d.lin_k_heads, d.lin_k_head_dim))
        del kn, dk

        # --- conv bwd (silu recomputed internally by the kernel) ---
        d_post = torch.cat([
            dq_pre.view(t, d.key_dim),
            dk_pre.view(t, d.key_dim),
            dv.squeeze(0).reshape(t, d.value_dim),
        ], dim=-1)
        del dq_pre, dk_pre, dv
        dconv_in = torch.empty_like(conv_in)
        dwconv = torch.empty_like(w["w_conv"])
        K.causal_conv1d_silu_bwd(kctx, conv_in, d_post, w["w_conv"], dconv_in, dwconv, cu)
        del conv_in, d_post
        acc("w_conv", dwconv)

        # --- assemble projection grads ---
        db_raw = (db.squeeze(0).float() * (beta.float() * (1 - beta.float()))).to(torch.bfloat16)
        d_qkvz = torch.cat([dconv_in, dz.reshape(t, d.value_dim)], dim=-1)
        del dconv_in, dz, dz2
        d_ba = torch.cat([db_raw, da.squeeze(0).to(torch.bfloat16)], dim=-1)
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        if acc.wanted("w_qkvz"):
            acc("w_qkvz", h1.T @ d_qkvz)
        if acc.wanted("w_ba"):
            acc("w_ba", h1.T @ d_ba)
        del h1
        dh1 = d_qkvz @ w["w_qkvz"].T
        dh1.addmm_(d_ba, w["w_ba"].T)
        del d_qkvz, d_ba
        dx_n, dattn = norm_bwd(dh1, x, a["rstd_attn"], w["attn_norm_w"])
        del dh1
        acc("attn_norm_w", dattn)
        torch.add(dxo, dx_n, out=dx_out)


# ---------------------------------------------------------------------------
# Gated full-attention block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen35AttnBlockFwd(BlockFwd):
    dims: Qwen35Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen35_attn_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen35_attn_activation_layout(self.dims)

    @staticmethod
    def _partial_rope(kctx, K, d, x, heads, pos, *, bwd: bool = False):
        """Rotate only the first rot_dim channels per head; pass the rest.
        ``pos`` = the run_args prologue positions (always varlen)."""
        t = x.shape[0]
        rot = d.rot_dim
        xh = x.view(t, heads, d.head_dim)
        rs = xh[:, :, :rot].contiguous().view(t, heads * rot)
        out = torch.empty_like(rs)
        fn = K.rope_bwd if bwd else K.rope_fwd
        fn(kctx, rs, out, pos, heads, rot, d.rope_base)
        y = torch.empty_like(xh)
        y[:, :, :rot] = out.view(t, heads, rot)
        y[:, :, rot:] = xh[:, :, rot:]
        return y.view(t, heads * d.head_dim)

    @staticmethod
    def _stage_attn_norm(kctx, K, d, st):
        BlockFwd._stage_attn_norm(kctx, K, d, st)

    @staticmethod
    def _stage_qkv_gate(kctx, K, d, st):
        # linear-triple conversion pending (exemplar: llama3)
        h1, w, a = st["h1"], st["w"], st["a"]
        qg = h1 @ w["wq"]  # ONE doubled GEMM: [Q_all | gate_all] — its two
        qm, gate = qg[:, : d.attn_dim], qg[:, d.attn_dim :]  # ctx fields are
        # slices of one output, so those copies stay (contiguity contract)
        if a is not None:
            km, v = a["km"], a["v"]
            torch.matmul(h1, w["wk"], out=km)
            torch.matmul(h1, w["wv"], out=v)
            a["qm"].copy_(qm)
            a["gate"].copy_(gate)
            st.pop("h1")
            st.update(qm=a["qm"], gate=a["gate"], km=km, v=v)
        else:
            km = h1 @ w["wk"]
            v = h1 @ w["wv"]
            st.pop("h1")
            st.update(qm=qm.contiguous(), gate=gate.contiguous(), km=km, v=v)

    @staticmethod
    def _stage_qknorm_rope(kctx, K, d, st):
        t = st["x"].shape[0]
        h, kvh, hd = d.n_heads, d.n_kv_heads, d.head_dim
        qn = torch.empty_like(st["qm"])
        rstd_q = torch.empty(t * h, dtype=torch.float32, device=qn.device)
        K.rmsnorm_fwd(kctx, st["qm"].view(t * h, hd), st["w"]["q_norm_w"], qn.view(t * h, hd), rstd_q)
        kn = torch.empty_like(st["km"])
        rstd_k = torch.empty(t * kvh, dtype=torch.float32, device=kn.device)
        K.rmsnorm_fwd(kctx, st["km"].view(t * kvh, hd), st["w"]["k_norm_w"], kn.view(t * kvh, hd), rstd_k)
        st["q"] = Qwen35AttnBlockFwd._partial_rope(kctx, K, d, qn, h, st["seg"].positions)
        st["k"] = Qwen35AttnBlockFwd._partial_rope(kctx, K, d, kn, kvh, st["seg"].positions)
        st.pop("qm"), st.pop("km")
        if st["a"] is not None:
            st["a"]["rstd_q"].copy_(rstd_q)
            st["a"]["rstd_k"].copy_(rstd_k)

    @staticmethod
    def _stage_attn(kctx, K, d, st):
        attn_out, lse = ops.flash_fwd(
            st["q"], st["k"], st["v"], d.n_heads, d.n_kv_heads, d.head_dim,
            cu_seqlens=st["seg"].cu, max_seqlen=st["seg"].max_len,
        )
        st.pop("q"), st.pop("k"), st.pop("v")
        st["attn_out"] = attn_out
        if st["a"] is not None:
            st["a"]["lse"].copy_(lse)
            st["a"]["attn_out"].copy_(attn_out)

    @staticmethod
    def _stage_gate_o(kctx, K, d, st):
        # linear-triple conversion pending (exemplar: llama3)
        t = st["x"].shape[0]
        gated = st.pop("attn_out") * torch.sigmoid(st.pop("gate").float()).to(torch.bfloat16)
        a = st["a"]
        if a is not None:
            xo = a["xo"]
            torch.addmm(st["x"], gated, st["w"]["wo"], out=xo)
        else:
            xo = torch.addmm(st["x"], gated, st["w"]["wo"])
        st["h_mid"] = xo

    _stage_ffn_norm = staticmethod(Qwen35LinBlockFwd._stage_ffn_norm)

    @staticmethod
    def _stage_up_proj(kctx, K, d, st):
        BlockFwd._stage_up_proj(kctx, K, d, st)

    @staticmethod
    def _stage_swiglu(kctx, K, d, st):
        BlockFwd._stage_swiglu(kctx, K, d, st)

    @staticmethod
    def _stage_down_resid(kctx, K, d, st):
        BlockFwd._stage_down_resid(kctx, K, d, st)

    STAGES = (
        ("attn_norm", _stage_attn_norm.__func__, ("rstd_attn",)),
        ("qkv_gate", _stage_qkv_gate.__func__, ("qm", "km", "gate", "v")),
        ("qknorm_rope", _stage_qknorm_rope.__func__, ("rstd_q", "rstd_k")),
        ("attn", _stage_attn.__func__, ("lse", "attn_out")),
        ("gate_o", _stage_gate_o.__func__, ("xo",)),
        ("ffn_norm", _stage_ffn_norm.__func__, ("rstd_ffn",)),
        ("up_proj", _stage_up_proj.__func__, ("x1", "x3")),
        ("swiglu", _stage_swiglu.__func__, ()),
        ("down_resid", _stage_down_resid.__func__, ()),
    )


@dataclass(frozen=True)
class Qwen35AttnBlockRecompute(Qwen35AttnBlockFwd, Qwen35LinBlockRecompute):
    pass


@dataclass(frozen=True)
class Qwen35AttnBlockBwd(BlockBwd):
    dims: Qwen35Dims = None  # type: ignore[assignment]

    MLP_RESID_FIELD = "xo"  # the post-attention residual field (plays h_mid)

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen35_attn_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen35_attn_activation_layout(self.dims)

    def _attn_bwd(self, kctx, dxo, a, x, w, acc, norm_bwd, dx_out) -> None:
        # gated-attention part; the shared dense MLP tail already ran via
        # BlockBwd._backward's template (_mlp_bwd, xo plays h_mid).
        # Scratch discipline: del each (t, .) temporary at last use;
        # additive joins in place (see the lin-block backward note).
        d = self.dims
        K = self.kernels
        t = dxo.shape[0]
        h, kvh, hd = d.n_heads, d.n_kv_heads, d.head_dim

        # --- output gate + o projection ---
        # linear-triple conversion pending (exemplar: llama3)
        sig = torch.sigmoid(a["gate"].float())
        gated = a["attn_out"] * sig.to(torch.bfloat16)
        if acc.wanted("wo"):
            acc("wo", gated.T @ dxo)
        del gated
        dgated = dxo @ w["wo"].T
        d_attn = (dgated.float() * sig).to(torch.bfloat16)
        d_gate = (dgated.float() * a["attn_out"].float() * sig * (1 - sig)).to(torch.bfloat16)
        del dgated, sig

        # --- flash bwd needs post-norm/post-rope q, k (rebuild from saved) ---
        qn = torch.empty_like(a["qm"])
        K.rmsnorm_apply(kctx, a["qm"].view(t * h, hd), a["rstd_q"], w["q_norm_w"], qn.view(t * h, hd))
        kn = torch.empty_like(a["km"])
        K.rmsnorm_apply(kctx, a["km"].view(t * kvh, hd), a["rstd_k"], w["k_norm_w"], kn.view(t * kvh, hd))
        seg = a["_seg"]
        pos = seg.positions          # always varlen; run_args prologue
        q = Qwen35AttnBlockFwd._partial_rope(kctx, K, d, qn, h, pos)
        del qn
        k = Qwen35AttnBlockFwd._partial_rope(kctx, K, d, kn, kvh, pos)
        del kn
        dq, dk, dv = ops.flash_bwd(
            d_attn, q, k, a["v"], a["attn_out"], a["lse"], h, kvh, hd,
            cu_seqlens=seg.cu, max_seqlen=seg.max_len,
        )
        del d_attn, q, k
        dqn = Qwen35AttnBlockFwd._partial_rope(kctx, K, d, dq, h, pos, bwd=True)
        del dq
        dkn = Qwen35AttnBlockFwd._partial_rope(kctx, K, d, dk, kvh, pos, bwd=True)
        del dk

        dqm, dqnorm = norm_bwd(dqn.view(t * h, hd), a["qm"].view(t * h, hd), a["rstd_q"], w["q_norm_w"])
        del dqn
        acc("q_norm_w", dqnorm)
        dkm, dknorm = norm_bwd(dkn.view(t * kvh, hd), a["km"].view(t * kvh, hd), a["rstd_k"], w["k_norm_w"])
        del dkn
        acc("k_norm_w", dknorm)
        dqm = dqm.view(t, d.attn_dim)
        dkm = dkm.view(t, d.kv_dim)

        # --- projections (wq is the doubled [Q_all | gate_all]) ---
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        d_qg = torch.cat([dqm, d_gate], dim=-1)
        del dqm, d_gate
        if acc.wanted("wq"):
            acc("wq", h1.T @ d_qg)
        if acc.wanted("wk"):
            acc("wk", h1.T @ dkm)
        if acc.wanted("wv"):
            acc("wv", h1.T @ dv)
        del h1
        dh1 = d_qg @ w["wq"].T
        dh1.addmm_(dkm, w["wk"].T)
        dh1.addmm_(dv, w["wv"].T)
        del d_qg, dkm, dv
        dx_n, dattn_norm = norm_bwd(dh1, x, a["rstd_attn"], w["attn_norm_w"])
        del dh1
        acc("attn_norm_w", dattn_norm)
        torch.add(dxo, dx_n, out=dx_out)


def _opt_block_layout(d, task, w_size):
    """optimizer_block spans BOTH layer kinds: derive the layer from the
    mutated W_{i}, pick the kind's layout at that layer's dtype sub-policy
    (the AdamWStep size assert stays as the tripwire)."""
    layer = AdamWStep.parse_layer(task)
    build = (
        qwen35_attn_weight_layout if d.kinds[layer] == "full"
        else qwen35_lin_weight_layout
    )
    return build(d, layer=layer), None


def _opt_embed_layout(d, task, w_size):
    """Untied: bare table (policy names embed.*). Tied: W_embed IS the head
    layout [table | final_norm_w] and stays policy-addressed as head.*."""
    el = embed_weight_layout(d)
    if el.total_bytes == w_size:
        return el, "embed"
    hl = head_weight_layout(d)
    if hl.total_bytes == w_size:
        return hl, "head"
    raise ValueError(
        f"no embed/tied weight layout matches W of {w_size} bytes ({task.id!r})"
    )


def build_qwen35_resolver(
    dims: Qwen35Dims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    kernels = kernels if kernels is not None else resolve_kernels()
    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "linattn_fwd": Qwen35LinBlockFwd(dims, kernels),
        "linattn_recompute": Qwen35LinBlockRecompute(dims, kernels),
        "linattn_bwd": Qwen35LinBlockBwd(dims, kernels),
        "gattn_fwd": Qwen35AttnBlockFwd(dims, kernels),
        "gattn_recompute": Qwen35AttnBlockRecompute(dims, kernels),
        "gattn_bwd": Qwen35AttnBlockBwd(dims, kernels),
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(dims, kernels, hyper, resolve_layout=_opt_block_layout),
        "optimizer_embed": AdamWStep(dims, kernels, hyper, resolve_layout=_opt_embed_layout),
        "optimizer_head": AdamWStep(dims, kernels, hyper, kind="head"),  # untied only
    }

    def resolver(task: TaskSpec):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key {key!r} (task {task.id!r})")
        return table[key]

    resolver.kernel_set = kernels
    return resolver
