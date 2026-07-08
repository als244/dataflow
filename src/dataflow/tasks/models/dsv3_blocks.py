"""DeepSeek-V3 block executables: MLA attention + dense/MoE hybrid depth.

Same buffer-order contract, staged authoring, and kernel registry as every
family (embed/head/loss/optimizer reused verbatim). Two kinds:

- ``mladense_*`` (first_k_dense layers): MLA attention + the shared dense
  SwiGLU tail (w1/w3/w2, ``dense_mlp_tail_bwd``).
- ``mlamoe_*``: MLA attention + the pluggable MoE tail in its DeepSeek
  flavor — sigmoid_noaux_tc routing and the UNGATED shared expert
  (``MOE_SHARED_NOGATE_STAGES``); the balance bias's dW slot carries
  expert counts (see tasks/moe/stages.py) and the optimizer applies the
  sign rule via ``update_specials``.

MLA conventions (pinned in tasks/mla_reference.py + tests): two low-rank
stacks with mid-stack RMSNorm; decoupled rope (per-head q_rope, ONE
shared k_rope per token); flash at shared head_dim = qk with ZERO-PADDED
v (exact); ctx saves the COMPRESSED pre-norm latents (q_a, kv_a) +
rstds + lse + attn_out at the true (t, h*v) — backward re-expands
through w_q_b / w_kv_b and reconstructs padded tensors from known-zeros.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.core import TaskSpec

from .. import ops
from ..kernels import KernelSet, resolve_kernels
from ..layouts import (
    Dsv3Dims,
    PackedLayout,
    dsv3_dense_context_layout,
    dsv3_dense_weight_layout,
    dsv3_moe_context_layout,
    dsv3_moe_weight_layout,
)
from ..base_blocks import AdamWHyper, AdamWStep, EmbedBwd, EmbedFwd, HeadLoss
from .llama3_blocks import BlockBwd, BlockFwd, BlockRecompute
from ..modules.moe.stages import (
    MOE_SHARED_NOGATE_STAGES,
    MoEMetaState,
    MoEProfileFill,
    moe_bias_update,
    moe_mlp_tail_bwd,
)


def _mla_expand_q(kctx, K, d, q_a_pre, rstd_qa, w, pos):
    """Saved/streamed pre-norm q_a -> per-head roped q_full (t, h*qk)."""
    t = q_a_pre.shape[0]
    h, nope, rope = d.n_heads, d.qk_nope_dim, d.qk_rope_dim
    qk = d.qk_head_dim
    q_lora_n = torch.empty_like(q_a_pre)
    K.rmsnorm_apply(kctx, q_a_pre, rstd_qa, w["q_a_norm_w"], q_lora_n)
    # in-place strided rope on the assembled projection: the GEMM output
    # already has the final [nope | rope] per-head layout — no extract,
    # no temp, no cat
    q_full = q_lora_n @ w["w_q_b"]
    K.rope_fwd(kctx, q_full, q_full, pos, h, rope, d.rope_base,
               row_stride=h * qk, head_stride=qk, col_base=nope)
    return q_lora_n, q_full


def _mla_expand_kv(kctx, K, d, kv_a_pre, rstd_kva, w, pos):
    """Saved/streamed pre-norm kv_a -> (latent_n, k_full (t,h*qk),
    v (t,h,v)). k_rope is roped from the LAST rope columns and broadcast
    across heads."""
    t = kv_a_pre.shape[0]
    h, nope, rope, v = d.n_heads, d.qk_nope_dim, d.qk_rope_dim, d.v_head_dim
    kvl, qk = d.kv_lora_rank, d.qk_head_dim
    latent_pre = kv_a_pre[:, :kvl].contiguous()
    latent_n = torch.empty_like(latent_pre)
    K.rmsnorm_apply(kctx, latent_pre, rstd_kva, w["kv_a_norm_w"], latent_n)
    k_rope = torch.empty(t, rope, dtype=kv_a_pre.dtype, device=kv_a_pre.device)
    K.rope_fwd(kctx, kv_a_pre[:, kvl:].contiguous(), k_rope, pos, 1, rope,
               d.rope_base)
    kvb = (latent_n @ w["w_kv_b"]).view(t, h, nope + v)
    k_full = torch.cat(
        [kvb[..., :nope], k_rope.view(t, 1, rope).expand(t, h, rope)], dim=-1,
    ).reshape(t, h * qk).contiguous()
    return latent_n, k_full, kvb[..., nope:]


def _pad_v(vals: torch.Tensor, qk: int) -> torch.Tensor:
    """(t, h, v) -> (t, h*qk) with zero padding (exactness pinned)."""
    t, h, v = vals.shape
    out = torch.zeros(t, h, qk, dtype=vals.dtype, device=vals.device)
    out[..., :v] = vals
    return out.reshape(t, h * qk)


@dataclass(frozen=True)
class Dsv3DenseBlockFwd(BlockFwd):
    dims: Dsv3Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv3_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv3_dense_context_layout(self.dims)

    # --- stages ---------------------------------------------------------------

    @staticmethod
    def _stage_attn_norm(kctx, K, d, st):
        BlockFwd._stage_attn_norm(kctx, K, d, st)

    @staticmethod
    def _stage_mla_q(kctx, K, d, st):
        h1, w, a = st["h1"], st["w"], st["a"]
        t = d.tokens
        if a is not None and "q_a" in a:
            q_a = a["q_a"]
            torch.matmul(h1, w["w_q_a"], out=q_a)
        else:
            q_a = h1 @ w["w_q_a"]
        rstd_qa = torch.empty(t, dtype=torch.float32, device=h1.device)
        # rmsnorm_fwd emits the normalized value AND rstd; we keep rstd in
        # ctx (bwd re-applies) and the normalized value locally
        q_lora_n = torch.empty_like(q_a)
        K.rmsnorm_fwd(kctx, q_a, w["q_a_norm_w"], q_lora_n, rstd_qa)
        pos = ops.positions_for(d.seq_spec, t, h1.device)
        h, nope, rope = d.n_heads, d.qk_nope_dim, d.qk_rope_dim
        q_full = q_lora_n @ w["w_q_b"]
        K.rope_fwd(kctx, q_full, q_full, pos, h, rope, d.rope_base,
                   row_stride=h * d.qk_head_dim, head_stride=d.qk_head_dim,
                   col_base=nope)
        st.update(q_full=q_full, pos=pos)
        if a is not None and "rstd_qa" in a:
            a["rstd_qa"].copy_(rstd_qa)
        else:
            st["rstd_qa"] = rstd_qa

    @staticmethod
    def _stage_mla_kv(kctx, K, d, st):
        h1, w, a = st.pop("h1"), st["w"], st["a"]
        t = d.tokens
        if a is not None and "kv_a" in a:
            kv_a = a["kv_a"]
            torch.matmul(h1, w["w_kv_a"], out=kv_a)
        else:
            kv_a = h1 @ w["w_kv_a"]
        kvl = d.kv_lora_rank
        latent_pre = kv_a[:, :kvl].contiguous()
        rstd_kva = torch.empty(t, dtype=torch.float32, device=h1.device)
        latent_n = torch.empty_like(latent_pre)
        K.rmsnorm_fwd(kctx, latent_pre, w["kv_a_norm_w"], latent_n, rstd_kva)
        h, nope, rope, v = d.n_heads, d.qk_nope_dim, d.qk_rope_dim, d.v_head_dim
        k_rope = torch.empty(t, rope, dtype=kv_a.dtype, device=kv_a.device)
        K.rope_fwd(kctx, kv_a[:, kvl:].contiguous(), k_rope, st["pos"], 1, rope,
                   d.rope_base)
        kvb = (latent_n @ w["w_kv_b"]).view(t, h, nope + v)
        k_full = torch.cat(
            [kvb[..., :nope], k_rope.view(t, 1, rope).expand(t, h, rope)], dim=-1,
        ).reshape(t, h * d.qk_head_dim).contiguous()
        st.update(k_full=k_full, v_pad=_pad_v(kvb[..., nope:], d.qk_head_dim))
        if a is not None and "rstd_kva" in a:
            a["rstd_kva"].copy_(rstd_kva)
        else:
            st["rstd_kva"] = rstd_kva

    @staticmethod
    def _stage_mla_attn(kctx, K, d, st):
        a = st["a"]
        t, h, qk, v = d.tokens, d.n_heads, d.qk_head_dim, d.v_head_dim
        out_pad, lse = ops.flash_fwd(
            st.pop("q_full"), st.pop("k_full"), st.pop("v_pad"),
            h, h, qk, d.seq_spec,
        )
        attn_out = out_pad.view(t, h, qk)[..., :v].reshape(t, h * v).contiguous()
        del out_pad
        if a is not None:
            a["lse"].copy_(lse)
            if "attn_out" in a:
                a["attn_out"].copy_(attn_out)
        st.update(attn_out=attn_out, lse=lse)

    @staticmethod
    def _stage_resid1_norm2(kctx, K, d, st):
        # h_mid = x + attn_out @ wo (ctx write-through); h2 = rmsnorm(h_mid)
        # — wo consumes the TRUE (t, h*v) attention output
        x, w, a = st["x"], st["w"], st["a"]
        attn_out = st.pop("attn_out")
        st.pop("lse", None)
        st.pop("pos", None)
        if a is not None and "h_mid" in a:
            h_mid = a["h_mid"]
            torch.addmm(x, attn_out, w["wo"], out=h_mid)
        else:
            h_mid = torch.addmm(x, attn_out, w["wo"])
        rstd_ffn = torch.empty(d.tokens, dtype=torch.float32, device=x.device)
        h2 = torch.empty_like(h_mid)
        K.rmsnorm_fwd(kctx, h_mid, w["ffn_norm_w"], h2, rstd_ffn)
        if a is not None:
            if "rstd_ffn" in a:
                a["rstd_ffn"].copy_(rstd_ffn)
        st.update(h_mid=h_mid, h2=h2)

    @staticmethod
    def _stage_up_proj(kctx, K, d, st):
        BlockFwd._stage_up_proj(kctx, K, d, st)

    @staticmethod
    def _stage_swiglu(kctx, K, d, st):
        BlockFwd._stage_swiglu(kctx, K, d, st)

    @staticmethod
    def _stage_down_resid(kctx, K, d, st):
        BlockFwd._stage_down_resid(kctx, K, d, st)

    MLA_STAGES = (
        ("attn_norm", _stage_attn_norm.__func__, ("rstd_attn",)),
        ("mla_q", _stage_mla_q.__func__, ("q_a", "rstd_qa")),
        ("mla_kv", _stage_mla_kv.__func__, ("kv_a", "rstd_kva")),
        ("mla_attn", _stage_mla_attn.__func__, ("lse", "attn_out")),
        ("resid1_norm2", _stage_resid1_norm2.__func__, ("h_mid", "rstd_ffn")),
    )

    STAGES = MLA_STAGES + (
        ("up_proj", _stage_up_proj.__func__, ("x1", "x3")),
        ("swiglu", _stage_swiglu.__func__, ()),
        ("down_resid", _stage_down_resid.__func__, ()),
    )


@dataclass(frozen=True)
class Dsv3DenseBlockRecompute(Dsv3DenseBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Dsv3DenseBlockBwd(BlockBwd):
    dims: Dsv3Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv3_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv3_dense_context_layout(self.dims)

    def _attn_bwd(self, kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out) -> None:
        # MLA backward: re-expand from the compressed latents, flash-bwd at
        # padded head_dim, chain through both low-rank stacks.
        d = self.dims
        K = self.kernels
        t = d.tokens
        h, nope, rope, v = d.n_heads, d.qk_nope_dim, d.qk_rope_dim, d.v_head_dim
        qk, kvl = d.qk_head_dim, d.kv_lora_rank

        d_attn_v = dh_mid @ w["wo"].T                       # (t, h*v)
        if acc.wanted("wo"):
            acc("wo", a["attn_out"].T @ dh_mid)

        pos = ops.positions_for(d.seq_spec, t, x.device)
        q_lora_n, q_full = _mla_expand_q(kctx, K, d, a["q_a"], a["rstd_qa"], w, pos)
        latent_n, k_full, vals = _mla_expand_kv(
            kctx, K, d, a["kv_a"], a["rstd_kva"], w, pos,
        )
        v_pad = _pad_v(vals, qk)
        del vals

        # padded attn_out / d_attn reconstructed from known-zeros
        attn_out_pad = _pad_v(a["attn_out"].view(t, h, v), qk)
        d_attn_pad = _pad_v(d_attn_v.view(t, h, v), qk)
        del d_attn_v
        dq, dk, dv_pad = ops.flash_bwd(
            d_attn_pad, q_full, k_full, v_pad, attn_out_pad, a["lse"],
            h, h, qk, d.seq_spec,
        )
        del d_attn_pad, q_full, k_full, v_pad, attn_out_pad

        # ---- kv stack ----
        dk3 = dk.view(t, h, qk)
        dv = dv_pad.view(t, h, qk)[..., :v]                 # pad cols provably zero
        dkvb = torch.cat([dk3[..., :nope], dv], dim=-1).reshape(
            t, h * (nope + v)).contiguous()
        del dv_pad, dv
        # shared decoupled k_rope: SUM the per-head grads, then un-rope
        dk_rope_sum = dk3[..., nope:].sum(dim=1).contiguous()   # (t, rope)
        del dk, dk3
        dk_rope_pre = torch.empty_like(dk_rope_sum)
        K.rope_bwd(kctx, dk_rope_sum, dk_rope_pre, pos, 1, rope, d.rope_base)
        del dk_rope_sum
        if acc.wanted("w_kv_b"):
            acc("w_kv_b", latent_n.T @ dkvb)
        dlatent_n = dkvb @ w["w_kv_b"].T
        del dkvb
        latent_pre = a["kv_a"][:, :kvl].contiguous()
        dlatent, d_kv_norm = norm_bwd(dlatent_n, latent_pre, a["rstd_kva"],
                                      w["kv_a_norm_w"])
        del dlatent_n, latent_pre, latent_n
        acc("kv_a_norm_w", d_kv_norm)
        d_kv_a = torch.cat([dlatent, dk_rope_pre], dim=-1)
        del dlatent, dk_rope_pre

        # ---- q stack ----
        K.rope_bwd(kctx, dq, dq, pos, h, rope, d.rope_base,
                   row_stride=h * qk, head_stride=qk, col_base=nope)
        dq_pre = dq
        del dq
        if acc.wanted("w_q_b"):
            acc("w_q_b", q_lora_n.T @ dq_pre)
        dq_lora_n = dq_pre @ w["w_q_b"].T
        del dq_pre, q_lora_n
        dq_lora, d_q_norm = norm_bwd(dq_lora_n, a["q_a"], a["rstd_qa"],
                                     w["q_a_norm_w"])
        del dq_lora_n
        acc("q_a_norm_w", d_q_norm)

        # ---- down-projections + attn norm ----
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        if acc.wanted("w_q_a"):
            acc("w_q_a", h1.T @ dq_lora)
        if acc.wanted("w_kv_a"):
            acc("w_kv_a", h1.T @ d_kv_a)
        del h1
        dh1 = dq_lora @ w["w_q_a"].T
        dh1.addmm_(d_kv_a, w["w_kv_a"].T)
        del dq_lora, d_kv_a
        dx_n, dattn_norm = norm_bwd(dh1, x, a["rstd_attn"], w["attn_norm_w"])
        del dh1
        acc("attn_norm_w", dattn_norm)
        torch.add(dh_mid, dx_n, out=dx_out)


@dataclass(frozen=True)
class Dsv3MoeBlockFwd(MoEMetaState, MoEProfileFill, Dsv3DenseBlockFwd):
    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv3_moe_context_layout(self.dims)

    STAGES = Dsv3DenseBlockFwd.MLA_STAGES + MOE_SHARED_NOGATE_STAGES


@dataclass(frozen=True)
class Dsv3MoeBlockRecompute(Dsv3MoeBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Dsv3MoeBlockBwd(MoEMetaState, MoEProfileFill, Dsv3DenseBlockBwd):
    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv3_moe_context_layout(self.dims)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        return moe_mlp_tail_bwd(
            kctx, self.kernels, self.dims, dy, a, w, dw, accum, acc, norm_bwd,
            resid_field=self.MLP_RESID_FIELD,
        )


def build_dsv3_resolver(
    dims: Dsv3Dims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    kernels = kernels if kernels is not None else resolve_kernels()
    from functools import partial

    bias_special = {
        "w_router_bias": partial(moe_bias_update,
                                 speed=dims.moe.bias_update_speed),
    }

    def _opt_layout(d, task, size):
        layer = AdamWStep.layer_of(task)
        if d.kind_of(layer) == "dense":
            return dsv3_dense_weight_layout(d, layer=layer), None
        return dsv3_moe_weight_layout(d, layer=layer), None

    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "mladense_fwd": Dsv3DenseBlockFwd(dims, kernels),
        "mladense_recompute": Dsv3DenseBlockRecompute(dims, kernels),
        "mladense_bwd": Dsv3DenseBlockBwd(dims, kernels),
        "mlamoe_fwd": Dsv3MoeBlockFwd(dims, kernels),
        "mlamoe_recompute": Dsv3MoeBlockRecompute(dims, kernels),
        "mlamoe_bwd": Dsv3MoeBlockBwd(dims, kernels),
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(
            dims, kernels, hyper, layout_for=_opt_layout,
            update_specials=bias_special,
        ),
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
