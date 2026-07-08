"""Qwen3-dense block executables.

Same buffer-order contract, staged-forward authoring, and kernel registry as
llama3 (`llama3_blocks.py` — embed/head/loss/optimizer executables are reused
verbatim); the transformer block differs by **qk-norm**: a per-head RMSNorm
(one shared ``(head_dim,)`` weight for q, one for k) between the q/k
projections and rope. No new kernels — the rmsnorm family runs at
``head_dim``-wide rows (``tokens * heads`` of them) through reshaped views.

Saved-context choice: instead of post-rope q/k we save the PRE-norm
projections (``qm``/``km``) + per-head rstds; backward re-applies norm+rope
(cheap elementwise) to rebuild flash-bwd's q/k and feeds ``rmsnorm_bwd``
exactly the tensors it needs for dW_qnorm/dW_knorm and the projection grads.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.core import TaskSpec

from . import ops
from .kernels import KernelSet, resolve_kernels
from .layouts import PackedLayout, Qwen3Dims, qwen3_context_layout, qwen3_weight_layout
from .base_blocks import AdamWHyper, AdamWStep, EmbedBwd, EmbedFwd, HeadLoss
from .llama3_blocks import BlockBwd, BlockFwd, BlockRecompute


@dataclass(frozen=True)
class Qwen3BlockFwd(BlockFwd):
    dims: Qwen3Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen3_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen3_context_layout(self.dims)

    # --- stages (see BlockFwd for the authoring contract) ---------------------

    @staticmethod
    def _stage_attn_norm(kctx, K, d, st):
        BlockFwd._stage_attn_norm(kctx, K, d, st)

    @staticmethod
    def _stage_qkv_qknorm(kctx, K, d, st):
        h1, w, a = st["h1"], st["w"], st["a"]
        t, h, kvh, hd = d.tokens, d.n_heads, d.n_kv_heads, d.head_dim
        # write-through: projections land directly in the ctx views when a
        # context is attached (bwd reads PRE-norm qm/km from ctx)
        if a is not None:
            qm, km, v = a["qm"], a["km"], a["v"]
            torch.matmul(h1, w["wq"], out=qm)
            torch.matmul(h1, w["wk"], out=km)
            torch.matmul(h1, w["wv"], out=v)
        else:
            qm = h1 @ w["wq"]
            km = h1 @ w["wk"]
            v = h1 @ w["wv"]
        qn = torch.empty_like(qm)
        rstd_q = torch.empty(t * h, dtype=torch.float32, device=qm.device)
        K.rmsnorm_fwd(kctx, qm.view(t * h, hd), w["q_norm_w"], qn.view(t * h, hd), rstd_q)
        kn = torch.empty_like(km)
        rstd_k = torch.empty(t * kvh, dtype=torch.float32, device=km.device)
        K.rmsnorm_fwd(kctx, km.view(t * kvh, hd), w["k_norm_w"], kn.view(t * kvh, hd), rstd_k)
        st.pop("h1")
        st.update(qn=qn, kn=kn, v=v)
        if a is not None:
            a["rstd_q"].copy_(rstd_q)
            a["rstd_k"].copy_(rstd_k)

    @staticmethod
    def _stage_rope(kctx, K, d, st):
        q = torch.empty_like(st["qn"])
        pos = ops.positions_for(d.seq_spec, q.shape[0], q.device)
        K.rope_fwd(kctx, st["qn"], q, pos, d.n_heads, d.head_dim, d.rope_base)
        k = torch.empty_like(st["kn"])
        K.rope_fwd(kctx, st["kn"], k, pos, d.n_kv_heads, d.head_dim, d.rope_base)
        st.pop("qn"), st.pop("kn")
        st.update(q=q, k=k)

    @staticmethod
    def _stage_attn(kctx, K, d, st):
        BlockFwd._stage_attn(kctx, K, d, st)

    @staticmethod
    def _stage_resid1_norm2(kctx, K, d, st):
        BlockFwd._stage_resid1_norm2(kctx, K, d, st)

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
        ("qkv_qknorm", _stage_qkv_qknorm.__func__, ("qm", "km", "rstd_q", "rstd_k", "v")),
        ("rope", _stage_rope.__func__, ()),
        ("attn", _stage_attn.__func__, ("lse", "attn_out")),
        ("resid1_norm2", _stage_resid1_norm2.__func__, ("h_mid", "rstd_ffn")),
        ("up_proj", _stage_up_proj.__func__, ("x1", "x3")),
        ("swiglu", _stage_swiglu.__func__, ()),
        ("down_resid", _stage_down_resid.__func__, ()),
    )


@dataclass(frozen=True)
class Qwen3BlockRecompute(Qwen3BlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Qwen3BlockBwd(BlockBwd):
    dims: Qwen3Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return qwen3_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return qwen3_context_layout(self.dims)

    def _attn_bwd(self, kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out) -> None:
        # attention part (qk-norm variant); the shared dense MLP tail runs
        # first via BlockBwd._backward's template (_mlp_bwd)
        d = self.dims
        K = self.kernels
        t, h, kvh, hd = d.tokens, d.n_heads, d.n_kv_heads, d.head_dim

        d_attn = dh_mid @ w["wo"].T
        acc("wo", a["attn_out"].T @ dh_mid)

        # rebuild flash-bwd's q/k from saved qm/km + rstds: norm re-apply + rope
        qm2, km2 = a["qm"].view(t * h, hd), a["km"].view(t * kvh, hd)
        qn = torch.empty_like(a["qm"])
        K.rmsnorm_apply(kctx, qm2, a["rstd_q"], w["q_norm_w"], qn.view(t * h, hd))
        kn = torch.empty_like(a["km"])
        K.rmsnorm_apply(kctx, km2, a["rstd_k"], w["k_norm_w"], kn.view(t * kvh, hd))
        q = torch.empty_like(qn)
        pos = ops.positions_for(d.seq_spec, qn.shape[0], qn.device)
        K.rope_fwd(kctx, qn, q, pos, h, hd, d.rope_base)
        del qn
        k = torch.empty_like(kn)
        K.rope_fwd(kctx, kn, k, pos, kvh, hd, d.rope_base)
        del kn

        dq, dk, dv = ops.flash_bwd(
            d_attn, q, k, a["v"], a["attn_out"], a["lse"], h, kvh, hd, d.seq_spec,
        )
        del d_attn, q, k
        dqn = torch.empty_like(dq)
        K.rope_bwd(kctx, dq, dqn, pos, h, hd, d.rope_base)
        del dq
        dkn = torch.empty_like(dk)
        K.rope_bwd(kctx, dk, dkn, pos, kvh, hd, d.rope_base)
        del dk

        dqm, dq_norm = norm_bwd(dqn.view(t * h, hd), qm2, a["rstd_q"], w["q_norm_w"])
        del dqn
        acc("q_norm_w", dq_norm)
        dkm, dk_norm = norm_bwd(dkn.view(t * kvh, hd), km2, a["rstd_k"], w["k_norm_w"])
        del dkn
        acc("k_norm_w", dk_norm)
        dqm = dqm.view(t, d.q_dim)
        dkm = dkm.view(t, d.kv_dim)

        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        acc("wq", h1.T @ dqm)
        acc("wk", h1.T @ dkm)
        acc("wv", h1.T @ dv)
        del h1
        dh1 = dqm @ w["wq"].T
        dh1.addmm_(dkm, w["wk"].T)
        dh1.addmm_(dv, w["wv"].T)
        del dqm, dkm, dv
        dx_n, dattn_norm = norm_bwd(dh1, x, a["rstd_attn"], w["attn_norm_w"])
        del dh1
        acc("attn_norm_w", dattn_norm)
        torch.add(dh_mid, dx_n, out=dx_out)


def build_qwen3_resolver(
    dims: Qwen3Dims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    """Executable resolver for the qwen3 family (same block keys as llama —
    the resolver is built per config, so families never collide)."""
    kernels = kernels if kernels is not None else resolve_kernels()
    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "block_fwd": Qwen3BlockFwd(dims, kernels),
        "block_recompute": Qwen3BlockRecompute(dims, kernels),
        "block_bwd": Qwen3BlockBwd(dims, kernels),
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(
            dims, kernels, hyper,
            layout_for=lambda d, task, size: (
                qwen3_weight_layout(d, layer=AdamWStep.layer_of(task)), None,
            ),
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
