"""OLMoE block executables: qwen3-shaped attention + the pluggable MoE tail.

Same buffer-order contract, staged authoring, and kernel registry as
llama3/qwen3 (embed/head/loss/optimizer executables reused verbatim). Two
differences from qwen3's block:

- **FULL-ROW qk-norm** (`qk_norm_per_head=False`): one RMSNorm over the
  whole (t, q_dim)/(t, kv_dim) rows — weights `(q_dim,)`/`(kv_dim,)`, ONE
  rstd per token. Same rmsnorm registry kernels at different row widths;
  backward re-applies norm+rope from the saved pre-norm projections
  exactly like qwen3, minus the per-head reshapes.
- **MoE FFN**: the dense SwiGLU tail is replaced by the spliced
  ``MOE_STAGES`` (route -> dispatch -> experts -> combine) and
  ``moe_mlp_tail_bwd`` — see tasks/moe/ for the module contract. The
  ``MoEProfileFill`` mixin seeds valid balanced routing for the profiler.

Rope is full-rotary at theta 1e4 (rebuilt in backward, not saved).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.core import TaskSpec

from .. import ops
from ..kernels import KernelSet, resolve_kernels
from ..layouts import (
    OlmoeDims,
    PackedLayout,
    olmoe_activation_layout,
    olmoe_weight_layout,
)
from ..base_blocks import AdamWHyper, AdamWStep, EmbedBwd, EmbedFwd, HeadLoss, RoundPrologue
from .llama3_blocks import BlockBwd, BlockFwd, BlockRecompute
from ..modules.moe.stages import MOE_STAGES, MoEAuxTempState, MoEProfileFill, moe_mlp_tail_bwd


@dataclass(frozen=True)
class OlmoeBlockFwd(MoEAuxTempState, MoEProfileFill, BlockFwd):
    dims: OlmoeDims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return olmoe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return olmoe_activation_layout(self.dims)

    # --- stages (see BlockFwd for the authoring contract) ---------------------

    @staticmethod
    def _stage_attn_norm(kctx, K, d, st):
        BlockFwd._stage_attn_norm(kctx, K, d, st)

    @staticmethod
    def _stage_qkv_qknorm(kctx, K, d, st):
        # FULL-ROW qk-norm: rmsnorm over (t, q_dim)/(t, kv_dim) rows —
        # no per-head reshape, one rstd per token. Projections write
        # through to ctx (bwd reads PRE-norm qm/km).
        h1, w, a = st["h1"], st["w"], st["a"]
        t = d.tokens
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
        rstd_q = torch.empty(t, dtype=torch.float32, device=qm.device)
        K.rmsnorm_fwd(kctx, qm, w["q_norm_w"], qn, rstd_q)
        kn = torch.empty_like(km)
        rstd_k = torch.empty(t, dtype=torch.float32, device=km.device)
        K.rmsnorm_fwd(kctx, km, w["k_norm_w"], kn, rstd_k)
        st.pop("h1")
        st.update(qn=qn, kn=kn, v=v)
        if a is not None:
            a["rstd_q"].copy_(rstd_q)
            a["rstd_k"].copy_(rstd_k)

    @staticmethod
    def _stage_rope(kctx, K, d, st):
        q = torch.empty_like(st["qn"])
        pos = st["seg"].positions   # always varlen; run_args prologue
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

    STAGES = (
        ("attn_norm", _stage_attn_norm.__func__, ("rstd_attn",)),
        ("qkv_qknorm", _stage_qkv_qknorm.__func__, ("qm", "km", "rstd_q", "rstd_k", "v")),
        ("rope", _stage_rope.__func__, ()),
        ("attn", _stage_attn.__func__, ("lse", "attn_out")),
        ("resid1_norm2", _stage_resid1_norm2.__func__, ("h_mid", "rstd_ffn")),
    ) + MOE_STAGES


@dataclass(frozen=True)
class OlmoeBlockRecompute(OlmoeBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class OlmoeBlockBwd(MoEAuxTempState, MoEProfileFill, BlockBwd):
    dims: OlmoeDims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return olmoe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return olmoe_activation_layout(self.dims)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        return moe_mlp_tail_bwd(
            kctx, self.kernels, self.dims, dy, a, w, dw, accum, acc, norm_bwd,
            resid_field=self.MLP_RESID_FIELD,
        )

    def _attn_bwd(self, kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out) -> None:
        # qwen3-shaped attention backward with FULL-ROW norm re-apply:
        # rebuild flash-bwd's q/k from saved pre-norm qm/km + rstds.
        d = self.dims
        K = self.kernels

        d_attn = dh_mid @ w["wo"].T
        if acc.wanted("wo"):
            acc("wo", a["attn_out"].T @ dh_mid)

        qn = torch.empty_like(a["qm"])
        K.rmsnorm_apply(kctx, a["qm"], a["rstd_q"], w["q_norm_w"], qn)
        kn = torch.empty_like(a["km"])
        K.rmsnorm_apply(kctx, a["km"], a["rstd_k"], w["k_norm_w"], kn)
        q = torch.empty_like(qn)
        seg = a["_seg"]
        pos = seg.positions          # always varlen; run_args prologue
        K.rope_fwd(kctx, qn, q, pos, d.n_heads, d.head_dim, d.rope_base)
        del qn
        k = torch.empty_like(kn)
        K.rope_fwd(kctx, kn, k, pos, d.n_kv_heads, d.head_dim, d.rope_base)
        del kn

        dq, dk, dv = ops.flash_bwd(
            d_attn, q, k, a["v"], a["attn_out"], a["lse"],
            d.n_heads, d.n_kv_heads, d.head_dim,
            cu_seqlens=seg.cu, max_seqlen=seg.max_len,
        )
        del d_attn, q, k
        dqn = torch.empty_like(dq)
        K.rope_bwd(kctx, dq, dqn, pos, d.n_heads, d.head_dim, d.rope_base)
        del dq
        dkn = torch.empty_like(dk)
        K.rope_bwd(kctx, dk, dkn, pos, d.n_kv_heads, d.head_dim, d.rope_base)
        del dk

        dqm, dq_norm = norm_bwd(dqn, a["qm"], a["rstd_q"], w["q_norm_w"])
        del dqn
        acc("q_norm_w", dq_norm)
        dkm, dk_norm = norm_bwd(dkn, a["km"], a["rstd_k"], w["k_norm_w"])
        del dkn
        acc("k_norm_w", dk_norm)

        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        if acc.wanted("wq"):
            acc("wq", h1.T @ dqm)
        if acc.wanted("wk"):
            acc("wk", h1.T @ dkm)
        if acc.wanted("wv"):
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


def build_olmoe_resolver(
    dims: OlmoeDims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    kernels = kernels if kernels is not None else resolve_kernels()
    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "prologue_round": RoundPrologue(dims, kernels),
        "moeattn_fwd": OlmoeBlockFwd(dims, kernels),
        "moeattn_recompute": OlmoeBlockRecompute(dims, kernels),
        "moeattn_bwd": OlmoeBlockBwd(dims, kernels),
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(
            dims, kernels, hyper,
            layout_for=lambda d, task, size: (
                olmoe_weight_layout(d, layer=AdamWStep.layer_of(task)), None,
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
