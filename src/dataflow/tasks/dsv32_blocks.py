"""DeepSeek-V3.2 block executables: dsv3's MLA + DSA sparse attention.

Deltas from dsv3_blocks (everything else inherited):
- mla_q additionally KEEPS the post-norm q_lora locally (indexer taps it);
  mla_kv additionally computes the indexer inputs from h1 before popping
  it: k^I = rope(LayerNorm(h1 @ w_idx_k)) and the fp32 scaled weights.
- NEW stage dsa_select: per-sequence indexer scores (ReLU-weighted
  head-sum) -> stable top-k -> ctx ``dsa_idx`` (t, k) int32. The (L, L)
  score matrix is a PER-SEQUENCE local (never (t, t) across the batch).
- mla_attn -> dsa sparse core: masked softmax over the padded-v MLA
  tensors via ``dsa_sparse_attn_fwd`` (emits the MASKED lse).
- _attn_bwd: ``dsa_sparse_attn_bwd`` replaces flash_bwd (same low-rank
  chains after), then the INDEXER KL INJECTION: rebuild the indexer
  inputs, per sequence recompute the head-summed attention target p from
  the saved masked lse, dI = softmax_live(I) - p, chain through the four
  indexer weights ONLY (the detachment seam: zero dh1/dq_lora
  contribution — report: 'we detach the indexer input').

Training-schedule note (documented, deliberate): the paper trains the
indexer at its own learning rate; our single optimizer trains it at the
shared AdamW lr. The GOLDEN uses the same convention, so parity gates
compare like against like.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dataflow.core import TaskSpec

from . import ops
from .kernels import KernelSet, resolve_kernels
from .layouts import (
    Dsv32Dims,
    PackedLayout,
    dsv32_dense_context_layout,
    dsv32_dense_weight_layout,
    dsv32_moe_context_layout,
    dsv32_moe_weight_layout,
)
from .llama3_blocks import AdamWHyper, AdamWStep, BlockRecompute, EmbedBwd, EmbedFwd, HeadLoss
from .dsv3_blocks import (
    Dsv3DenseBlockBwd,
    Dsv3DenseBlockFwd,
    _mla_expand_kv,
    _mla_expand_q,
    _pad_v,
)
from .moe.stages import MOE_SHARED_NOGATE_STAGES, MoEProfileFill, moe_bias_update, moe_mlp_tail_bwd

_LN_EPS = 1e-5


def _seq_bounds(d):
    lens = ops.seq_lens_of(d.seq_spec, d.tokens)
    return tuple(ops.seq_bounds_of(lens, d.tokens))


def _indexer_inputs(kctx, K, d, h1, q_lora_n, w, pos):
    """(q_idx (t, H*dI), k_idx (t, dI), wts (t, H) fp32-scaled) — the
    rope-first assembled indexer tensors."""
    t = h1.shape[0]
    hi, di, rope = d.index_n_heads, d.index_head_dim, d.qk_rope_dim
    q = (q_lora_n @ w["w_idx_q"]).view(t, hi, di)
    q_pe = torch.empty(t, hi * rope, dtype=q.dtype, device=q.device)
    K.rope_fwd(kctx, q[..., :rope].reshape(t, hi * rope).contiguous(), q_pe,
               pos, hi, rope, d.rope_base)
    q_idx = torch.cat([q_pe.view(t, hi, rope), q[..., rope:]], dim=-1
                      ).reshape(t, hi * di).contiguous()
    k_pre = h1 @ w["w_idx_k"]
    k = F.layer_norm(k_pre.float(), (di,), w["idx_k_ln_w"].float(),
                     w["idx_k_ln_b"].float(), _LN_EPS).to(k_pre.dtype)
    k_pe = torch.empty(t, rope, dtype=k.dtype, device=k.device)
    K.rope_fwd(kctx, k[:, :rope].contiguous(), k_pe, pos, 1, rope, d.rope_base)
    k_idx = torch.cat([k_pe, k[:, rope:]], dim=-1).contiguous()
    wts = (h1.float() @ w["w_idx_w"].float()) \
        * (hi ** -0.5) * (di ** -0.5)
    return q_idx, k_idx, wts.contiguous()


class Dsv32ProfileFill(MoEProfileFill):
    """Profiling fill for DSA blocks: floats (+ MoE routing when the kind
    has it) via the base mixin, PLUS valid ``dsa_idx`` seeding for bwd
    signatures (sliding-window within each sequence — garbage indices
    are out-of-range scatter targets in the mask rebuild)."""

    def profile_fill(self, ctx) -> None:
        from ..interop import torch_view

        cl = self.cl
        names = {f.name for f in cl.fields}
        if "route_ids" in names:
            super().profile_fill(ctx)          # floats + MoE routing
        else:
            # dense kind: floats only (base's loop, sans MoE branch)
            import hashlib as _hl

            key = ctx.task.compute_block_key
            seed = int.from_bytes(_hl.sha256(key.encode()).digest()[:4], "little")
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed)
            for oid in ctx.task.inputs:
                b = ctx.inputs[oid]
                n = b.size_bytes // 2
                v = torch_view(b, (n,), torch.bfloat16)
                v.copy_(
                    torch.rand(n, generator=gen, dtype=torch.bfloat16,
                               device="cuda").sub_(0.5).mul_(0.05)
                )
        if not ctx.task.compute_block_key.endswith("_bwd"):
            return
        d = self.dims
        a_buf = ctx.inputs[ctx.task.inputs[1]]
        idx = cl.view(a_buf, "dsa_idx")
        t, k = idx.shape
        rows = torch.arange(t, device="cuda").unsqueeze(1)
        offs = torch.arange(k, device="cuda").unsqueeze(0)
        lo_of = torch.empty(t, dtype=torch.long, device="cuda")
        lo = 0
        for L in ops.seq_lens_of(d.seq_spec, t):
            lo_of[lo:lo + L] = lo
            lo += L
        seeded = torch.maximum((rows - offs), lo_of.unsqueeze(1))
        idx.copy_(seeded.to(idx.dtype))


@dataclass(frozen=True)
class Dsv32DenseBlockFwd(Dsv32ProfileFill, Dsv3DenseBlockFwd):
    # Dsv32ProfileFill is mixed in via the concrete class list below
    # (kept off this base to keep the stage-definition class minimal)
    dims: Dsv32Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_dense_context_layout(self.dims)

    @staticmethod
    def _stage_mla_q(kctx, K, d, st):
        Dsv3DenseBlockFwd._stage_mla_q(kctx, K, d, st)
        # the indexer taps the post-norm q_lora: recompute it here from
        # the ctx/local q_a (cheap (t, q_lora) norm) and stash for select
        a = st["a"]
        if a is not None:
            q_a, rstd = a["q_a"], a["rstd_qa"]
        else:
            q_a, rstd = None, None
        # mla_q's body already produced q_lora_n locally; rebuild it the
        # same way (rmsnorm_apply from the write-through pre-norm value)
        w = st["w"]
        src = a["q_a"] if a is not None else None
        if src is None:
            # scratch path: recompute from x is unnecessary — mla_q kept
            # nothing; recompute from st? Rebuild from h1 kept in st.
            src = st["h1"] @ w["w_q_a"]
            q_lora_n = torch.empty_like(src)
            r = torch.empty(d.tokens, dtype=torch.float32, device=src.device)
            K.rmsnorm_fwd(kctx, src, w["q_a_norm_w"], q_lora_n, r)
        else:
            q_lora_n = torch.empty_like(src)
            K.rmsnorm_apply(kctx, src, a["rstd_qa"], w["q_a_norm_w"], q_lora_n)
        st["q_lora_n_idx"] = q_lora_n

    @staticmethod
    def _stage_mla_kv(kctx, K, d, st):
        # indexer inputs need h1 — compute BEFORE the base stage pops it
        h1, w = st["h1"], st["w"]
        q_idx, k_idx, wts = _indexer_inputs(
            kctx, K, d, h1, st.pop("q_lora_n_idx"), w, st["pos"],
        )
        st.update(q_idx=q_idx, k_idx=k_idx, idx_wts=wts)
        Dsv3DenseBlockFwd._stage_mla_kv(kctx, K, d, st)

    @staticmethod
    def _stage_dsa_select(kctx, K, d, st):
        a = st["a"]
        t, k_sel = d.tokens, d.index_topk
        q_idx, k_idx, wts = st.pop("q_idx"), st.pop("k_idx"), st.pop("idx_wts")
        if a is not None:
            idx = a["dsa_idx"]
        else:
            idx = torch.empty(t, k_sel, dtype=torch.int32, device=q_idx.device)
        for lo, hi in _seq_bounds(d):
            length = hi - lo
            scores = torch.empty(length, length, dtype=torch.float32,
                                 device=q_idx.device)
            K.dsa_index_scores(
                kctx, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi], scores,
                n_heads=d.index_n_heads, head_dim=d.index_head_dim,
                seq_bounds=((0, length),),
            )
            K.dsa_topk(kctx, scores, idx[lo:hi])
            idx[lo:hi].add_(lo)  # sequence-local -> global token ids
            del scores
        st["idx"] = idx

    @staticmethod
    def _stage_dsa_attn(kctx, K, d, st):
        a = st["a"]
        t, h, qk, v = d.tokens, d.n_heads, d.qk_head_dim, d.v_head_dim
        out_pad = torch.empty(t, h * qk, dtype=torch.bfloat16,
                              device=st["q_full"].device)
        lse = torch.empty(h, t, dtype=torch.float32, device=out_pad.device)
        K.dsa_sparse_attn_fwd(
            kctx, st.pop("q_full"), st.pop("k_full"), st.pop("v_pad"),
            st.pop("idx"), out_pad, lse,
            n_heads=h, head_dim=qk, seq_bounds=_seq_bounds(d),
        )
        attn_out = out_pad.view(t, h, qk)[..., :v].reshape(t, h * v).contiguous()
        del out_pad
        if a is not None:
            a["lse"].copy_(lse.reshape(a["lse"].shape))
            a["attn_out"].copy_(attn_out)
        st.update(attn_out=attn_out, lse=lse)

    STAGES = (
        Dsv3DenseBlockFwd.MLA_STAGES[0],                    # attn_norm
        ("mla_q", _stage_mla_q.__func__, ("q_a", "rstd_qa")),
        ("mla_kv", _stage_mla_kv.__func__, ("kv_a", "rstd_kva")),
        ("dsa_select", _stage_dsa_select.__func__, ("dsa_idx",)),
        ("dsa_attn", _stage_dsa_attn.__func__, ("lse", "attn_out")),
        Dsv3DenseBlockFwd.MLA_STAGES[4],                    # resid1_norm2
    ) + Dsv3DenseBlockFwd.STAGES[5:]                        # dense FFN tail

    MLA_STAGES = STAGES[:6]


@dataclass(frozen=True)
class Dsv32DenseBlockRecompute(Dsv32DenseBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Dsv32DenseBlockBwd(Dsv32ProfileFill, Dsv3DenseBlockBwd):
    dims: Dsv32Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_dense_context_layout(self.dims)

    def _attn_bwd(self, kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out) -> None:
        d = self.dims
        K = self.kernels
        t = d.tokens
        h, nope, rope, v = d.n_heads, d.qk_nope_dim, d.qk_rope_dim, d.v_head_dim
        qk, kvl = d.qk_head_dim, d.kv_lora_rank
        bounds = _seq_bounds(d)

        d_attn_v = dh_mid @ w["wo"].T
        acc("wo", a["attn_out"].T @ dh_mid)

        pos = ops.positions_for(d.seq_spec, t, x.device)
        q_lora_n, q_full = _mla_expand_q(kctx, K, d, a["q_a"], a["rstd_qa"], w, pos)
        latent_n, k_full, vals = _mla_expand_kv(
            kctx, K, d, a["kv_a"], a["rstd_kva"], w, pos,
        )
        v_pad = _pad_v(vals, qk)
        del vals
        d_attn_pad = _pad_v(d_attn_v.view(t, h, v), qk)
        del d_attn_v

        lse = a["lse"].reshape(h, t)
        idx = a["dsa_idx"]
        dq = torch.empty_like(q_full)
        dk = torch.empty_like(k_full)
        dv_pad = torch.empty_like(v_pad)
        K.dsa_sparse_attn_bwd(
            kctx, d_attn_pad, q_full, k_full, v_pad, idx, lse,
            dq, dk, dv_pad, n_heads=h, head_dim=qk, seq_bounds=bounds,
        )
        del d_attn_pad

        # ---- indexer KL injection (detached seam: touches ONLY idx weights)
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        q_idx, k_idx, wts = _indexer_inputs(kctx, K, d, h1, q_lora_n, w, pos)
        dq_idx = torch.empty_like(q_idx)
        dk_idx = torch.empty_like(k_idx)
        dwts = torch.empty_like(wts)
        scale_qk = qk ** -0.5
        hi_, di = d.index_n_heads, d.index_head_dim
        q3 = q_full.view(t, h, qk)
        k3 = k_full.view(t, h, qk)
        for lo, hi in bounds:
            length = hi - lo
            # rebuild indexer scores for this sequence
            iscores = torch.empty(length, length, dtype=torch.float32,
                                  device=x.device)
            K.dsa_index_scores(
                kctx, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi], iscores,
                n_heads=hi_, head_dim=di, seq_bounds=((0, length),),
            )
            # mask from saved selection (scatter + causal — pad-safe)
            m = torch.full((length, length), float("-inf"), device=x.device)
            m.scatter_(-1, (idx[lo:hi].long() - lo).clamp_(0, length - 1), 0.0)
            rows = torch.arange(length, device=x.device).unsqueeze(1)
            cols = torch.arange(length, device=x.device).unsqueeze(0)
            m.masked_fill_(cols > rows, float("-inf"))
            live = m == 0
            # target p: head-summed attention probs from the saved lse
            p = torch.zeros(length, length, device=x.device)
            for hh in range(h):
                lg = (q3[lo:hi, hh].float() @ k3[lo:hi, hh].float().T) * scale_qk
                p += torch.exp(lg + m - lse[hh, lo:hi].unsqueeze(1))
            p = p.masked_fill(~live, 0.0)
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
            sig = torch.softmax(iscores + m, dim=-1)
            d_scores = (sig - p).masked_fill(~live, 0.0)
            K.dsa_index_bwd(
                kctx, d_scores, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi],
                dq_idx[lo:hi], dk_idx[lo:hi], dwts[lo:hi],
                n_heads=hi_, head_dim=di, seq_bounds=((0, length),),
            )
            del iscores, m, p, sig, d_scores
        # chain to the four indexer weights (inputs detached)
        dq_pre = torch.empty_like(dq_idx)
        dq3i = dq_idx.view(t, hi_, di)
        rb = torch.empty(t, hi_ * rope, dtype=dq_idx.dtype, device=x.device)
        K.rope_bwd(kctx, dq3i[..., :rope].reshape(t, hi_ * rope).contiguous(),
                   rb, pos, hi_, rope, d.rope_base)
        dq_pre = torch.cat([rb.view(t, hi_, rope), dq3i[..., rope:]], dim=-1
                           ).reshape(t, hi_ * di).contiguous()
        acc("w_idx_q", q_lora_n.T @ dq_pre)
        del dq_idx, dq_pre, rb
        dk3i = dk_idx
        rbk = torch.empty(t, rope, dtype=dk_idx.dtype, device=x.device)
        K.rope_bwd(kctx, dk3i[:, :rope].contiguous(), rbk, pos, 1, rope,
                   d.rope_base)
        dk_post_ln = torch.cat([rbk, dk3i[:, rope:]], dim=-1)
        del dk_idx, rbk
        # LayerNorm backward (eager: standard formulas, fp32)
        k_pre = (h1 @ w["w_idx_k"]).float()
        mu = k_pre.mean(-1, keepdim=True)
        xc = k_pre - mu
        var = xc.pow(2).mean(-1, keepdim=True)
        rstd = (var + _LN_EPS).rsqrt()
        xhat = xc * rstd
        g = dk_post_ln.float() * w["idx_k_ln_w"].float()
        acc("idx_k_ln_w", (dk_post_ln.float() * xhat).sum(0).to(torch.bfloat16))
        acc("idx_k_ln_b", dk_post_ln.float().sum(0).to(torch.bfloat16))
        dk_pre = rstd * (g - g.mean(-1, keepdim=True)
                         - xhat * (g * xhat).mean(-1, keepdim=True))
        acc("w_idx_k", h1.T @ dk_pre.to(torch.bfloat16))
        del k_pre, mu, xc, var, rstd, xhat, g, dk_post_ln, dk_pre
        acc("w_idx_w", (h1.float().T @ (dwts * (hi_ ** -0.5) * (di ** -0.5))))
        del dwts, q_idx, k_idx, wts

        # ---- main-path chains (identical to dsv3 from here) ---------------
        dk3 = dk.view(t, h, qk)
        dv = dv_pad.view(t, h, qk)[..., :v]
        dkvb = torch.cat([dk3[..., :nope], dv], dim=-1).reshape(
            t, h * (nope + v)).contiguous()
        del dv_pad, dv
        dk_rope_sum = dk3[..., nope:].sum(dim=1).contiguous()
        del dk, dk3
        dk_rope_pre = torch.empty_like(dk_rope_sum)
        K.rope_bwd(kctx, dk_rope_sum, dk_rope_pre, pos, 1, rope, d.rope_base)
        del dk_rope_sum
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

        dq3 = dq.view(t, h, qk)
        dq_rope_pre = torch.empty(t, h * rope, dtype=dq.dtype, device=dq.device)
        K.rope_bwd(kctx, dq3[..., nope:].reshape(t, h * rope).contiguous(),
                   dq_rope_pre, pos, h, rope, d.rope_base)
        dq_pre_m = torch.cat(
            [dq3[..., :nope], dq_rope_pre.view(t, h, rope)], dim=-1,
        ).reshape(t, h * qk).contiguous()
        del dq, dq3, dq_rope_pre
        acc("w_q_b", q_lora_n.T @ dq_pre_m)
        dq_lora_n = dq_pre_m @ w["w_q_b"].T
        del dq_pre_m, q_lora_n
        dq_lora, d_q_norm = norm_bwd(dq_lora_n, a["q_a"], a["rstd_qa"],
                                     w["q_a_norm_w"])
        del dq_lora_n
        acc("q_a_norm_w", d_q_norm)

        acc("w_q_a", h1.T @ dq_lora)
        acc("w_kv_a", h1.T @ d_kv_a)
        dh1 = dq_lora @ w["w_q_a"].T
        dh1.addmm_(d_kv_a, w["w_kv_a"].T)
        del dq_lora, d_kv_a, h1
        dx_n, dattn_norm = norm_bwd(dh1, x, a["rstd_attn"], w["attn_norm_w"])
        del dh1
        acc("attn_norm_w", dattn_norm)
        torch.add(dh_mid, dx_n, out=dx_out)


@dataclass(frozen=True)
class Dsv32MoeBlockFwd(Dsv32DenseBlockFwd):
    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_moe_context_layout(self.dims)

    STAGES = Dsv32DenseBlockFwd.MLA_STAGES + MOE_SHARED_NOGATE_STAGES


@dataclass(frozen=True)
class Dsv32MoeBlockRecompute(Dsv32MoeBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Dsv32MoeBlockBwd(Dsv32DenseBlockBwd):
    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_moe_context_layout(self.dims)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        return moe_mlp_tail_bwd(
            kctx, self.kernels, self.dims, dy, a, w, dw, accum, acc, norm_bwd,
            resid_field=self.MLP_RESID_FIELD,
        )


def build_dsv32_resolver(
    dims: Dsv32Dims,
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
            return dsv32_dense_weight_layout(d, layer=layer), None
        return dsv32_moe_weight_layout(d, layer=layer), None

    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "dsadense_fwd": Dsv32DenseBlockFwd(dims, kernels),
        "dsadense_recompute": Dsv32DenseBlockRecompute(dims, kernels),
        "dsadense_bwd": Dsv32DenseBlockBwd(dims, kernels),
        "dsamoe_fwd": Dsv32MoeBlockFwd(dims, kernels),
        "dsamoe_recompute": Dsv32MoeBlockRecompute(dims, kernels),
        "dsamoe_bwd": Dsv32MoeBlockBwd(dims, kernels),
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
