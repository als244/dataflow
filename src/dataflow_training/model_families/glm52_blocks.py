"""GLM-5.2 IndexShare block executables: dsv32's DSA blocks + cross-layer
metadata consumption.

Roles (per Glm52Dims.indexer_types):
- LEADERS (gdl dense-FFN / gml MoE): dsv32 blocks verbatim — compute the
  lightning indexer, select, emit their M (dsa_idx first, then the
  routing pack for gml). The KL target generalizes to the paper's
  multi-layer distillation (arXiv 2603.12201, Prop 1): when the group
  has followers, the leader's bwd consumes dM (the followers'
  accumulated, per-member L1-normalized targets gathered at the shared
  selection), adds its OWN target, and chains
  dI = sigma - (p_own + dM)/N through its indexer weights. Singleton
  leaders degenerate to dsv32's per-layer KL exactly (no dM object).
- FOLLOWERS (gmf, MoE only): NO indexer weights. Plain dsv3 MLA stages +
  the sparse core reading the PRODUCER's M ("dsa_idx" at offset 0 — a
  pinned ABI so followers never need the producer's kind) + their own
  routing M. Their bwd contributes the layer's target into dM
  (create/mutate per the reverse-order chain) and runs the standard
  low-rank chains — no indexer gradient anywhere.

Metadata plumbing convention (family-local, documented): the launch-time
merge threads cross-layer state through the saved-state dict — the
follower's `a` gains "dsa_idx" (the shared selection view); leader and
follower bwds gain "_dm_view" / "_dm_create" / "_kl_n" when the group
has an accumulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from dataflow.core import TaskSpec

from ..blocks import ops
from dataflow.runtime.interop import torch_view
from ..kernels import KernelSet, resolve_kernels
from ..blocks.layouts import (
    Glm52Dims,
    PackedLayout,
    dsv3_dense_activation_layout,
    dsv3_moe_weight_layout,
    dsv32_dense_weight_layout,
    dsv32_moe_weight_layout,
    glm52_aux_temp_layout,
)
from ..blocks.base_blocks import AdamWHyper, AdamWStep, EmbedBwd, EmbedFwd, HeadLoss, RoundPrologue
from .llama3_blocks import BlockRecompute
from .dsv3_blocks import Dsv3DenseBlockFwd, Dsv3MoeBlockBwd, Dsv3MoeBlockFwd
from .dsv32_blocks import (
    Dsv32DenseBlockBwd,
    Dsv32DenseBlockFwd,
    Dsv32MoeBlockBwd,
    Dsv32MoeBlockFwd,
    Dsv32ProfileFill,
    Dsv32WarmupDenseBlockBwd,
    Dsv32WarmupDenseBlockFwd,
    Dsv32WarmupDenseBlockRecompute,
    Dsv32WarmupMoeBlockBwd,
    Dsv32WarmupMoeBlockFwd,
    Dsv32WarmupMoeBlockRecompute,
    _IDX_FIELDS,
    _causal_bits,
    _seq_bounds,
)
from ..blocks.modules.moe.stages import MOE_SHARED_NOGATE_STAGES, moe_mlp_tail_bwd


def _glm52_moe_activation_layout(dims: Glm52Dims) -> PackedLayout:
    from ..blocks.layouts import _dsv3_attn_ctx_specs, _warmup_ctx_filter
    from ..blocks.modules.moe.spec import moe_context_specs

    return PackedLayout.build(_warmup_ctx_filter(
        _dsv3_attn_ctx_specs(dims) + moe_context_specs(dims, dims.moe, aux_temp=True),
        dims))


def _dm_cols(d) -> int:
    """dM row width: topk-gathered targets in sparse mode; FULL-PREFIX
    rows (seq_len) in dense warm-up."""
    return d.index_topk if getattr(d, "sparse_mode", True) else d.seq_len


class Glm52AuxTempState:
    """Own M + (followers) producer's M + (grouped bwds) dM."""

    AUX_TEMP_KIND = "gdl"

    def _aux_temp_layout(self):
        return glm52_aux_temp_layout(self.dims, self.AUX_TEMP_KIND)

    def _aux_temp_state(self, ctx):
        d = self.dims
        layer = ctx.task.block_params["layer"]
        layout = self._aux_temp_layout()
        key = ctx.task.compute_block_key
        consuming = key.endswith(("_recompute", "_bwd"))
        st: dict = {}
        buf_of = {}
        if consuming:
            for j, oid in enumerate(ctx.task.inputs):
                if oid.startswith("AuxTemp_"):
                    buf_of[int(oid.rsplit("_", 1)[1])] = self._in(ctx, j)
                elif oid.startswith("dAuxTemp_"):
                    st["_dm_view"] = torch_view(
                        self._in(ctx, j), (d.tokens, _dm_cols(d)), torch.float32)
            for oid in ctx.task.mutates:
                if oid.startswith("dAuxTemp_"):
                    st["_dm_view"] = torch_view(
                        ctx.mutates[oid], (d.tokens, _dm_cols(d)), torch.float32)
            if key.endswith("_bwd"):
                for j, o in enumerate(ctx.task.outputs):
                    if o.id.startswith("dAuxTemp_"):
                        st["_dm_view"] = torch_view(
                            self._out(ctx, j), (d.tokens, _dm_cols(d)),
                            torch.float32)
                        st["_dm_create"] = True
            if key.endswith("_recompute"):
                st["aux_temp_ready"] = True
        else:
            for j, o in enumerate(ctx.task.outputs):
                if o.id.startswith("AuxTemp_"):
                    buf_of[layer] = self._out(ctx, j)
            for j, oid in enumerate(ctx.task.inputs):
                if oid.startswith("AuxTemp_"):
                    buf_of[int(oid.rsplit("_", 1)[1])] = self._in(ctx, j)
        if layout.fields and layer in buf_of:
            st["aux_temp"] = layout.views(buf_of[layer])
        producer = d.leader_of(layer)
        if (getattr(d, "sparse_mode", True)
                and producer != layer and producer in buf_of):
            # the shared selection: dsa_idx is the FIRST field (offset 0)
            # in every producer layout — kind-agnostic by construction
            st["shared_idx"] = torch_view(
                buf_of[producer], (d.tokens, d.index_topk), torch.int32)
        st["_kl_n"] = len(d.group_members(producer))
        return st or None


class Glm52ProfileFill(Dsv32ProfileFill):
    """glm52 fill: floats (skipping M_/dM_), own-M seeding via the kind's
    layout, PRODUCER-M seeding as a sliding-window selection at offset 0,
    dM zeroed."""

    def profile_fill(self, ctx) -> None:
        import hashlib as _hl

        d = self.dims
        layer = ctx.task.block_params["layer"]
        key = ctx.task.compute_block_key
        seed = int.from_bytes(_hl.sha256(key.encode()).digest()[:4], "little")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        for oid in ctx.task.inputs:
            if oid.startswith(("AuxTemp_", "dAuxTemp_")):
                continue
            b = ctx.inputs[oid]
            n = b.size_bytes // 2
            v = torch_view(b, (n,), torch.bfloat16)
            v.copy_(torch.rand(n, generator=gen, dtype=torch.bfloat16,
                               device="cuda").sub_(0.5).mul_(0.05))
        layout = self._aux_temp_layout()
        for oid in ctx.task.inputs:
            if oid.startswith("dAuxTemp_"):
                torch_view(ctx.inputs[oid], (d.tokens, _dm_cols(d)),
                           torch.float32).zero_()
                continue
            if not oid.startswith("AuxTemp_"):
                continue
            own = int(oid.rsplit("_", 1)[1]) == layer
            if own and layout.fields:
                m = layout.views(ctx.inputs[oid])
            elif not getattr(d, "sparse_mode", True):
                # warm-up: producer M is routing-only (no selection) and
                # the follower never reads it — nothing to seed
                continue
            else:
                m = {"dsa_idx": torch_view(ctx.inputs[oid],
                                           (d.tokens, d.index_topk),
                                           torch.int32)}
            if "dsa_idx" in m:
                idx = m["dsa_idx"]
                rows = torch.arange(d.tokens, device="cuda").unsqueeze(1)
                offs = torch.arange(d.index_topk, device="cuda").unsqueeze(0)
                lo_of = torch.empty(d.tokens, dtype=torch.long, device="cuda")
                lo = 0
                # profiler seed data: dims-derived (uniform) per-seq lengths
                for L in ops.Segments.of_dims(d).lengths:
                    lo_of[lo:lo + L] = lo
                    lo += L
                idx.copy_(torch.maximum(rows - offs,
                                        lo_of.unsqueeze(1)).to(idx.dtype))
            if "route_ids" in m:
                moe = d.moe
                rows_n = m["route_order"].shape[0]
                flat = (torch.arange(rows_n, dtype=torch.int64, device="cuda")
                        % moe.n_experts)
                m["route_ids"].copy_(flat.view(d.tokens, moe.top_k).to(torch.int32))
                m["route_order"].copy_(torch.argsort(flat, stable=True).to(torch.int32))
                counts = torch.bincount(flat, minlength=moe.n_experts)
                m["route_offsets"][:1].zero_()
                m["route_offsets"][1:].copy_(counts.cumsum(0).to(torch.int32))
                m["route_w"].fill_(1.0 / moe.top_k)


class _Glm52LeaderKL:
    """Leader backward KL: dI = sigma - (p_own + dM)/N when followers
    exist (dM consumed), else dsv32's per-layer rule via super()."""

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum, aux_temp=None):
        if aux_temp:
            a = {**a, **aux_temp.get("aux_temp", {})}
            for k in ("_dm_view", "_dm_create", "_kl_n"):
                if k in aux_temp:
                    a[k] = aux_temp[k]
        super(Dsv32DenseBlockBwd, self)._backward(
            kctx, dy, a, x, w, dx_out, dw, accum)

    def _indexer_kl_bwd(self, kctx, d, a, x, w, acc, h1, q_lora_n,
                        q_full, k_full, idx, lse, bounds, pos,
                        bits_by_seq=None):
        dm = a.get("_dm_view")
        if dm is None:
            return super()._indexer_kl_bwd(
                kctx, d, a, x, w, acc, h1, q_lora_n, q_full, k_full,
                idx, lse, bounds, pos, bits_by_seq=bits_by_seq)
        from .dsv32_blocks import _indexer_inputs

        K = self.kernels
        t = d.tokens
        h, qk, rope = d.n_heads, d.qk_head_dim, d.qk_rope_dim
        n = float(a["_kl_n"])
        q_idx, k_idx, wts = _indexer_inputs(kctx, K, d, h1, q_lora_n, w, pos)
        dq_idx = torch.empty_like(q_idx)
        dk_idx = torch.empty_like(k_idx)
        dwts = torch.empty_like(wts)
        hi_, di = d.index_n_heads, d.index_head_dim
        for si, (lo, hi) in enumerate(bounds):
            length = hi - lo
            iscores = torch.empty(length, length, dtype=torch.float32,
                                  device=x.device)
            K.dsa_index_scores(
                kctx, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi], iscores,
                n_heads=hi_, head_dim=di, seq_bounds=((0, length),),
            )
            local = (idx[lo:hi].long() - lo).clamp_(0, length - 1)
            m = torch.full((length, length), float("-inf"), device=x.device)
            m.scatter_(-1, local, 0.0)
            rows = torch.arange(length, device=x.device).unsqueeze(1)
            cols = torch.arange(length, device=x.device).unsqueeze(0)
            m.masked_fill_(cols > rows, float("-inf"))
            live = m == 0
            p = torch.empty(length, length, device=x.device)
            K.dsa_probs_sum(
                kctx, q_full, k_full, idx, lse, p,
                n_heads=h, head_dim=qk, seq_bounds=((lo, hi),),
                bits_by_seq=None if bits_by_seq is None else [bits_by_seq[si]],
            )
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
            # centroid target: own p + the followers' accumulated targets
            # (dM holds per-member L1-normalized rows gathered at idx)
            total = p
            total.scatter_add_(-1, local, dm[lo:hi])
            total = total / n
            # in-place trio (workspace audit) — see dsv32 counterpart
            iscores.add_(m)
            iscores.sub_(torch.logsumexp(iscores, -1, keepdim=True)).exp_()
            d_scores = iscores.sub_(total).masked_fill_(~live, 0.0)
            K.dsa_index_bwd(
                kctx, d_scores, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi],
                dq_idx[lo:hi], dk_idx[lo:hi], dwts[lo:hi],
                n_heads=hi_, head_dim=di, seq_bounds=((0, length),),
            )
            del m, p, d_scores, total
        # indexer weight chains — identical to dsv32's tail
        self._indexer_chains(kctx, d, w, acc, h1, q_lora_n,
                             dq_idx, dk_idx, dwts, pos, x)

    def _indexer_chains(self, kctx, d, w, acc, h1, q_lora_n,
                        dq_idx, dk_idx, dwts, pos, x):
        import torch.nn.functional as F  # noqa: F401

        from .dsv32_blocks import _LN_EPS

        K = self.kernels
        t = d.tokens
        rope = d.qk_rope_dim
        hi_, di = d.index_n_heads, d.index_head_dim
        K.rope_bwd(kctx, dq_idx, dq_idx, pos, hi_, rope, d.rope_base,
                   row_stride=hi_ * di, head_stride=di, col_base=0)
        if acc.wanted("w_idx_q"):
            acc("w_idx_q", q_lora_n.T @ dq_idx)
        del dq_idx
        K.rope_bwd(kctx, dk_idx, dk_idx, pos, 1, rope, d.rope_base,
                   row_stride=di, head_stride=di, col_base=0)
        dk_post_ln = dk_idx
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
        if acc.wanted("w_idx_k"):
            acc("w_idx_k", h1.T @ dk_pre.to(torch.bfloat16))
        del k_pre, mu, xc, var, rstd, xhat, g, dk_post_ln, dk_pre
        if acc.wanted("w_idx_w"):
            acc("w_idx_w", (h1.float().T @ (dwts * (hi_ ** -0.5) * (di ** -0.5))))


# ---- LEADER kinds (dsv32 blocks + glm52 layouts) ---------------------------


@dataclass(frozen=True)
class Glm52DlBlockFwd(Glm52AuxTempState, Glm52ProfileFill, Dsv32DenseBlockFwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gdl"

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv3_dense_activation_layout(self.dims)


@dataclass(frozen=True)
class Glm52DlBlockRecompute(Glm52DlBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Glm52DlBlockBwd(_Glm52LeaderKL, Glm52AuxTempState, Glm52ProfileFill,
                      Dsv32DenseBlockBwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gdl"

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv3_dense_activation_layout(self.dims)


@dataclass(frozen=True)
class Glm52MlBlockFwd(Glm52AuxTempState, Glm52ProfileFill, Dsv32MoeBlockFwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gml"

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return _glm52_moe_activation_layout(self.dims)


@dataclass(frozen=True)
class Glm52MlBlockRecompute(Glm52MlBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Glm52MlBlockBwd(_Glm52LeaderKL, Glm52AuxTempState, Glm52ProfileFill,
                      Dsv32MoeBlockBwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gml"

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return _glm52_moe_activation_layout(self.dims)


# ---- FOLLOWER kind (dsv3 MLA + shared selection; no indexer) ---------------


@dataclass(frozen=True)
class Glm52MfBlockFwd(Glm52AuxTempState, Glm52ProfileFill, Dsv32MoeBlockFwd):
    """Follower forward: PLAIN dsv3 MLA stages (no indexer tap, no
    select) + the sparse core on the producer's selection + MoE tail."""

    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gmf"

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return _glm52_moe_activation_layout(self.dims)

    STAGES = (
        Dsv3DenseBlockFwd.MLA_STAGES[0],                     # attn_norm
        Dsv3DenseBlockFwd.MLA_STAGES[1],                     # mla_q (plain)
        Dsv3DenseBlockFwd.MLA_STAGES[2],                     # mla_kv (plain)
        ("dsa_attn", Dsv32DenseBlockFwd._stage_dsa_attn, ("lse", "attn_out")),
        Dsv3DenseBlockFwd.MLA_STAGES[4],                     # resid1_norm2
    ) + MOE_SHARED_NOGATE_STAGES


@dataclass(frozen=True)
class Glm52MfBlockRecompute(Glm52MfBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Glm52MfBlockBwd(Glm52AuxTempState, Glm52ProfileFill, Dsv32MoeBlockBwd):
    """Follower backward: sparse core bwd on the shared selection + the
    layer's target contribution into dM; NO indexer chains (the follower
    has no indexer weights)."""

    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gmf"

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return _glm52_moe_activation_layout(self.dims)

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum, aux_temp=None):
        if aux_temp:
            a = {**a, **aux_temp.get("aux_temp", {})}
            if "shared_idx" in aux_temp:
                a["dsa_idx"] = aux_temp["shared_idx"]
            for k in ("_dm_view", "_dm_create", "_kl_n"):
                if k in aux_temp:
                    a[k] = aux_temp[k]
        super(Dsv32DenseBlockBwd, self)._backward(
            kctx, dy, a, x, w, dx_out, dw, accum)

    def _indexer_kl_bwd(self, kctx, d, a, x, w, acc, h1, q_lora_n,
                        q_full, k_full, idx, lse, bounds, pos,
                        bits_by_seq=None):
        # the follower's KL role: deposit its L1-normalized target,
        # gathered at the shared selection, into dM
        K = self.kernels
        h, qk = d.n_heads, d.qk_head_dim
        dm = a["_dm_view"]
        if a.get("_dm_create"):
            dm.zero_()
        for si, (lo, hi) in enumerate(bounds):
            length = hi - lo
            local = (idx[lo:hi].long() - lo).clamp_(0, length - 1)
            p = torch.empty(length, length, device=x.device)
            K.dsa_probs_sum(
                kctx, q_full, k_full, idx, lse, p,
                n_heads=h, head_dim=qk, seq_bounds=((lo, hi),),
                bits_by_seq=None if bits_by_seq is None else [bits_by_seq[si]],
            )
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
            dm[lo:hi] += p.gather(-1, local)
            del p


# ---------------------------------------------------------------------------
# Dense warm-up (sparse_mode=False): main model frozen and its wgrads/dgrads
# SKIPPED (the _WarmupKLMixin _backward from dsv32); only the indexer trains,
# against FULL-PREFIX targets. IndexShare twist: followers deposit their full
# L1-normalized attention rows into the group dM (t, seq_len); the leader's
# KL uses the group centroid (p_own + dM)/N.

class _Glm52WarmupLeaderKL:
    """Leader warm-up KL: dI = sigma_full - (p_own_full + dM_full)/N.
    Standalone mixin (NOT _Glm52LeaderKL — its _backward would resurrect
    the skipped main backward); borrows the weight-chain tail."""

    _indexer_chains = _Glm52LeaderKL._indexer_chains

    def _indexer_kl_bwd(self, kctx, d, a, x, w, acc, h1, q_lora_n,
                        q_full, k_full, idx, lse, bounds, pos,
                        bits_by_seq=None):
        dm = a.get("_dm_view")
        if dm is None:
            # singleton leader: dsv32's per-layer full-prefix rule
            return Dsv32DenseBlockBwd._indexer_kl_bwd(
                self, kctx, d, a, x, w, acc, h1, q_lora_n, q_full,
                k_full, idx, lse, bounds, pos, bits_by_seq=bits_by_seq)
        from .dsv32_blocks import _indexer_inputs

        K = self.kernels
        n = float(a["_kl_n"])
        q_idx, k_idx, wts = _indexer_inputs(kctx, K, d, h1, q_lora_n, w, pos)
        dq_idx = torch.empty_like(q_idx)
        dk_idx = torch.empty_like(k_idx)
        dwts = torch.empty_like(wts)
        hi_, di = d.index_n_heads, d.index_head_dim
        h, qk = d.n_heads, d.qk_head_dim
        for si, (lo, hi) in enumerate(bounds):
            length = hi - lo
            iscores = torch.empty(length, length, dtype=torch.float32,
                                  device=x.device)
            K.dsa_index_scores(
                kctx, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi], iscores,
                n_heads=hi_, head_dim=di, seq_bounds=((0, length),),
            )
            rows = torch.arange(length, device=x.device).unsqueeze(1)
            cols = torch.arange(length, device=x.device).unsqueeze(0)
            live = cols <= rows
            p = torch.empty(length, length, device=x.device)
            K.dsa_probs_sum(
                kctx, q_full, k_full, None, lse, p,
                n_heads=h, head_dim=qk, seq_bounds=((lo, hi),),
                bits_by_seq=[_causal_bits(length, x.device)],
            )
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
            # group centroid: own full rows + followers' deposited rows
            total = (p + dm[lo:hi, :length]) / n
            iscores.masked_fill_(~live, float("-inf"))
            iscores.sub_(torch.logsumexp(iscores, -1, keepdim=True))
            lv = a.get("_loss_view")
            if lv is not None:
                # reported objective: KL(centroid || sigma) per group —
                # iscores holds log-sigma at this point
                kl = (total * (total.clamp_min(1e-20).log() - iscores)) \
                    .masked_fill(~live, 0.0).sum()
                if a.get("_loss_create"):
                    lv.copy_(kl.reshape(1))
                    a["_loss_create"] = False
                else:
                    lv.add_(kl.reshape(1))
            iscores.exp_()
            d_scores = iscores.sub_(total).masked_fill_(~live, 0.0)
            K.dsa_index_bwd(
                kctx, d_scores, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi],
                dq_idx[lo:hi], dk_idx[lo:hi], dwts[lo:hi],
                n_heads=hi_, head_dim=di, seq_bounds=((0, length),),
            )
            del p, d_scores, total
        self._indexer_chains(kctx, d, w, acc, h1, q_lora_n,
                             dq_idx, dk_idx, dwts, pos, x)


class _Glm52WarmupFollowerKL:
    """Follower warm-up role: deposit the layer's FULL-PREFIX
    L1-normalized attention rows into the group dM. No indexer weights,
    no chains."""

    def _indexer_kl_bwd(self, kctx, d, a, x, w, acc, h1, q_lora_n,
                        q_full, k_full, idx, lse, bounds, pos,
                        bits_by_seq=None):
        K = self.kernels
        h, qk = d.n_heads, d.qk_head_dim
        dm = a["_dm_view"]
        if a.get("_dm_create"):
            dm.zero_()
        for si, (lo, hi) in enumerate(bounds):
            length = hi - lo
            p = torch.empty(length, length, device=x.device)
            K.dsa_probs_sum(
                kctx, q_full, k_full, None, lse, p,
                n_heads=h, head_dim=qk, seq_bounds=((lo, hi),),
                bits_by_seq=[_causal_bits(length, x.device)],
            )
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
            dm[lo:hi, :length] += p
            del p


@dataclass(frozen=True)
class Glm52WarmupDlBlockFwd(Glm52AuxTempState, Dsv32WarmupDenseBlockFwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gdl"

    @property
    def cl(self) -> PackedLayout:
        return dsv3_dense_activation_layout(self.dims)


@dataclass(frozen=True)
class Glm52WarmupDlBlockRecompute(Glm52WarmupDlBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Glm52WarmupDlBlockBwd(Glm52AuxTempState, _Glm52WarmupLeaderKL,
                            Dsv32WarmupDenseBlockBwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gdl"

    @property
    def cl(self) -> PackedLayout:
        return dsv3_dense_activation_layout(self.dims)


@dataclass(frozen=True)
class Glm52WarmupMlBlockFwd(Glm52AuxTempState, Glm52ProfileFill,
                            Dsv32WarmupMoeBlockFwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gml"

    @property
    def cl(self) -> PackedLayout:
        return _glm52_moe_activation_layout(self.dims)


@dataclass(frozen=True)
class Glm52WarmupMlBlockRecompute(Glm52WarmupMlBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Glm52WarmupMlBlockBwd(Glm52AuxTempState, Glm52ProfileFill,
                            _Glm52WarmupLeaderKL, Dsv32WarmupMoeBlockBwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gml"

    @property
    def cl(self) -> PackedLayout:
        return _glm52_moe_activation_layout(self.dims)


@dataclass(frozen=True)
class Glm52WarmupMfBlockFwd(Glm52AuxTempState, Glm52ProfileFill,
                            Dsv32WarmupMoeBlockFwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gmf"

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return _glm52_moe_activation_layout(self.dims)


@dataclass(frozen=True)
class Glm52WarmupMfBlockRecompute(Glm52WarmupMfBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Glm52WarmupMfBlockBwd(Glm52AuxTempState, Glm52ProfileFill,
                            _Glm52WarmupFollowerKL, Dsv32WarmupMoeBlockBwd):
    dims: Glm52Dims = None  # type: ignore[assignment]
    AUX_TEMP_KIND = "gmf"

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv3_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return _glm52_moe_activation_layout(self.dims)


def build_glm52_resolver(
    dims: Glm52Dims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    kernels = kernels if kernels is not None else resolve_kernels()

    if not dims.sparse_mode and not dims.train_indexer:
        raise ValueError(
            "dense warm-up trains ONLY the indexer; train_indexer=False "
            "in dense mode would train nothing"
        )
    _WL = {
        "gdl": dsv32_dense_weight_layout,
        "gml": dsv32_moe_weight_layout,
        "gmf": dsv3_moe_weight_layout,
    }

    def _opt_layout(d, task, size):
        layer = AdamWStep.layer_of(task)
        return _WL[d.kinds[layer]](d, layer=layer), None

    if dims.sparse_mode:
        blocks = {
            "gdl_fwd": Glm52DlBlockFwd(dims, kernels),
            "gdl_recompute": Glm52DlBlockRecompute(dims, kernels),
            "gdl_bwd": Glm52DlBlockBwd(dims, kernels),
            "gml_fwd": Glm52MlBlockFwd(dims, kernels),
            "gml_recompute": Glm52MlBlockRecompute(dims, kernels),
            "gml_bwd": Glm52MlBlockBwd(dims, kernels),
            "gmf_fwd": Glm52MfBlockFwd(dims, kernels),
            "gmf_recompute": Glm52MfBlockRecompute(dims, kernels),
            "gmf_bwd": Glm52MfBlockBwd(dims, kernels),
        }
        opt_embed = AdamWStep(dims, kernels, hyper, kind="embed")
        opt_head = AdamWStep(dims, kernels, hyper, kind="head")
    else:
        blocks = {
            "gdl_fwd": Glm52WarmupDlBlockFwd(dims, kernels),
            "gdl_recompute": Glm52WarmupDlBlockRecompute(dims, kernels),
            "gdl_bwd": Glm52WarmupDlBlockBwd(dims, kernels),
            "gml_fwd": Glm52WarmupMlBlockFwd(dims, kernels),
            "gml_recompute": Glm52WarmupMlBlockRecompute(dims, kernels),
            "gml_bwd": Glm52WarmupMlBlockBwd(dims, kernels),
            "gmf_fwd": Glm52WarmupMfBlockFwd(dims, kernels),
            "gmf_recompute": Glm52WarmupMfBlockRecompute(dims, kernels),
            "gmf_bwd": Glm52WarmupMfBlockBwd(dims, kernels),
        }

        opt_embed = AdamWStep(dims, kernels, hyper, kind="embed")
        opt_head = AdamWStep(dims, kernels, hyper, kind="head")
    table = {
        "embed_fwd": EmbedFwd(dims, kernels),
        "prologue_round": RoundPrologue(dims, kernels),
        **blocks,
        "head_loss": HeadLoss(dims, kernels),
        "embed_bwd": EmbedBwd(dims, kernels),
        "optimizer_block": AdamWStep(
            dims, kernels, hyper, layout_for=_opt_layout,
        ),
        "optimizer_embed": opt_embed,
        "optimizer_head": opt_head,
    }

    def resolver(task: TaskSpec):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key {key!r} (task {task.id!r})")
        return table[key]

    resolver.kernel_set = kernels
    return resolver
