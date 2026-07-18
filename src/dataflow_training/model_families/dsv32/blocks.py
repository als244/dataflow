"""DeepSeek-V3.2 block executables: dsv3's MLA + DSA sparse attention.

Deltas from dsv3/blocks.py (everything else inherited):
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

from ...blocks import ops
from ...kernels import KernelSet, resolve_kernels
from dataflow.runtime.interop import torch_view
from ...blocks.layouts import (
    Dsv32Dims,
    PackedLayout,
    dsv32_dense_activation_layout,
    dsv32_dense_weight_layout,
    dsv32_moe_activation_layout,
    dsv32_moe_weight_layout,
    dsv32_aux_temp_layout,
)
from ...blocks.base_blocks import AdamWHyper, AdamWStep, EmbedBwd, EmbedFwd, HeadLoss, RoundPrologue
from ..llama3.blocks import BlockRecompute
from ..dsv3.blocks import (
    Dsv3DenseBlockBwd,
    Dsv3DenseBlockFwd,
    Dsv3MoeBlockBwd,
    Dsv3MoeBlockFwd,
    Dsv3MoeBlockRecompute,
    _mla_expand_kv,
    _mla_expand_q,
)
from ...blocks.modules.moe.stages import MOE_SHARED_NOGATE_STAGES, MoEProfileFill, moe_mlp_tail_bwd

_LN_EPS = 1e-5


def _seq_bounds(seg):
    """Per-sequence (lo, hi) bounds for the round — from the round's Segments
    (DSA index scores / topk / sparse attention run per sequence)."""
    return tuple(seg.bounds)


def _causal_bits(length, device):
    # full-causal selection words: row r has bits set for cols 0..r
    words = (length + 63) // 64
    r = torch.arange(length, device=device).unsqueeze(1)
    w = torch.arange(words, device=device).unsqueeze(0)
    n_live = (r + 1 - 64 * w).clamp(0, 64)
    full = torch.full((), -1, dtype=torch.int64, device=device)
    partial = (torch.ones((), dtype=torch.int64, device=device)
               << n_live.clamp(0, 63)) - 1
    return torch.where(n_live >= 64, full, partial)


def _bits_for_bounds(idx, bounds, device):
    # per-sequence selection bitmasks (int64 words), packed once and
    # shared across sparse-bwd + KL kernels (eager impls ignore them)
    try:
        from ...kernels.dsa import _pack_local_bits
    except ImportError:
        return None
    out = []
    for lo, hi in bounds:
        length = hi - lo
        words = (length + 63) // 64
        bits = torch.empty(length, words, dtype=torch.int64, device=device)
        _pack_local_bits(idx[lo:hi], lo, length, bits)
        out.append(bits)
    return out


def _indexer_inputs(kctx, K, d, h1, q_lora_n, w, pos):
    """(q_idx (t, H*dI), k_idx (t, dI), wts (t, H) fp32-scaled) — the
    rope-first assembled indexer tensors."""
    # linear-triple conversion pending (exemplar: llama3)
    t = h1.shape[0]
    hi, di, rope = d.index_n_heads, d.index_head_dim, d.qk_rope_dim
    # in-place strided rope on the assembled projections (rope-FIRST
    # indexer layout: col_base=0) — no extracts, no temps, no cats
    q_idx = q_lora_n @ w["w_idx_q"]
    K.rope_fwd(kctx, q_idx, q_idx, pos, hi, rope, d.rope_base,
               row_stride=hi * di, head_stride=di, col_base=0)
    k_pre = h1 @ w["w_idx_k"]
    k_idx = F.layer_norm(k_pre.float(), (di,), w["idx_k_ln_w"].float(),
                         w["idx_k_ln_b"].float(), _LN_EPS).to(k_pre.dtype)
    K.rope_fwd(kctx, k_idx, k_idx, pos, 1, rope, d.rope_base,
               row_stride=di, head_stride=di, col_base=0)
    wts = (h1.float() @ w["w_idx_w"].float()) \
        * (hi ** -0.5) * (di ** -0.5)
    return q_idx, k_idx, wts.contiguous()


class Dsv32AuxTempState:
    """Metadata-object plumbing (one implementation for fwd/rc/bwd): the
    layer's M object packs ALL its never-recompute artifacts — the dsa
    selection ("dsa_idx") and the routing pack (moe kinds) — in one
    layout. Exposed to stages as st["aux_temp"] (views dict); recompute sets
    aux_temp_ready so the runner skips aux_temp-marked stages and the moe stages
    consume the decision verbatim — METADATA IS NEVER RECOMPUTED."""

    AUX_TEMP_KIND = "dense"  # overridden by moe classes

    def _aux_temp_layout(self):
        # hook: glm52 overrides with its per-kind layouts
        return dsv32_aux_temp_layout(self.dims, self.AUX_TEMP_KIND)

    def _aux_temp_state(self, ctx):
        layout = self._aux_temp_layout()
        if not layout.fields:
            return None
        key = ctx.task.compute_block_key
        if key.endswith(("_recompute", "_bwd")):
            for j, oid in enumerate(ctx.task.inputs):
                if oid.startswith("AuxTemp_"):
                    st = {"aux_temp": layout.views(self._in(ctx, j))}
                    if key.endswith("_recompute"):
                        st["aux_temp_ready"] = True
                    return st
            raise RuntimeError(f"no M_ input on {ctx.task.id}")
        for j, o in enumerate(ctx.task.outputs):
            if o.id.startswith("AuxTemp_"):
                return {"aux_temp": layout.views(self._out(ctx, j))}
        raise RuntimeError(f"no M_ output on {ctx.task.id}")


class Dsv32ProfileFill(MoEProfileFill):
    def _aux_temp_layout(self):
        return dsv32_aux_temp_layout(self.dims, self.AUX_TEMP_KIND)

    """Profiling fill: float inputs seeded deterministically (skipping the
    int-heavy M_ metadata inputs), then every M_ INPUT seeded validly per
    field — dsa_idx gets a sliding window, the routing pack balanced
    identity routing (garbage would be illegal gather/scatter targets)."""

    def profile_fill(self, ctx) -> None:
        import hashlib as _hl

        d = self.dims
        key = ctx.task.compute_block_key
        seed = int.from_bytes(_hl.sha256(key.encode()).digest()[:4], "little")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        for oid in ctx.task.inputs:
            if oid.startswith("AuxTemp_"):
                continue
            b = ctx.inputs[oid]
            n = b.size_bytes // 2
            v = torch_view(b, (n,), torch.bfloat16)
            v.copy_(
                torch.rand(n, generator=gen, dtype=torch.bfloat16,
                           device="cuda").sub_(0.5).mul_(0.05)
            )
        layout = self._aux_temp_layout()
        for oid in ctx.task.inputs:
            if not oid.startswith("AuxTemp_"):
                continue
            m = layout.views(ctx.inputs[oid])
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
                m["route_ids"].copy_(
                    flat.view(d.tokens, moe.top_k).to(torch.int32))
                m["route_order"].copy_(
                    torch.argsort(flat, stable=True).to(torch.int32))
                counts = torch.bincount(flat, minlength=moe.n_experts)
                m["route_offsets"][:1].zero_()
                m["route_offsets"][1:].copy_(counts.cumsum(0).to(torch.int32))
                m["route_w"].fill_(1.0 / moe.top_k)


@dataclass(frozen=True)
class Dsv32DenseBlockFwd(Dsv32AuxTempState, Dsv32ProfileFill, Dsv3DenseBlockFwd):
    # Dsv32ProfileFill is mixed in via the concrete class list below
    # (kept off this base to keep the stage-definition class minimal)
    dims: Dsv32Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_dense_activation_layout(self.dims)

    @staticmethod
    def _stage_mla_q(kctx, K, d, st):
        Dsv3DenseBlockFwd._stage_mla_q(kctx, K, d, st)
        if st.get("aux_temp_ready"):
            return  # selection supplied — the indexer tap is dead weight
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
            # linear-triple conversion pending (exemplar: llama3)
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
        if st.get("aux_temp_ready"):
            Dsv3DenseBlockFwd._stage_mla_kv(kctx, K, d, st)
            return
        # indexer inputs need h1 — compute BEFORE the base stage pops it
        h1, w = st["h1"], st["w"]
        q_idx, k_idx, wts = _indexer_inputs(
            kctx, K, d, h1, st.pop("q_lora_n_idx"), w, st["seg"].positions,
        )
        st.update(q_idx=q_idx, k_idx=k_idx, idx_wts=wts)
        Dsv3DenseBlockFwd._stage_mla_kv(kctx, K, d, st)

    @staticmethod
    def _stage_dsa_select(kctx, K, d, st):
        # writes the AuxTemp object's dsa_idx field; marked "aux_temp" in STAGES so
        # the runner SKIPS it whenever the metadata is supplied
        # (recompute) — metadata is never recomputed
        idx = st["aux_temp"]["dsa_idx"]
        q_idx, k_idx, wts = st.pop("q_idx"), st.pop("k_idx"), st.pop("idx_wts")
        for lo, hi in _seq_bounds(st["seg"]):
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

    @staticmethod
    def _stage_dsa_attn(kctx, K, d, st):
        a = st["a"]
        t, h, qk, v = d.tokens, d.n_heads, d.qk_head_dim, d.v_head_dim
        # native-DV core: our kernels don't need flash's equal-dims pad —
        # strip the base stage's zero pad (columns provably zero)
        vals = st.pop("v_pad").view(t, h, qk)[..., :v].reshape(t, h * v)
        vals = vals.contiguous()
        attn_out = torch.empty(t, h * v, dtype=torch.bfloat16,
                               device=st["q_full"].device)
        lse = torch.empty(h, t, dtype=torch.float32, device=attn_out.device)
        K.dsa_sparse_attn_fwd(
            kctx, st.pop("q_full"), st.pop("k_full"), vals,
            st.get("shared_idx", None) if "shared_idx" in st
            else st["aux_temp"]["dsa_idx"], attn_out, lse,
            n_heads=h, head_dim=qk, seq_bounds=_seq_bounds(st["seg"]), v_head_dim=v,
        )
        del vals
        if a is not None:
            a["lse"].copy_(lse.reshape(a["lse"].shape))
            a["attn_out"].copy_(attn_out)
        st.update(attn_out=attn_out, lse=lse)

    STAGES = (
        Dsv3DenseBlockFwd.MLA_STAGES[0],                    # attn_norm
        ("mla_q", _stage_mla_q.__func__, ("q_a", "rstd_qa")),
        ("mla_kv", _stage_mla_kv.__func__, ("kv_a", "rstd_kva")),
        ("dsa_select", _stage_dsa_select.__func__, (), "aux_temp"),
        ("dsa_attn", _stage_dsa_attn.__func__, ("lse", "attn_out")),
        Dsv3DenseBlockFwd.MLA_STAGES[4],                    # resid1_norm2
    ) + Dsv3DenseBlockFwd.STAGES[5:]                        # dense FFN tail

    MLA_STAGES = STAGES[:6]


@dataclass(frozen=True)
class Dsv32DenseBlockRecompute(Dsv32DenseBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Dsv32DenseBlockBwd(Dsv32AuxTempState, Dsv32ProfileFill, Dsv3DenseBlockBwd):
    dims: Dsv32Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_dense_activation_layout(self.dims)

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum, aux_temp=None):
        # merge the M views into the saved-state dict: downstream reads
        # (a["dsa_idx"], a["route_*"]) work unchanged — `a` is "the saved
        # state", now composed from the ctx + metadata objects
        if aux_temp:
            a = {**a, **aux_temp["aux_temp"]}
        super()._backward(kctx, dy, a, x, w, dx_out, dw, accum)

    def _indexer_kl_bwd(self, kctx, d, a, x, w, acc, h1, q_lora_n,
                        q_full, k_full, idx, lse, bounds, pos,
                        bits_by_seq=None):
        """The indexer's KL training path (detached seam). Extracted so
        the train_indexer=False ablation skips it wholesale."""
        # linear-triple conversion pending (exemplar: llama3)
        K = self.kernels
        t = d.tokens
        h, qk, rope = d.n_heads, d.qk_head_dim, d.qk_rope_dim
        q_idx, k_idx, wts = _indexer_inputs(kctx, K, d, h1, q_lora_n, w, pos)
        dq_idx = torch.empty_like(q_idx)
        dk_idx = torch.empty_like(k_idx)
        dwts = torch.empty_like(wts)
        hi_, di = d.index_n_heads, d.index_head_dim
        dense = idx is None  # warm-up: KL over the FULL causal prefix
        for si, (lo, hi) in enumerate(bounds):
            length = hi - lo
            # rebuild indexer scores for this sequence (causal -inf
            # outside the prefix is built into the score op)
            iscores = torch.empty(length, length, dtype=torch.float32,
                                  device=x.device)
            K.dsa_index_scores(
                kctx, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi], iscores,
                n_heads=hi_, head_dim=di, seq_bounds=((0, length),),
            )
            if dense:
                seq_bits = [_causal_bits(length, x.device)]
                sig = torch.softmax(iscores, dim=-1)  # causal via -inf
                rows = torch.arange(length, device=x.device).unsqueeze(1)
                cols = torch.arange(length, device=x.device).unsqueeze(0)
                live = cols <= rows
            else:
                seq_bits = (None if bits_by_seq is None
                            else [bits_by_seq[si]])
                # mask from saved selection (scatter + causal — pad-safe)
                m = torch.full((length, length), float("-inf"),
                               device=x.device)
                m.scatter_(-1, (idx[lo:hi].long() - lo).clamp_(0, length - 1),
                           0.0)
                rows = torch.arange(length, device=x.device).unsqueeze(1)
                cols = torch.arange(length, device=x.device).unsqueeze(0)
                m.masked_fill_(cols > rows, float("-inf"))
                live = m == 0
                sig = torch.softmax(iscores + m, dim=-1)
            # target p: head-summed attention probs from the saved lse —
            # fused kernel (flash tiling, all heads inside a tile); in
            # dense mode the "selection" is the full causal prefix
            p = torch.empty(length, length, device=x.device)
            K.dsa_probs_sum(
                kctx, q_full, k_full,
                idx if idx is not None else torch.zeros(
                    d.tokens, 1, dtype=torch.int32, device=x.device),
                lse, p,
                n_heads=h, head_dim=qk, seq_bounds=((lo, hi),),
                bits_by_seq=seq_bits,
            )
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-20)
            lv = a.get("_loss_view")
            if lv is not None:
                # warm-up reported objective: KL(p || sigma) on the live
                # set, accumulated across contributors into loss_{s}_{r}
                kl = (p * (p.clamp_min(1e-20).log()
                           - sig.clamp_min(1e-20).log())) \
                    .masked_fill(~live, 0.0).sum()
                if a.get("_loss_create"):
                    lv.copy_(kl.reshape(1))
                    a["_loss_create"] = False
                else:
                    lv.add_(kl.reshape(1))
            d_scores = (sig - p).masked_fill(~live, 0.0)
            K.dsa_index_bwd(
                kctx, d_scores, q_idx[lo:hi], k_idx[lo:hi], wts[lo:hi],
                dq_idx[lo:hi], dk_idx[lo:hi], dwts[lo:hi],
                n_heads=hi_, head_dim=di, seq_bounds=((0, length),),
            )
            del p, d_scores
        # chain to the four indexer weights (inputs detached) — rope_bwd
        # runs IN PLACE on the assembled grads (rope-first: col_base=0)
        K.rope_bwd(kctx, dq_idx, dq_idx, pos, hi_, rope, d.rope_base,
                   row_stride=hi_ * di, head_stride=di, col_base=0)
        if acc.wanted("w_idx_q"):
            acc("w_idx_q", q_lora_n.T @ dq_idx)
        del dq_idx
        K.rope_bwd(kctx, dk_idx, dk_idx, pos, 1, rope, d.rope_base,
                   row_stride=di, head_stride=di, col_base=0)
        dk_post_ln = dk_idx
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
        if acc.wanted("w_idx_k"):
            acc("w_idx_k", h1.T @ dk_pre.to(torch.bfloat16))
        del k_pre, mu, xc, var, rstd, xhat, g, dk_post_ln, dk_idx, dk_pre
        if acc.wanted("w_idx_w"):
            acc("w_idx_w", (h1.float().T @ (dwts * (hi_ ** -0.5) * (di ** -0.5))))
        del dwts, q_idx, k_idx, wts


    def _attn_bwd(self, kctx, dh_mid, a, x, w, acc, norm_bwd, dx_out) -> None:
        # linear-triple conversion pending (exemplar: llama3)
        d = self.dims
        K = self.kernels
        t = d.tokens
        h, nope, rope, v = d.n_heads, d.qk_nope_dim, d.qk_rope_dim, d.v_head_dim
        qk, kvl = d.qk_head_dim, d.kv_lora_rank
        seg = a["_seg"]
        bounds = _seq_bounds(seg)

        d_attn_v = (dh_mid @ w["wo"].T).contiguous()
        if acc.wanted("wo"):
            acc("wo", a["attn_out"].T @ dh_mid)

        pos = seg.positions          # always varlen; run_args prologue
        q_lora_n, q_full = _mla_expand_q(kctx, K, d, a["q_a"], a["rstd_qa"], w, pos)
        latent_n, k_full, vals = _mla_expand_kv(
            kctx, K, d, a["kv_a"], a["rstd_kva"], w, pos,
        )
        vals = vals.reshape(t, h * v).contiguous()

        lse = a["lse"].reshape(h, t)
        idx = a["dsa_idx"]
        # selection bitmask packed ONCE per sequence, shared by the sparse
        # backward and the KL target kernel
        bits_by_seq = _bits_for_bounds(idx, bounds, x.device)
        dq = torch.empty_like(q_full)
        dk = torch.empty_like(k_full)
        dv = torch.empty_like(vals)
        K.dsa_sparse_attn_bwd(
            kctx, d_attn_v, q_full, k_full, vals, idx, lse,
            dq, dk, dv, n_heads=h, head_dim=qk, seq_bounds=bounds,
            out=a["attn_out"], bits_by_seq=bits_by_seq, v_head_dim=v,
        )
        del d_attn_v

        # ---- indexer KL injection (detached seam: touches ONLY idx weights)
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        # train_indexer=False: the freeze policy already removed the idx
        # fields from dW/O (nothing to zero, nothing to step) — only the
        # KL computation itself is skipped here
        if d.train_indexer:
            self._indexer_kl_bwd(kctx, d, a, x, w, acc, h1, q_lora_n,
                                 q_full, k_full, idx, lse, bounds, pos,
                                 bits_by_seq=bits_by_seq)

        # ---- main-path chains (identical to dsv3 from here) ---------------
        dk3 = dk.view(t, h, qk)
        dkvb = torch.cat([dk3[..., :nope], dv.view(t, h, v)], dim=-1).reshape(
            t, h * (nope + v)).contiguous()
        del dv
        dk_rope_sum = dk3[..., nope:].sum(dim=1).contiguous()
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

        K.rope_bwd(kctx, dq, dq, pos, h, rope, d.rope_base,
                   row_stride=h * qk, head_stride=qk, col_base=nope)
        dq_pre_m = dq
        del dq
        if acc.wanted("w_q_b"):
            acc("w_q_b", q_lora_n.T @ dq_pre_m)
        dq_lora_n = dq_pre_m @ w["w_q_b"].T
        del dq_pre_m, q_lora_n
        dq_lora, d_q_norm = norm_bwd(dq_lora_n, a["q_a"], a["rstd_qa"],
                                     w["q_a_norm_w"])
        del dq_lora_n
        acc("q_a_norm_w", d_q_norm)

        if acc.wanted("w_q_a"):
            acc("w_q_a", h1.T @ dq_lora)
        if acc.wanted("w_kv_a"):
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
    AUX_TEMP_KIND = "moe"
    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_moe_activation_layout(self.dims)

    STAGES = Dsv32DenseBlockFwd.MLA_STAGES + MOE_SHARED_NOGATE_STAGES


@dataclass(frozen=True)
class Dsv32MoeBlockRecompute(Dsv32MoeBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Dsv32MoeBlockBwd(Dsv32DenseBlockBwd):
    AUX_TEMP_KIND = "moe"
    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_moe_activation_layout(self.dims)

    def _mlp_bwd(self, kctx, dy, a, w, dw, accum, acc, norm_bwd):
        return moe_mlp_tail_bwd(
            kctx, self.kernels, self.dims, dy, a, w, dw, accum, acc, norm_bwd,
            resid_field=self.MLP_RESID_FIELD,
        )



class _WarmupKLMixin:
    """Dense warm-up backward: the paper trains ONLY the indexer (its
    signal is L_I alone; CE is monitoring), so ALL main wgrads and the
    dgrad chain are SKIPPED — dW views are zeroed on the create round
    (the grammar is unchanged: dW/dy objects exist; the frozen optimizer
    no-ops on them) and dy_prev is zeroed. Then the FULL-PREFIX indexer
    KL injection runs (report formula 3), re-deriving the MLA expansions
    it needs from the saved context."""

    def launch(self, ctx) -> None:
        # warm-up bwd contract: inputs (A, x, W[, M..., dM..., loss]);
        # no dy in, no dy out (the chain carries no gradients — the
        # objective is layer-local). dW create/accumulate and the loss
        # accumulator are discovered by id, never position.
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            a = self.cl.views(self._in(ctx, 0))
            x = torch_view(self._in(ctx, 1), (d.tokens, d.d_model),
                           torch.bfloat16)
            w = self.wl_for(ctx.task).views(self._in(ctx, 2))
            dw = None
            accum = False
            for m in ctx.task.mutates:
                if m.startswith("dW_"):
                    dw = self.gl_for(ctx.task).views(ctx.mutates[m])
                    accum = True
            if dw is None:
                for j, o in enumerate(ctx.task.outputs):
                    if o.id.startswith("dW_"):
                        dw = self.gl_for(ctx.task).views(self._out(ctx, j))
            aux_temp = self._aux_temp_state(ctx) or {}
            lv, lcreate = None, False
            for j, o in enumerate(ctx.task.outputs):
                if o.id.startswith("loss_"):
                    lv, lcreate = torch_view(self._out(ctx, j), (1,),
                                             torch.float32), True
            for m in ctx.task.mutates:
                if m.startswith("loss_"):
                    lv = torch_view(ctx.mutates[m], (1,), torch.float32)
            aux_temp["_loss_view"] = lv
            aux_temp["_loss_create"] = lcreate
            a = {**a, "_seg": self._attn_meta(ctx)}
            self._backward(kctx, None, a, x, w, None, dw, accum,
                           aux_temp=aux_temp)

    def _backward(self, kctx, dy, a, x, w, dx_out, dw, accum, aux_temp=None):
        if aux_temp:
            a = {**a, **aux_temp.get("aux_temp", {})}
            for k in ("_dm_view", "_dm_create", "_kl_n",
                      "_loss_view", "_loss_create"):
                if k in aux_temp:
                    a[k] = aux_temp[k]
        d = self.dims
        K = self.kernels
        # frozen fields have NO dW storage (policy-filtered grad layout);
        # followers carry no dW object at all — dw arrives None there.
        # (warm-up + train_indexer=False is rejected at resolver build.)
        acc = (self._acc_fn(dw, accum) if dw
               else (lambda name, value: None))
        t = d.tokens
        seg = a["_seg"]
        bounds = _seq_bounds(seg)
        pos = seg.positions          # always varlen; run_args prologue
        h1 = torch.empty_like(x)
        K.rmsnorm_apply(kctx, x, a["rstd_attn"], w["attn_norm_w"], h1)
        q_lora_n, q_full = _mla_expand_q(kctx, K, d, a["q_a"], a["rstd_qa"],
                                         w, pos)
        _latent, k_full, _vals = _mla_expand_kv(
            kctx, K, d, a["kv_a"], a["rstd_kva"], w, pos,
        )
        del _latent, _vals
        lse = a["lse"].reshape(d.n_heads, t)
        self._indexer_kl_bwd(kctx, d, a, x, w, acc, h1, q_lora_n,
                             q_full, k_full, None, lse, bounds, pos)


@dataclass(frozen=True)
class Dsv32WarmupDenseBlockFwd(Dsv32AuxTempState, Dsv3DenseBlockFwd):
    """Dense warm-up forward = dsv3's flash path verbatim (no selection,
    no dsa_idx ctx); only the layouts widen for the indexer weights."""

    dims: Dsv32Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_dense_activation_layout(self.dims)


@dataclass(frozen=True)
class Dsv32WarmupDenseBlockRecompute(Dsv32WarmupDenseBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Dsv32WarmupDenseBlockBwd(Dsv32AuxTempState, _WarmupKLMixin, Dsv3DenseBlockBwd):
    dims: Dsv32Dims = None  # type: ignore[assignment]
    _indexer_kl_bwd = Dsv32DenseBlockBwd._indexer_kl_bwd

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_dense_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_dense_activation_layout(self.dims)


@dataclass(frozen=True)
class Dsv32WarmupMoeBlockFwd(Dsv32AuxTempState, Dsv32ProfileFill, Dsv3MoeBlockFwd):
    AUX_TEMP_KIND = "moe"
    dims: Dsv32Dims = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_moe_activation_layout(self.dims)


@dataclass(frozen=True)
class Dsv32WarmupMoeBlockRecompute(Dsv32WarmupMoeBlockFwd, BlockRecompute):
    pass


@dataclass(frozen=True)
class Dsv32WarmupMoeBlockBwd(Dsv32AuxTempState, Dsv32ProfileFill, _WarmupKLMixin, Dsv3MoeBlockBwd):
    AUX_TEMP_KIND = "moe"
    dims: Dsv32Dims = None  # type: ignore[assignment]
    _indexer_kl_bwd = Dsv32DenseBlockBwd._indexer_kl_bwd

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        return dsv32_moe_weight_layout(self.dims, layer=layer)

    @property
    def cl(self) -> PackedLayout:
        return dsv32_moe_activation_layout(self.dims)


_IDX_FIELDS = ("w_idx_q", "w_idx_k", "idx_k_ln_w", "idx_k_ln_b", "w_idx_w")


def build_dsv32_resolver(
    dims: Dsv32Dims,
    hyper: AdamWHyper = AdamWHyper(),
    kernels: KernelSet | None = None,
):
    kernels = kernels if kernels is not None else resolve_kernels()
    from functools import partial

    if not dims.sparse_mode and not dims.train_indexer:
        raise ValueError(
            "dense warm-up trains ONLY the indexer; train_indexer=False "
            "in dense mode would train nothing"
        )
    def _opt_layout(d, task, size):
        layer = AdamWStep.layer_of(task)
        if d.kinds[layer] == "dense":
            return dsv32_dense_weight_layout(d, layer=layer), None
        return dsv32_moe_weight_layout(d, layer=layer), None

    if dims.sparse_mode:
        blocks = {
            "dsadense_fwd": Dsv32DenseBlockFwd(dims, kernels),
            "dsadense_recompute": Dsv32DenseBlockRecompute(dims, kernels),
            "dsadense_bwd": Dsv32DenseBlockBwd(dims, kernels),
            "dsamoe_fwd": Dsv32MoeBlockFwd(dims, kernels),
            "dsamoe_recompute": Dsv32MoeBlockRecompute(dims, kernels),
            "dsamoe_bwd": Dsv32MoeBlockBwd(dims, kernels),
        }
        opt_embed = AdamWStep(dims, kernels, hyper, kind="embed")
        opt_head = AdamWStep(dims, kernels, hyper, kind="head")
    else:
        blocks = {
            "dsadense_fwd": Dsv32WarmupDenseBlockFwd(dims, kernels),
            "dsadense_recompute": Dsv32WarmupDenseBlockRecompute(dims, kernels),
            "dsadense_bwd": Dsv32WarmupDenseBlockBwd(dims, kernels),
            "dsamoe_fwd": Dsv32WarmupMoeBlockFwd(dims, kernels),
            "dsamoe_recompute": Dsv32WarmupMoeBlockRecompute(dims, kernels),
            "dsamoe_bwd": Dsv32WarmupMoeBlockBwd(dims, kernels),
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
