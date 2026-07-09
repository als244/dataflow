"""Eager-torch op library for llama-family blocks.

Each op is a pair of plain functions over torch tensors:

- the *launch* form used by executables (writes into provided `out` tensors
  where torch supports it; anything torch self-allocates goes through the
  bounded scratch lane and is measured by the profiling harness), and
- a *reference* form — pure, autograd-able — used by the gradcheck helpers
  and by the composer-derived task references.

Numerics: parameters/activations bf16; reductions and normalization
statistics in fp32 (matching the golden reference model bit-for-bit in
structure, so parity tolerances stay tight).
"""
from __future__ import annotations

import math

import torch

# --- rmsnorm -------------------------------------------------------------------

RMS_EPS = 1e-5


def rmsnorm_fwd(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, rstd_out: torch.Tensor) -> None:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1) + RMS_EPS)
    rstd_out.copy_(rstd)
    out.copy_((xf * rstd.unsqueeze(-1)).to(x.dtype) * w)


def rmsnorm_apply(x: torch.Tensor, rstd: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Recompute the normalized output from a saved rstd (cheap, exact)."""
    return (x.float() * rstd.unsqueeze(-1)).to(x.dtype) * w


def rmsnorm_bwd(
    dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (dx, dw). Row-chunked (per-row math is unchanged; only the dw
    accumulation order differs, at fp32 noise level)."""
    dx = torch.empty_like(x)
    dw_acc = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
    wf = w.float()
    t = x.shape[0]
    for lo in range(0, t, ROWWISE_CHUNK):
        hi = min(lo + ROWWISE_CHUNK, t)
        xf = x[lo:hi].float()
        dyf = dy[lo:hi].float()
        xhat = xf * rstd[lo:hi].unsqueeze(-1)
        dxhat = dyf * wf
        dw_acc += (dyf * xhat).sum(0)
        dx[lo:hi] = (
            rstd[lo:hi].unsqueeze(-1) * (dxhat - xhat * (dxhat * xhat).mean(-1, keepdim=True))
        ).to(x.dtype)
    return dx, dw_acc.to(w.dtype)


def rmsnorm_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    # output stays at the ACTIVATION dtype (kernel semantics) even when the
    # weight is stored wider (dtype-policy fp32 norm weights)
    return ((xf * rstd).to(x.dtype) * w).to(x.dtype)


def rmsnorm_noweight(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's FINAL norm before the LM head (weightless here; llama
    initializes that weight to 1 — adequate for runtime work, documented).
    Returns (normalized bf16, rstd fp32)."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1) + RMS_EPS)
    return (xf * rstd.unsqueeze(-1)).to(x.dtype), rstd


def rmsnorm_noweight_reference(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (xf * rstd).to(x.dtype)


# --- rope (llama rotate-half) ----------------------------------------------------

# --- sequence structure -----------------------------------------------------
#
# ``seq`` arguments across the ops are polymorphic (varlen-first design,
# varlen-first convention): an int means uniform sequences of that
# length (tokens // seq of them — the historical fast paths), a tuple is the
# explicit per-sequence length list (ragged packing), None means one
# sequence spanning all tokens.


from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Segments:
    """How one round's tokens split into sequences — the SINGLE varlen
    descriptor shared by packing, engine blocks, and reference models.

    ``lengths`` (host) are the per-sequence token counts (sum == tokens)
    and fully define the geometry. The device tensors the varlen flash
    kernels and rope need are carried as FIELDS, materialized ONCE by
    ``.on(device)``:
      - ``cu``        (n_seq + 1,) int32 cumulative segment boundaries
      - ``positions`` (tokens,)    int32 per-sequence rope indices
    ``.on`` is called exactly once per round in the engine's run prologue
    (and once per golden forward); every stage/op downstream then reads
    ``seg.cu`` / ``seg.positions`` as plain attributes. Nothing rebuilds a
    device tensor from host data mid-round — that would be a hidden
    host->device sync (the aten-hidden-syncs discipline). ``cu`` /
    ``positions`` are excluded from equality/hash (identity is ``lengths``).

    Replaces the old seq_spec (int | tuple) + the seq_lens_of /
    seq_bounds_of / positions_for / attn_meta free-function family.
    """
    lengths: tuple[int, ...]
    cu: torch.Tensor | None = field(default=None, compare=False)
    positions: torch.Tensor | None = field(default=None, compare=False)

    @classmethod
    def uniform(cls, seq_len: int, batch: int) -> "Segments":
        return cls((int(seq_len),) * int(batch))

    @classmethod
    def from_boundaries(cls, cu) -> "Segments":
        """[0, b1, ..., tokens] cumulative boundaries -> Segments (host)."""
        cu = [int(x) for x in cu]
        if len(cu) < 2 or cu[0] != 0 or any(b < a for a, b in zip(cu, cu[1:])):
            raise ValueError(f"cumulative boundaries from 0 required, got {cu}")
        return cls(tuple(b - a for a, b in zip(cu, cu[1:])))

    @classmethod
    def of_dims(cls, d) -> "Segments":
        """The round's segmentation implied by a dims config (host):
        explicit ``seq_lens`` when ragged, else ``batch`` uniform
        ``seq_len`` sequences. Materialize with ``.on(device)``."""
        sl = getattr(d, "seq_lens", None)
        if sl is not None:
            return cls(tuple(int(n) for n in sl))
        return cls.uniform(d.seq_len, d.tokens // d.seq_len)

    @property
    def tokens(self) -> int:
        return sum(self.lengths)

    @property
    def max_len(self) -> int:
        return max(self.lengths)

    @property
    def bounds(self) -> list[tuple[int, int]]:
        out, lo = [], 0
        for n in self.lengths:
            out.append((lo, lo + n))
            lo += n
        return out

    @property
    def boundaries(self) -> list[int]:
        """[0, b1, ..., tokens] cumulative host boundaries — the inverse of
        ``from_boundaries`` and the form run_args['seq_lens'] carries."""
        out, acc = [0], 0
        for n in self.lengths:
            acc += n
            out.append(acc)
        return out

    @property
    def materialized(self) -> bool:
        return self.cu is not None

    def on(self, device) -> "Segments":
        """Materialize ``cu`` / ``positions`` on ``device`` ONCE and return a
        Segments carrying them as fields. Pinned staging + non_blocking copy
        — never a pageable H2D (the hidden-sync rule). Idempotent when the
        tensors already live on ``device``."""
        if self.cu is not None and self.cu.device == torch.device(device):
            return self
        b = [0]
        for n in self.lengths:
            b.append(b[-1] + n)
        cu_host = torch.tensor(b, dtype=torch.int32).pin_memory()
        if self.lengths:
            pos_host = torch.cat(
                [torch.arange(n, dtype=torch.int32) for n in self.lengths]
            ).pin_memory()
        else:
            pos_host = torch.empty(0, dtype=torch.int32).pin_memory()
        return replace(
            self,
            cu=cu_host.to(device, non_blocking=True),
            positions=pos_host.to(device, non_blocking=True),
        )


def _rope_cos_sin(seq_len: int, head_dim: int, base: float, device, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rope_cos_sin_pos(positions: torch.Tensor, head_dim: int, base: float, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_fwd(x: torch.Tensor, positions: torch.Tensor, n_heads: int, head_dim: int, base: float) -> torch.Tensor:
    """x: (tokens, n_heads*head_dim); positions: (tokens,) int PER-SEQUENCE
    indices (``Segments.positions``) — sequence structure is fully explicit."""
    t = x.shape[0]
    cos, sin = _rope_cos_sin_pos(positions, head_dim, base)
    xh = x.view(t, n_heads, head_dim).float()
    out = xh * cos[:, None, :] + _rotate_half(xh) * sin[:, None, :]
    return out.to(x.dtype).view(t, n_heads * head_dim)


def rope_bwd(dx: torch.Tensor, positions: torch.Tensor, n_heads: int, head_dim: int, base: float) -> torch.Tensor:
    """Gradient through the rotation = rotation by -theta (its transpose)."""
    t = dx.shape[0]
    cos, sin = _rope_cos_sin_pos(positions, head_dim, base)
    dh = dx.view(t, n_heads, head_dim).float()
    out = dh * cos[:, None, :] - _rotate_half(dh) * sin[:, None, :]
    return out.to(dx.dtype).view(t, n_heads * head_dim)


# --- flash attention (aten low-level fwd/bwd split) ------------------------------

def flash_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
    cu_seqlens: torch.Tensor, max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-launch VARLEN flash-attention forward — the ONLY path (a
    uniform batch is just equal-length segments). q: (t, d); k, v: (t, kv);
    causal PER SEGMENT. ``cu_seqlens`` is the device int32 cumulative
    boundary vector (``Segments.cu``, incl. the final total); ``max_seqlen``
    is the STATIC host int flash grid (``Segments.max_len`` — NEVER derived
    from device data, the hidden-sync rule). GQA native (no kv-head
    expansion). Returns (attn_out (t, d), lse (n_heads, t) ragged layout).
    Probed on this box: bit-clean segment isolation, deterministic-twice,
    sync-audit clean."""
    t = q.shape[0]
    mq = int(max_seqlen)
    out, lse, _rng, _unused, _ = torch.ops.aten._flash_attention_forward(
        q.view(t, n_heads, head_dim),
        k.view(t, n_kv_heads, head_dim),
        v.view(t, n_kv_heads, head_dim),
        cu_seqlens, cu_seqlens, mq, mq, 0.0, True, False)
    return out.reshape(t, n_heads * head_dim), lse


def flash_bwd(
    d_attn: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attn_out: torch.Tensor, lse: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
    cu_seqlens: torch.Tensor, max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-launch VARLEN flash-attention backward — the ONLY path,
    mirroring flash_fwd. Returns (dq (t,d), dk (t,kv), dv (t,kv)); GQA head
    grads come back reduced natively. ``cu_seqlens`` = ``Segments.cu``,
    ``max_seqlen`` = ``Segments.max_len`` (static host int). lse is
    (n_heads, t). philox zeros are valid (dropout 0 — gate-verified equal to
    round-tripped rng_state). ``.contiguous()`` on lse is LOAD-BEARING: the
    aten flash-bwd kernel reads it assuming contiguous rows (the fla
    contiguity lesson, aten edition — silent garbage grads otherwise)."""
    t = q.shape[0]
    mq = int(max_seqlen)
    philox = torch.zeros(2, dtype=torch.uint64, device=q.device)
    dq3, dk3, dv3 = torch.ops.aten._flash_attention_backward(
        d_attn.view(t, n_heads, head_dim),
        q.view(t, n_heads, head_dim),
        k.view(t, n_kv_heads, head_dim),
        v.view(t, n_kv_heads, head_dim),
        attn_out.view(t, n_heads, head_dim),
        lse.contiguous(), cu_seqlens, cu_seqlens, mq, mq,
        0.0, True, philox, philox)
    return (dq3.reshape(t, n_heads * head_dim),
            dk3.reshape(t, n_kv_heads * head_dim),
            dv3.reshape(t, n_kv_heads * head_dim))


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
    segments: "Segments | None" = None,
) -> torch.Tensor:
    """Golden causal attention over ``segments`` (block-diagonal — each
    segment is an independent causal sequence). Uniform batches take the
    batched-SDPA fast path; ragged packs recurse per segment. ``segments``
    uses only host structure (lengths/bounds), so an unmaterialized Segments
    is fine; None = one sequence spanning all tokens."""
    t = q.shape[0]
    if segments is None:
        segments = Segments.uniform(t, 1)
    lens = segments.lengths
    rep = n_heads // n_kv_heads
    if len(set(lens)) > 1:
        return torch.cat([
            attention_reference(q[lo:hi], k[lo:hi], v[lo:hi],
                                n_heads, n_kv_heads, head_dim,
                                Segments.uniform(hi - lo, 1))
            for lo, hi in segments.bounds
        ])
    s = lens[0]
    b = t // s
    q4 = q.view(b, s, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(b, s, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    v4 = v.view(b, s, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
    return out.transpose(1, 2).reshape(t, n_heads * head_dim)


# --- swiglu ---------------------------------------------------------------------

def swiglu_fwd(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """Pure form (autograd-able) — golden references compose this."""
    return torch.nn.functional.silu(x1.float()).to(x1.dtype) * x3


ROWWISE_CHUNK = 2048  # bounds fp32 temporaries of t x d_ff ops to ~chunk x d_ff


def swiglu_fwd_out(x1: torch.Tensor, x3: torch.Tensor, out: torch.Tensor) -> None:
    """Launch form: row-chunked into a preallocated output (an unchunked
    version holds several t x d_ff fp32 temporaries — GBs at batched scale)."""
    t = x1.shape[0]
    for lo in range(0, t, ROWWISE_CHUNK):
        hi = min(lo + ROWWISE_CHUNK, t)
        out[lo:hi] = torch.nn.functional.silu(x1[lo:hi].float()).to(x1.dtype) * x3[lo:hi]


def swiglu_bwd(
    ds: torch.Tensor, x1: torch.Tensor, x3: torch.Tensor,
    dx1_out: torch.Tensor | None = None, dx3_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-chunked backward (elementwise: chunking is numerics-identical)."""
    dx1 = dx1_out if dx1_out is not None else torch.empty_like(x1)
    dx3 = dx3_out if dx3_out is not None else torch.empty_like(x3)
    t = x1.shape[0]
    for lo in range(0, t, ROWWISE_CHUNK):
        hi = min(lo + ROWWISE_CHUNK, t)
        x1f = x1[lo:hi].float()
        sig = torch.sigmoid(x1f)
        dsf = ds[lo:hi].float()
        dx1[lo:hi] = (dsf * x3[lo:hi].float() * (sig * (1 + x1f * (1 - sig)))).to(x1.dtype)
        dx3[lo:hi] = (dsf * (x1f * sig)).to(x3.dtype)
    return dx1, dx3


# --- fused cross-entropy loss ------------------------------------------------------

CE_CHUNK_ROWS = 1024  # bounds fp32 softmax temporaries to ~2 x chunk x vocab


def ce_loss_fwd_bwd(
    logits: torch.Tensor, targets: torch.Tensor, loss_out: torch.Tensor, dlogits_out: torch.Tensor,
    *, total_rows: int | None = None,
) -> None:
    """CE over tokens; writes fp32 scalar loss and bf16 dlogits.

    ``total_rows`` is the normalization count (defaults to logits' rows =
    plain mean CE); a chunked caller passes the FULL token count so partial
    losses/grads sum to the true mean. Row-chunked internally: per-row math
    (logsumexp/softmax) is unchanged; only the final mean accumulates
    across chunks. Unchunked, the fp32 temporaries are ~2 x tokens x vocab
    x 4 bytes (4+ GB at llama vocab)."""
    t = logits.shape[0]
    total = int(total_rows) if total_rows is not None else t
    nll_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    tl = targets.long()
    for lo in range(0, t, CE_CHUNK_ROWS):
        hi = min(lo + CE_CHUNK_ROWS, t)
        lf = logits[lo:hi].float()
        lse = torch.logsumexp(lf, dim=-1, keepdim=True)
        tc = tl[lo:hi]
        # ignore-index (targets < 0, e.g. packing pads): zero fwd
        # contribution + zero dlogits row; mask-mul keeps reduction
        # order fixed (deterministic)
        valid = (tc >= 0)
        tc_safe = tc.clamp_min(0)
        nll_rows = lse.squeeze(-1) - lf.gather(
            1, tc_safe.unsqueeze(1)).squeeze(1)
        nll_sum += (nll_rows * valid.float()).sum()
        soft = torch.exp(lf - lse)
        soft.scatter_add_(
            1, tc_safe.unsqueeze(1),
            torch.full((hi - lo, 1), -1.0, device=logits.device, dtype=torch.float32),
        )
        soft *= valid.unsqueeze(1).float()
        dlogits_out[lo:hi].copy_((soft / total).to(dlogits_out.dtype))
    loss_out.copy_((nll_sum / total).reshape(loss_out.shape))


def ce_loss_reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.float(), targets.long())


# --- adamw ---------------------------------------------------------------------

ADAMW_CHUNK_ELEMS = 1 << 24  # 16.7M elems -> ~400MB of fp32 temporaries max


def adamw_step(
    w: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    *, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, step: int,
) -> None:
    """In-place AdamW on flat views. States bf16, math fp32 (the golden
    reference implements the identical update, incl. the bf16 state
    round-trip, so parity is exact in structure).

    Chunked: the update is elementwise, so processing the flat parameter in
    chunks is bit-identical while bounding fp32 temporaries (an unchunked
    embed/head update materializes ~6x param bytes — 12+ GB at vocab scale)."""
    n = w.numel()
    for lo in range(0, n, ADAMW_CHUNK_ELEMS):
        hi = min(lo + ADAMW_CHUNK_ELEMS, n)
        wc, gc, mc, vc = w[lo:hi], g[lo:hi], m[lo:hi], v[lo:hi]
        gf = gc.float()
        mf = mc.float().mul_(beta1).add_(gf, alpha=1 - beta1)
        vf = vc.float().mul_(beta2).addcmul_(gf, gf, value=1 - beta2)
        mc.copy_(mf.to(mc.dtype))
        vc.copy_(vf.to(vc.dtype))
        # bias-corrected using the ROUND-TRIPPED states (what the next step sees)
        mhat = mc.float() / (1 - beta1 ** step)
        vhat = vc.float() / (1 - beta2 ** step)
        wf = wc.float()
        wf -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * wf)
        wc.copy_(wf.to(wc.dtype))


# --- embedding ---------------------------------------------------------------------

def embed_fwd(tokens: torch.Tensor, w_embed: torch.Tensor, out: torch.Tensor) -> None:
    torch.index_select(w_embed, 0, tokens.int(), out=out)


def embed_bwd_accum(tokens: torch.Tensor, dy: torch.Tensor, dw_embed: torch.Tensor, *, zero_first: bool) -> None:
    """DETERMINISTIC embedding-gradient accumulation.

    ``index_add_`` on CUDA is float atomicAdd: with duplicate tokens the
    add ORDER varies run to run — a one-ulp W_embed lottery (measured
    ~1-in-5 fixed-seed pairs differing by one element; every family's
    bitwise-determinism gate was silently rolling these dice).

    Fix, sync-free and fixed-shape (no nonzero()): stable-sort tokens,
    fp32 cumsum over the sorted rows, per-position segment totals via
    cumsum-diff against each segment's start (cummax-propagated), then a
    single index_add_ where every position contributes — but non-
    segment-end positions contribute EXACT ZEROS, so the atomic order is
    irrelevant (x + 0.0 is exact regardless of order; each vocab row
    receives exactly one nonzero add)."""
    if zero_first:
        dw_embed.zero_()
    t = tokens.shape[0]
    tok = tokens.long()
    order = torch.argsort(tok, stable=True)
    st = tok[order]
    csum = dy[order].float().cumsum(0)                       # (t, d) fp32

    idx = torch.arange(t, device=tok.device)
    is_start = torch.ones(t, dtype=torch.bool, device=tok.device)
    is_start[1:] = st[1:] != st[:-1]
    is_end = torch.ones(t, dtype=torch.bool, device=tok.device)
    is_end[:-1] = st[1:] != st[:-1]
    seg_start = torch.cummax(torch.where(is_start, idx, idx.new_zeros(())), 0).values
    before = torch.where(
        (seg_start > 0).unsqueeze(1),
        csum[(seg_start - 1).clamp_min(0)],
        torch.zeros_like(csum[:1]),
    )
    totals = torch.where(is_end.unsqueeze(1), csum - before,
                         torch.zeros_like(csum))
    dw_embed.index_add_(0, st.int(), totals.to(dw_embed.dtype))


# --- qwen3.5 reference forms (DeltaNet + gated attention) ---------------------
#
# Pure, autograd-able torch: the math SPEC for the qwen35 family. The golden
# model composes these; ladder tests pin the fla kernels against them (the
# sequential delta-rule recurrence is additionally cross-checked against
# fla.ops.gated_delta_rule.naive at fp32 — spec vs spec).

L2NORM_EPS = 1e-6  # fla.modules.l2norm convention


def l2norm_reference(x: torch.Tensor) -> torch.Tensor:
    """Per-row (last-dim) L2 normalization, fla convention."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + L2NORM_EPS)
    return (xf * rstd).to(x.dtype)


def gated_rmsnorm_reference(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMSNormGated: silu(z) * rmsnorm(o) * w, all over the last dim
    (lin_v_head_dim). Matches the flextrain/HF gated-delta output norm."""
    of = o.float()
    rstd = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    zf = z.float()
    y = torch.nn.functional.silu(zf) * (of * rstd)
    return (y * w.float()).to(o.dtype)


def causal_conv1d_silu_reference(
    x: torch.Tensor, w: torch.Tensor, *, segments: "Segments | None" = None,
) -> torch.Tensor:
    """Depthwise causal conv1d + silu. x: (T, D) token-major; w: (D, W).
    ``segments`` resets the causal window at packed-sequence boundaries
    (positions 0..W-2 of every sequence see zero padding, never the previous
    sequence's tail); host structure only. None = one sequence over all
    tokens."""
    T, D = x.shape
    W = w.shape[-1]
    if segments is None:
        segments = Segments.uniform(T, 1)
    lens = segments.lengths
    if len(set(lens)) > 1:
        return torch.cat([
            causal_conv1d_silu_reference(
                x[lo:hi], w, segments=Segments.uniform(hi - lo, 1))
            for lo, hi in segments.bounds
        ])
    B = T // lens[0]
    xf = x.float().T.reshape(D, B, -1).transpose(0, 1)  # (B, D, seq)
    wf = w.float().unsqueeze(1)                         # (D, 1, W)
    y = torch.nn.functional.conv1d(
        torch.nn.functional.pad(xf, (W - 1, 0)), wf, groups=D,
    )
    y = y.transpose(0, 1).reshape(D, T).T
    return torch.nn.functional.silu(y).to(x.dtype)


def gated_delta_gate_reference(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Per-token decay log: g = -exp(A_log) * softplus(a + dt_bias), fp32."""
    return -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())


def gated_delta_rule_reference(
    q: torch.Tensor,      # (T, HK, K) post-l2norm
    k: torch.Tensor,      # (T, HK, K) post-l2norm
    v: torch.Tensor,      # (T, HV, V)
    beta: torch.Tensor,   # (T, HV)
    g: torch.Tensor,      # (T, HV) decay log, fp32
    *,
    scale: float | None = None,
    segments: "Segments | None" = None,
) -> torch.Tensor:
    """Sequential gated delta rule (the recurrence itself; fp32 state):

        S_t = S_{t-1} * exp(g_t)
        u_t = beta_t * (v_t - k_t^T S_{t-1..gated})
        S_t = S_t + k_t (x) u_t
        o_t = (scale * q_t)^T S_t

    GVA: q/k arrive with HK heads and are expanded so v-head i reads
    k-head i // (HV // HK) — same mapping fla's kernels apply internally.
    ``segments`` resets the recurrent state at packed-sequence boundaries
    (host structure only). None = one sequence over all tokens.
    """
    T, HK, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    rep = HV // HK
    qf = q.float().repeat_interleave(rep, dim=1)
    kf = k.float().repeat_interleave(rep, dim=1)
    vf = v.float()
    betaf = beta.float()
    gf = g.float()
    if scale is None:
        scale = K ** -0.5
    starts = {0} if segments is None else {lo for lo, _hi in segments.bounds}
    state = torch.zeros(HV, K, V, dtype=torch.float32, device=q.device)
    outs = []
    for t in range(T):
        if t in starts:
            state = torch.zeros(HV, K, V, dtype=torch.float32, device=q.device)
        state = state * gf[t].exp()[:, None, None]
        err = vf[t] - torch.einsum("hk,hkv->hv", kf[t], state)
        upd = betaf[t][:, None] * err
        state = state + kf[t][:, :, None] * upd[:, None, :]
        outs.append(torch.einsum("hk,hkv->hv", qf[t] * scale, state))
    return torch.stack(outs).to(v.dtype)


def partial_rope_reference(
    x: torch.Tensor, segments: "Segments | None", n_heads: int, head_dim: int,
    rot_dim: int, base: float,
) -> torch.Tensor:
    """Partial RoPE: rotate only the first rot_dim channels of each head
    (pair-interleaved, via the family rope reference); the rest pass through.
    ``segments`` supplies the per-sequence rope positions (the materialized
    ``segments.positions`` device field); None = one sequence over all
    tokens."""
    t = x.shape[0]
    xh = x.view(t, n_heads, head_dim)
    pos = (Segments.uniform(t, 1).on(x.device).positions
           if segments is None else segments.positions)
    rot = rope_fwd(xh[:, :, :rot_dim].reshape(t, n_heads * rot_dim), pos, n_heads, rot_dim, base)
    return torch.cat([rot.view(t, n_heads, rot_dim), xh[:, :, rot_dim:]], dim=-1).view(t, n_heads * head_dim)

