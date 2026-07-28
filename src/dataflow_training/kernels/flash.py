"""flash attention (varlen, causal): native aten + provider generations.

flash is a named kernel like any other: implementations register here
and ``resolve_kernels`` pins one per device at runtime. ``aten`` wraps
ops.flash_fwd/flash_bwd (torch's native varlen flash — every arch).
``fa3`` calls the flash_attn_3 provider's raw entry points DIRECTLY,
per call — no torch-dispatcher activation, no process-global state —
so ``DATAFLOW_KERNELS=eager`` and per-op overrides keep exact meaning
and the native impl stays truly native in the same process. FA3
serves EXACTLY sm90; a future FA4 provider serves EXACTLY sm100
(consumer sm120 is served by neither generation) — mirror the fa3
registration with its own capability gate when a wheel exists.

The fa3 backward passes deterministic=True: bitwise repeatability
(probed twice-equal on H100) costs ~20% of backward over the atomics
path, and the parity instruments depend on it. Probed vs native on
H100 (GQA 32/8/128, 2x8192): out 1.0e-03 / grads <= 6.2e-03 (bf16
reduction-order class), lse layout identical (n_heads, t) fp32 at
1e-07, fwd 2.08x, bwd 1.64x.

Optional output buffers: fwd ``out=``/``lse_out=``, bwd ``dq_out=``/
``dk_out=``/``dv_out=``. The fa3 forward hands ``out`` straight to the
provider's out_ parameter (no copy); aten has no out variant, so its
impl computes fresh and copies — the same bytes the call sites used
to move themselves.

Shared knobs: ``causal`` (default True) and ``softmax_scale`` (default
head_dim ** -0.5) — both backends honor them. Provider-only knobs
(windowing, softcap, descale, pack_gqa, num_splits) are deliberately
NOT part of the op contract; widen it when a workload needs one.
"""
from __future__ import annotations

import torch

from ..blocks import ops
from .registry import internal, register


def fa3_ready(caps: dict) -> bool:
    return bool(caps.get("flash3"))


def flash_fwd_hint(q, k, v, n_heads, n_kv_heads, head_dim,
                   cu_seqlens, max_seqlen, out=None, lse_out=None,
                   causal=True, softmax_scale=None) -> int:
    return q.numel() * 2 + n_heads * q.shape[0] * 4


def flash_bwd_hint(d_attn, q, k, v, attn_out, lse, n_heads, n_kv_heads,
                   head_dim, cu_seqlens, max_seqlen,
                   dq_out=None, dk_out=None, dv_out=None,
                   causal=True, softmax_scale=None) -> int:
    return (q.numel() + k.numel() + v.numel()) * 2


def aten_flash_fwd(kctx, q, k, v, n_heads, n_kv_heads, head_dim,
                   cu_seqlens, max_seqlen, out=None, lse_out=None,
                   causal=True, softmax_scale=None):
    return ops.flash_fwd(
        q, k, v, n_heads, n_kv_heads, head_dim, cu_seqlens, max_seqlen,
        causal=causal, softmax_scale=softmax_scale, out=out,
        lse_out=lse_out)


def aten_flash_bwd(kctx, d_attn, q, k, v, attn_out, lse, n_heads,
                   n_kv_heads, head_dim, cu_seqlens, max_seqlen,
                   dq_out=None, dk_out=None, dv_out=None,
                   causal=True, softmax_scale=None):
    return ops.flash_bwd(
        d_attn, q, k, v, attn_out, lse, n_heads, n_kv_heads, head_dim,
        cu_seqlens, max_seqlen, causal=causal, softmax_scale=softmax_scale,
        dq_out=dq_out, dk_out=dk_out, dv_out=dv_out)


def fa3_flash_fwd(kctx, q, k, v, n_heads, n_kv_heads, head_dim,
                  cu_seqlens, max_seqlen, out=None, lse_out=None,
                  causal=True, softmax_scale=None):
    import flash_attn_interface as provider

    t = q.shape[0]
    mq = int(max_seqlen)
    out3 = None if out is None else out.view(t, n_heads, head_dim)
    # Positional schema of _flash_attn_forward (probed against the wheel):
    # (q, k, v, k_new, v_new, qv, out_, cu_seqlens_q, cu_seqlens_k,
    #  cu_seqlens_k_new, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k,
    #  page_table, kv_batch_idx, leftpad_k, rotary_cos, rotary_sin,
    #  seqlens_rotary, q_descale, k_descale, v_descale, softmax_scale,
    #  causal, window_size_left, window_size_right, attention_chunk,
    #  softcap, rotary_interleaved, scheduler_metadata, num_splits,
    #  pack_gqa, sm_margin) -> (out, lse, ...)
    res = provider._flash_attn_forward(
        q.view(t, n_heads, head_dim),
        k.view(t, n_kv_heads, head_dim),
        v.view(t, n_kv_heads, head_dim),
        None, None, None, out3,
        cu_seqlens, cu_seqlens, None, None, None, mq, mq,
        None, None, None, None, None, None, None, None, None,
        (float(head_dim) ** -0.5 if softmax_scale is None
         else float(softmax_scale)),
        bool(causal), -1, -1, 0, 0.0, False, None, 1, None, 0)
    lse = res[1]
    if lse_out is not None:
        lse_out.copy_(lse)
        lse = lse_out
    if out is not None:
        return out, lse
    return res[0].reshape(t, n_heads * head_dim), lse


def fa3_flash_bwd(kctx, d_attn, q, k, v, attn_out, lse, n_heads,
                  n_kv_heads, head_dim, cu_seqlens, max_seqlen,
                  dq_out=None, dk_out=None, dv_out=None,
                  causal=True, softmax_scale=None):
    import flash_attn_interface as provider

    t = q.shape[0]
    mq = int(max_seqlen)
    q3 = q.view(t, n_heads, head_dim)
    k3 = k.view(t, n_kv_heads, head_dim)
    v3 = v.view(t, n_kv_heads, head_dim)
    dq3 = (torch.empty_like(q3) if dq_out is None
           else dq_out.view(t, n_heads, head_dim))
    dk3 = (torch.empty_like(k3) if dk_out is None
           else dk_out.view(t, n_kv_heads, head_dim))
    dv3 = (torch.empty_like(v3) if dv_out is None
           else dv_out.view(t, n_kv_heads, head_dim))
    # Positional schema of _flash_attn_backward (probed): (dout, q, k, v,
    # out, softmax_lse, cu_seqlens_q, cu_seqlens_k, sequed_q, sequed_k,
    # max_seqlen_q, max_seqlen_k, dq, dk, dv, softmax_scale, is_causal,
    # window_size_left, window_size_right, softcap, deterministic,
    # sm_margin); writes dq/dk/dv in place.
    provider._flash_attn_backward(
        d_attn.view(t, n_heads, head_dim), q3, k3, v3,
        attn_out.view(t, n_heads, head_dim), lse.contiguous(),
        cu_seqlens, cu_seqlens, None, None, mq, mq, dq3, dk3, dv3,
        (float(head_dim) ** -0.5 if softmax_scale is None
         else float(softmax_scale)),
        bool(causal), -1, -1, 0.0, True, 0)
    return ((dq3.reshape(t, n_heads * head_dim) if dq_out is None
             else dq_out),
            (dk3.reshape(t, n_kv_heads * head_dim) if dk_out is None
             else dk_out),
            (dv3.reshape(t, n_kv_heads * head_dim) if dv_out is None
             else dv_out))


register(
    "flash_fwd", "aten", deterministic=True, allocates="torch",
    workspace=internal(flash_fwd_hint), priority=0, fn=aten_flash_fwd,
)
register(
    "flash_bwd", "aten", deterministic=True, allocates="torch",
    workspace=internal(flash_bwd_hint), priority=0, fn=aten_flash_bwd,
)
register(
    "flash_fwd", "fa3", deterministic=True, allocates="torch",
    workspace=internal(flash_fwd_hint), priority=10, requires=fa3_ready,
    fn=fa3_flash_fwd,
)
register(
    "flash_bwd", "fa3", deterministic=True, allocates="torch",
    workspace=internal(flash_bwd_hint), priority=10, requires=fa3_ready,
    fn=fa3_flash_bwd,
)
