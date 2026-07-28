"""The varlen flash-attention surface, isolated with its dispatch.

Blocks and model code never import this module — they call the ops
surface, which re-exports it. What lives here is exactly the pair of
single-launch varlen kernels every family attends with, plus the
implementation dispatch: at import, if a flash-attention provider is
installed (the ``flash_attn_3`` wheel exposing
``flash_attn_interface``), it is activated into torch's dispatcher,
and the SAME aten calls below run the provider's kernels on hardware
they support — torch gates per call by compute capability, so on any
other device they fall back to the native kernels with identical
call shapes and semantics. No provider installed means the native
path, silently. ``active_flash_impl()`` names what actually runs, so
cost profiles can key on it — timings measured under one
implementation must never be reused under another.
"""
from __future__ import annotations

import torch


def resolve_flash_impl() -> str:
    """Activate the best available flash implementation, once, and
    return its name. Never raises: any provider absence or activation
    failure means the native kernels, which are always present.

    Activation is attempted only on hardware the provider actually
    serves — FA3 on Hopper (sm90), FA4 on datacenter Blackwell
    (exactly sm100; consumer Blackwell sm120 is served by neither):
    torch would otherwise ACCEPT the activation and fall back to
    native per call, and a cost profile keyed "FA3" would be
    measuring native kernels. On a box without CUDA the capability
    probe raises and we land on native."""
    import torch.nn.attention as attention_registry

    current = attention_registry.current_flash_attention_impl()
    if current is not None:
        return current
    capability_impls = {(9, 0): "FA3", (10, 0): "FA4"}
    try:
        impl = capability_impls.get(torch.cuda.get_device_capability(0))
        if impl is None:
            return "native"
        attention_registry.activate_flash_attention_impl(impl)
    except Exception:
        return "native"
    return attention_registry.current_flash_attention_impl() or "native"


ACTIVE_FLASH_IMPL = resolve_flash_impl()


def active_flash_impl() -> str:
    """The flash implementation the dispatcher runs ("FA3" or
    "native"). Part of any cost-profile cache key: kernels differ,
    so their timings do."""
    return ACTIVE_FLASH_IMPL


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
