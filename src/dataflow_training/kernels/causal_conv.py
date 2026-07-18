"""causal_conv1d_silu family: depthwise causal conv1d + silu (the DeltaNet
qkv short-convolution), token-major.

Op signatures (t tokens, D = conv_dim, W = kernel width; all contiguous):
- ``causal_conv1d_silu_fwd(kctx, x, w, out, cu_seqlens)``: x/out (t, D),
  w (D, W); ``cu_seqlens`` (int32 boundaries tensor) or None resets the
  causal window at packed-sequence boundaries.
- ``causal_conv1d_silu_bwd(kctx, x, dy, w, dx_out, dw_out, cu_seqlens)``:
  silu recomputed internally from x.

Default wraps fla's fused Triton kernels (layout-native token-major).
The Dao causal-conv1d package (channel-major) measured a bandwidth TIE in
the standalone A/B, so its registry
entry is deferred until a profile pulls for it — its raw CUDA binding
takes 12 undocumented positional args and earns nothing today. The eager
fallback runs the sequence-aware reference (autograd for bwd).
"""
from __future__ import annotations

import torch

from ..blocks import ops
from .registry import internal, register


def _fla_available(caps: dict) -> bool:
    if not (caps.get("cuda") and caps.get("triton")):
        return False
    try:
        import fla.modules.conv  # noqa: F401

        return True
    except Exception:
        return False


def _fwd_hint(x: torch.Tensor, *a) -> int:
    return x.numel() * x.element_size()


def _bwd_hint(x: torch.Tensor, *a) -> int:
    return 2 * x.numel() * x.element_size() + 64 * x.shape[-1] * 4


def _fla_fwd(kctx, x, w, out, cu_seqlens):
    import fla.modules.conv.triton.ops as fops

    post = fops.causal_conv1d_fwd(
        x.unsqueeze(0), w, None, None, activation="silu", cu_seqlens=cu_seqlens,
    )
    post = post[0] if isinstance(post, tuple) else post
    out.copy_(post.squeeze(0) if post.dim() == 3 else post)


def _fla_bwd(kctx, x, dy, w, dx_out, dw_out, cu_seqlens):
    import fla.modules.conv.triton.ops as fops

    dx, dw, _db, _dres, _dinit = fops.causal_conv1d_bwd(
        x.unsqueeze(0), dy.unsqueeze(0), None,
        weight=w, bias=None, residual=None, initial_state=None,
        activation="silu", cu_seqlens=cu_seqlens,
    )
    dx_out.copy_(dx.squeeze(0) if dx.dim() == 3 else dx)
    dw_out.copy_(dw.to(dw_out.dtype))


register(
    "causal_conv1d_silu_fwd", "fla-triton", deterministic=True,
    allocates="torch", workspace=internal(_fwd_hint),
    requires=_fla_available, priority=10, fn=_fla_fwd,
)
register(
    "causal_conv1d_silu_bwd", "fla-triton", deterministic=True,
    allocates="torch", workspace=internal(_bwd_hint),
    requires=_fla_available, priority=10, fn=_fla_bwd,
)


def _segments_of(cu_seqlens):
    """Rebuild the round's Segments from the conv kernel's cu_seqlens (device
    int tensor); None = one sequence. Eager fallback only (tiny-scale
    correctness path), so the device->host read is acceptable."""
    if cu_seqlens is None:
        return None
    return ops.Segments.from_boundaries([int(v) for v in cu_seqlens.tolist()])


def _eager_fwd(kctx, x, w, out, cu_seqlens):
    out.copy_(ops.causal_conv1d_silu_reference(x, w, segments=_segments_of(cu_seqlens)))


def _eager_bwd(kctx, x, dy, w, dx_out, dw_out, cu_seqlens):
    with torch.enable_grad():
        x_l = x.detach().requires_grad_()
        w_l = w.detach().requires_grad_()
        y = ops.causal_conv1d_silu_reference(x_l, w_l, segments=_segments_of(cu_seqlens))
    dx, dw = torch.autograd.grad(y, (x_l, w_l), grad_outputs=dy)
    dx_out.copy_(dx)
    dw_out.copy_(dw.to(dw_out.dtype))


register(
    "causal_conv1d_silu_fwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_fwd_hint), priority=0, fn=_eager_fwd,
)
register(
    "causal_conv1d_silu_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_bwd_hint), priority=0, fn=_eager_bwd,
)
