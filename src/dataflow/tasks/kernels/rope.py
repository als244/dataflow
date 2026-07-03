"""rope (llama rotate-half): fused Triton (default) + eager fallback.

Op signatures:
- ``rope_fwd(kctx, x, out, seq_len, n_heads, head_dim, base)``
- ``rope_bwd(kctx, dx, out, seq_len, n_heads, head_dim, base)``

x/out: (tokens, n_heads*head_dim) contiguous, tokens = batch x seq_len
(positions restart every seq_len rows). The eager form is the single worst
temporary offender in ops.py: ~4 UNCHUNKED full-tensor fp32 intermediates
plus an on-device cos/sin table rebuilt per call. The fused kernel computes
angle = pos * base^(-2i/hd) in registers (libdevice pow/cos/sin), processes
rotate-half pairs (i, i+hd/2) together, and touches memory once each way.

bwd is the rotation transpose (rotate by -theta): identical kernel with the
sin sign flipped, matching ops.rope_bwd exactly.
"""
from __future__ import annotations

import torch

from .. import ops
from .registry import internal, none, register


def _eager_hint(x: torch.Tensor, *a) -> int:
    return 4 * x.numel() * 4  # ~4 full fp32 temporaries (unchunked!)


register(
    "rope_fwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_hint), priority=0,
    fn=lambda kctx, x, out, s, h, hd, base: out.copy_(ops.rope_fwd(x, s, h, hd, base)),
)
register(
    "rope_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_hint), priority=0,
    fn=lambda kctx, dx, out, s, h, hd, base: out.copy_(ops.rope_bwd(dx, s, h, hd, base)),
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _rope_kernel(
        x_ptr, out_ptr, n_pairs, seq_len, width, head_dim,
        base, SIN_SIGN: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # one lane per rotate-half PAIR: pair p -> row, head, i (i < hd/2)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_pairs
        half = head_dim // 2
        pairs_per_row = width // 2
        row = offs // pairs_per_row
        rem = offs % pairs_per_row
        head = rem // half
        i = rem % half

        pos = (row % seq_len).to(tl.float32)
        # angle = pos * base^(-2i/hd); exp2/log2 form of pow for libdevice
        expo = -2.0 * i.to(tl.float32) / head_dim
        freq = tl.exp2(expo * tl.log2(base))
        angle = pos * freq
        cos = tl.cos(angle)
        sin = tl.sin(angle) * SIN_SIGN

        lo_idx = row * width + head * head_dim + i
        hi_idx = lo_idx + half
        x_lo = tl.load(x_ptr + lo_idx, mask=mask, other=0).to(tl.float32)
        x_hi = tl.load(x_ptr + hi_idx, mask=mask, other=0).to(tl.float32)
        # out = x*cos + rotate_half(x)*sin, rotate_half = (-x_hi, x_lo)
        out_lo = x_lo * cos - x_hi * sin
        out_hi = x_hi * cos + x_lo * sin
        tl.store(out_ptr + lo_idx, out_lo.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_ptr + hi_idx, out_hi.to(out_ptr.dtype.element_ty), mask=mask)

    _BLOCK = 512

    def _launch(x, out, seq_len, n_heads, head_dim, base, sign):
        assert x.is_contiguous() and out.is_contiguous() and x.shape == out.shape
        width = n_heads * head_dim
        n_pairs = x.shape[0] * (width // 2)
        _rope_kernel[(triton.cdiv(n_pairs, _BLOCK),)](
            x, out, n_pairs, seq_len, width, head_dim,
            float(base), SIN_SIGN=sign, BLOCK=_BLOCK,
        )

    @register("rope_fwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _fwd(kctx, x, out, seq_len, n_heads, head_dim, base):
        _launch(x, out, seq_len, n_heads, head_dim, base, 1.0)

    @register("rope_bwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _bwd(kctx, dx, out, seq_len, n_heads, head_dim, base):
        _launch(dx, out, seq_len, n_heads, head_dim, base, -1.0)
