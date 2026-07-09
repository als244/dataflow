"""rope (llama rotate-half): fused Triton (default) + eager fallback.

Op signatures:
- ``rope_fwd(kctx, x, out, positions, n_heads, head_dim, base,
             *, row_stride=None, head_stride=None, col_base=0)``
- ``rope_bwd(...)`` — same, sin sign flipped (rotation transpose).
  (``positions``: (tokens,) int32 per-sequence indices — sequence structure
  is explicit; the round's ``Segments.positions`` device field)

Default (strides omitted): x/out (tokens, n_heads*head_dim) contiguous.
STRIDED/IN-PLACE mode (the torch.cat killer): pass the FULL assembled
tensor as both x and out with ``row_stride`` (elements per row),
``head_stride`` (elements per head) and ``col_base`` (offset of the rope
columns inside each head) — the kernel rotates the rope slice in place,
so the extract -> contiguous -> rope -> cat round-trips disappear.
In-place is safe by construction: each lane owns one rotate-half pair
exclusively and loads both elements before storing either. The eager form is the single worst
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


def _eager_apply(fn, kctx, x, out, pos, h, hd, base, *,
                 row_stride=None, head_stride=None, col_base=0):
    if row_stride is None:
        out.copy_(fn(x, pos, h, hd, base))
        return
    assert x.data_ptr() == out.data_ptr(), "strided mode is in-place"
    t_rows = x.shape[0]
    sl = torch.as_strided(
        x.view(-1), (t_rows, h, hd), (row_stride, head_stride, 1),
        storage_offset=x.storage_offset() + col_base,
    )
    sl.copy_(fn(sl.reshape(t_rows, h * hd).contiguous(), pos, h, hd, base)
             .view(t_rows, h, hd))


register(
    "rope_fwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_hint), priority=0,
    fn=lambda kctx, x, out, pos, h, hd, base, **kw: _eager_apply(
        ops.rope_fwd, kctx, x, out, pos, h, hd, base, **kw),
)
register(
    "rope_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_hint), priority=0,
    fn=lambda kctx, dx, out, pos, h, hd, base, **kw: _eager_apply(
        ops.rope_bwd, kctx, dx, out, pos, h, hd, base, **kw),
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _rope_kernel(
        x_ptr, out_ptr, positions_ptr, n_pairs, width, head_dim,
        row_stride, head_stride, col_base,
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

        pos = tl.load(positions_ptr + row, mask=mask, other=0).to(tl.float32)
        # angle = pos * base^(-2i/hd); exp2/log2 form of pow for libdevice
        expo = -2.0 * i.to(tl.float32) / head_dim
        freq = tl.exp2(expo * tl.log2(base))
        angle = pos * freq
        cos = tl.cos(angle)
        sin = tl.sin(angle) * SIN_SIGN

        lo_idx = row * row_stride + head * head_stride + col_base + i
        hi_idx = lo_idx + half
        x_lo = tl.load(x_ptr + lo_idx, mask=mask, other=0).to(tl.float32)
        x_hi = tl.load(x_ptr + hi_idx, mask=mask, other=0).to(tl.float32)
        # out = x*cos + rotate_half(x)*sin, rotate_half = (-x_hi, x_lo)
        out_lo = x_lo * cos - x_hi * sin
        out_hi = x_hi * cos + x_lo * sin
        tl.store(out_ptr + lo_idx, out_lo.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_ptr + hi_idx, out_hi.to(out_ptr.dtype.element_ty), mask=mask)

    _BLOCK = 512

    def _launch(x, out, positions, n_heads, head_dim, base, sign,
                row_stride=None, head_stride=None, col_base=0):
        strided = row_stride is not None
        if not strided:
            assert x.is_contiguous() and out.is_contiguous() and x.shape == out.shape
            row_stride = n_heads * head_dim
            head_stride = head_dim
        else:
            assert x.data_ptr() == out.data_ptr(), "strided mode is in-place"
            assert head_stride is not None
        assert positions.is_contiguous() and positions.shape[0] == x.shape[0]
        width = n_heads * head_dim
        n_pairs = x.shape[0] * (width // 2)
        _rope_kernel[(triton.cdiv(n_pairs, _BLOCK),)](
            x, out, positions, n_pairs, width, head_dim,
            row_stride, head_stride, col_base,
            float(base), SIN_SIGN=sign, BLOCK=_BLOCK,
        )

    @register("rope_fwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _fwd(kctx, x, out, positions, n_heads, head_dim, base, *,
             row_stride=None, head_stride=None, col_base=0):
        _launch(x, out, positions, n_heads, head_dim, base, 1.0,
                row_stride=row_stride, head_stride=head_stride,
                col_base=col_base)

    @register("rope_bwd", "triton", deterministic=True,
              workspace=none(), requires=lambda c: c.get("triton"), priority=10)
    def _bwd(kctx, dx, out, positions, n_heads, head_dim, base, *,
             row_stride=None, head_stride=None, col_base=0):
        _launch(dx, out, positions, n_heads, head_dim, base, -1.0,
                row_stride=row_stride, head_stride=head_stride,
                col_base=col_base)
