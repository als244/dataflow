"""Fused cross-entropy loss fwd+bwd: Triton (default) + eager fallback.

Op signature:
- ``ce_loss_fwd_bwd(kctx, logits, targets, loss_out, dlogits_out,
  total_rows=None)``: loss_out fp32 scalar = sum(nll) / total_rows,
  dlogits scaled 1/total_rows; total_rows defaults to logits' row count
  (= plain mean CE). A CHUNKED caller (the fused head) passes the FULL
  token count so per-chunk partial means sum to the true mean.

The eager form materializes ~2 x chunk x vocab fp32 (~1 GB scratch at llama
vocab) and makes ~5 passes over the model's largest tensor. The fused kernel
is one program per row: pass 1 streams the row once with an ONLINE
max/sum-exp (running rescale, the flash-attention trick) to get lse; pass 2
streams it again writing dlogits = (softmax - onehot)/n directly in bf16.
Two reads + one write total; fp32 only in registers. Per-row nll lands in a
small fp32 buffer reduced by a fixed-shape torch ``sum`` (deterministic; no
atomics).
"""
from __future__ import annotations

import torch

from ..blocks import ops
from .registry import internal, register


def _eager_hint(logits: torch.Tensor, *a) -> int:
    v = logits.shape[-1]
    return 2 * min(logits.shape[0], ops.CE_CHUNK_ROWS) * v * 4


register(
    "ce_loss_fwd_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_eager_hint), priority=0,
    fn=lambda kctx, logits, targets, loss, dlogits, total_rows=None:
        ops.ce_loss_fwd_bwd(logits, targets, loss, dlogits, total_rows=total_rows),
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _ce_kernel(
        logits_ptr, targets_ptr, dlogits_ptr, nll_ptr,
        total_rows, vocab, BLOCK: tl.constexpr,
    ):
        # int64 row index: row * vocab overflows int32 once rows x vocab
        # crosses 2^31 elements (qwen3.5's 248,320 vocab hits it at 8,650
        # rows — a bs16 s1k round; bs8 sat 5% under the line)
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        base = logits_ptr + row * vocab

        # pass 1: online max + rescaled sum of exp (single streaming read)
        m = -float("inf")
        s = 0.0
        for off in range(0, vocab, BLOCK):
            mask = off + cols < vocab
            lv = tl.load(base + off + cols, mask=mask, other=-float("inf")).to(tl.float32)
            blk_max = tl.max(lv)
            m_new = tl.maximum(m, blk_max)
            s = s * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(lv - m_new), 0.0))
            m = m_new
        lse = m + tl.log(s)

        target = tl.load(targets_ptr + row).to(tl.int64)
        # ignore-index (< 0): zero nll + zero dlogits row (packing pads)
        valid = target >= 0
        t_safe = tl.where(valid, target, 0)
        x_t = tl.load(base + t_safe).to(tl.float32)
        tl.store(nll_ptr + row, tl.where(valid, lse - x_t, 0.0))

        # pass 2: dlogits = (exp(x - lse) - onehot) / total_rows
        inv_n = 1.0 / total_rows
        for off in range(0, vocab, BLOCK):
            mask = off + cols < vocab
            lv = tl.load(base + off + cols, mask=mask, other=0).to(tl.float32)
            soft = tl.exp(lv - lse)
            onehot = ((off + cols).to(tl.int64) == t_safe).to(tl.float32)
            g = tl.where(valid, (soft - onehot) * inv_n, 0.0)
            tl.store(dlogits_ptr + row * vocab + off + cols,
                     g.to(dlogits_ptr.dtype.element_ty), mask=mask)

    _BLOCK = 4096

    def _hint(logits: torch.Tensor, *a) -> int:
        return logits.shape[0] * 4  # per-row nll buffer

    @register("ce_loss_fwd_bwd", "triton", deterministic=True, allocates="torch",
              workspace=internal(_hint),
              requires=lambda c: c.get("triton"), priority=10)
    def _ce(kctx, logits, targets, loss_out, dlogits_out, total_rows=None):
        n_rows, vocab = logits.shape
        total = int(total_rows) if total_rows is not None else n_rows
        assert logits.is_contiguous() and dlogits_out.is_contiguous()
        nll = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        _ce_kernel[(n_rows,)](
            logits, targets.int(), dlogits_out, nll, total, vocab, BLOCK=_BLOCK,
        )
        loss_out.copy_((nll.sum() / total).reshape(loss_out.shape))
