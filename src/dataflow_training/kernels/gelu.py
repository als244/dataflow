"""gelu family (gpt2): the tanh approximation, aten-backed.

The forward/backward pair is exactly what autograd invokes on the twin
side (aten gelu / gelu_backward with approximate="tanh", fp32 opmath for
bf16 inputs) — parity by construction. Elementwise, no host reads.

Op signatures (all tensors contiguous, same shape):
- ``gelu_fwd_out(kctx, x, out)``
- ``gelu_bwd(kctx, dy, x, dx_out)``
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import internal, register


def _hint(x: torch.Tensor, *a) -> int:
    return x.numel() * x.element_size()


register(
    "gelu_fwd_out", "eager", deterministic=True, allocates="torch",
    workspace=internal(_hint), priority=0,
    fn=lambda kctx, x, out: out.copy_(F.gelu(x, approximate="tanh")),
)
register(
    "gelu_bwd", "eager", deterministic=True, allocates="torch",
    workspace=internal(_hint), priority=0,
    fn=lambda kctx, dy, x, dx_out: dx_out.copy_(
        torch.ops.aten.gelu_backward(dy, x, approximate="tanh")),
)
