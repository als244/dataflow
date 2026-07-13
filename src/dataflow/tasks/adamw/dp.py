"""Plain data-parallel optimizer step (and the standalone run).

Every rank holds full replicas and a partial gradient from its slice
of the batch: allreduce dW on the group stream (event edges both
ways, enqueue-only), then run the full per-field update — elementwise
math over one shared sum, so every rank lands the same trajectory
bitwise. With no group bound, the same artifact runs standalone
(valid exactly because plain-allreduce DP is rank-complete); that
standalone path is what the fleet warm-up executes.
"""
from __future__ import annotations

import torch

from ..interop import TORCH_DTYPE_BY_NAME, torch_view
from .update import update_fields


def launch(block, ctx, es, kctx, wl, gl, ol, ns,
           w_buf, g_buf, o_buf, gh) -> None:
    if gh is not None:
        grads_ready = torch.cuda.Event()
        grads_ready.record(es)
        gh.stream.wait_event(grads_ready)
        dtypes = {f.dtype for f in gl.fields}
        if len(dtypes) == 1:
            # contiguous grad layout, uniform dtype: ONE fused
            # exchange for the whole layer's grads (wire-floor sized)
            # instead of a round trip per field — elementwise sums,
            # so bitwise-identical results
            total = 0
            for f in gl.fields:
                n = 1
                for s in f.shape:
                    n *= s
                dt_f = TORCH_DTYPE_BY_NAME[f.dtype]
                end = f.offset_bytes + n * dt_f.itemsize
                if end > total:
                    total = end
            dt_all = TORCH_DTYPE_BY_NAME[gl.fields[0].dtype]
            fused = torch_view(g_buf, (total // dt_all.itemsize,),
                               dt_all)
            gh.allreduce(fused)
        else:
            for f in gl.fields:
                gview = torch_view(g_buf, f.shape,
                                   TORCH_DTYPE_BY_NAME[f.dtype],
                                   offset_bytes=f.offset_bytes)
                gh.allreduce(gview)
        summed = torch.cuda.Event()
        summed.record(gh.stream)
        es.wait_event(summed)
    update_fields(block, ctx, kctx, wl, gl, ol, ns,
                  w_buf, g_buf, o_buf)
