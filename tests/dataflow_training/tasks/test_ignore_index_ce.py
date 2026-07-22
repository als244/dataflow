"""Ignore-index CE (packing pads) — both impls.

Padded round (tail IGNORE_INDEX targets, valid-count normalization)
must equal the plain CE of the same real rows: loss AND per-row
dlogits; ignored rows contribute exactly zero gradient. Legacy
behavior (no negatives, default normalization) stays bit-identical.

Tests:
- test_padded_equals_unpadded: padded targets with valid-count normalization give bitwise-equal per-row dlogits, zero gradient on ignored rows, and loss matching plain CE of the real rows within reduction noise.
- test_no_ignore_rows_matches_torch_ce_and_rerun_is_bitwise: with no ignored rows the loss matches torch cross_entropy and a rerun is bitwise-identical in loss and dlogits.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason="needs CUDA")]

from dataflow_training.data.lpt_packing import IGNORE_INDEX
from dataflow_training.blocks import ops
from dataflow_training.kernels.registry import registered

V, T, VALID = 1024, 256, 199


def _case(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(T, V, generator=g, device="cuda",
                         dtype=torch.bfloat16)
    tgt = torch.randint(0, V, (T,), generator=g, device="cuda",
                        dtype=torch.int32)
    tgt_pad = tgt.clone()
    tgt_pad[VALID:] = IGNORE_INDEX
    return logits, tgt, tgt_pad


def _run(fn, logits, targets, total_rows):
    loss = torch.zeros(1, device="cuda", dtype=torch.float32)
    dlog = torch.empty_like(logits)
    fn(logits, targets, loss, dlog, total_rows=total_rows)
    torch.cuda.synchronize()
    return loss.item(), dlog


class _KCtx:  # minimal kernel ctx for direct registry entry calls
    device = "cuda"


def _impls():
    out = {"eager": lambda *a, **k: ops.ce_loss_fwd_bwd(*a, **k)}
    entries = registered("ce_loss_fwd_bwd")
    if "triton" in entries:
        fn = entries["triton"].fn
        out["triton"] = (lambda *a, **kw: fn(_KCtx(), *a, **kw))
    return out


@pytest.mark.parametrize("impl", ["eager", "triton"])
def test_padded_equals_unpadded(impl):
    fns = _impls()
    if impl not in fns:
        pytest.skip(f"{impl} unavailable")
    fn = fns[impl]
    logits, tgt, tgt_pad = _case(11)

    # padded grid, valid-count normalization
    loss_p, dlog_p = _run(fn, logits, tgt_pad, VALID)
    # plain CE over ONLY the real rows
    loss_u, dlog_u = _run(fn, logits[:VALID].contiguous(),
                          tgt[:VALID], VALID)

    # per-row grads are EXACT (shape-independent row math);
    # the scalar loss is a reduction over different tensor SIZES
    # (256-with-zeros vs 199) => different reduction trees => equal
    # only within reduction-order noise (~1e-7 rel; eager exhibits
    # it, triton happens to match bitwise). Contract pinned here.
    assert torch.equal(dlog_p[:VALID], dlog_u)
    assert torch.count_nonzero(dlog_p[VALID:]) == 0
    assert loss_p == pytest.approx(loss_u, rel=1e-6), (loss_p, loss_u)


@pytest.mark.parametrize("impl", ["eager", "triton"])
def test_no_ignore_rows_matches_torch_ce_and_rerun_is_bitwise(impl):
    fns = _impls()
    if impl not in fns:
        pytest.skip(f"{impl} unavailable")
    fn = fns[impl]
    logits, tgt, _ = _case(12)
    loss_a, dlog_a = _run(fn, logits, tgt, None)
    ref = torch.nn.functional.cross_entropy(
        logits.float(), tgt.long()).item()
    assert loss_a == pytest.approx(ref, abs=2e-3)
    loss_b, dlog_b = _run(fn, logits, tgt, None)
    assert loss_a == loss_b and torch.equal(dlog_a, dlog_b)
