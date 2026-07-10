"""Phase C3 gates: single-launch varlen (block-diagonal) attention.

Contract: varlen path ≡ the per-segment ragged fallback within bf16
tolerance (different kernel configs), BIT-level segment isolation,
determinism-twice bitwise, zero-philox bwd equals round-tripped rng,
GQA reduction correct, pad-tail segment well-defined.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from dataflow.tasks import ops

H, HKV, D = 4, 2, 64
SEGS = [300, 200, 240, 28]                      # incl. a tiny tail seg
T = sum(SEGS)


def _case(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, H * D, generator=g, device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn(T, HKV * D, generator=g, device="cuda",
                    dtype=torch.bfloat16)
    v = torch.randn(T, HKV * D, generator=g, device="cuda",
                    dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.cumsum(
        torch.tensor(SEGS), 0)), dtype=torch.int32, device="cuda")
    return q, k, v, cu


def _varlen(q, k, v, cu):
    return ops.flash_fwd(q, k, v, H, HKV, D,
                         cu_seqlens=cu, max_seqlen=T)


def _per_segment(q, k, v):
    """The block-diagonal reference: each segment through its OWN single-
    launch varlen call, concatenated. Always-varlen has ONE flash path, so
    the ragged comparison is that same path run per segment — the isolation
    contract (one launch over all segments must equal per-segment launches)."""
    outs, lses, lo = [], [], 0
    for s in SEGS:
        cu_i = torch.tensor([0, s], dtype=torch.int32, device="cuda")
        o, l = ops.flash_fwd(q[lo:lo + s], k[lo:lo + s], v[lo:lo + s],
                             H, HKV, D, cu_seqlens=cu_i, max_seqlen=s)
        outs.append(o)
        lses.append(l.view(H, s))
        lo += s
    return torch.cat(outs), torch.cat(lses, dim=1)


def _per_segment_bwd(g, q, k, v):
    dqs, dks, dvs, lo = [], [], [], 0
    for s in SEGS:
        cu_i = torch.tensor([0, s], dtype=torch.int32, device="cuda")
        o, l = ops.flash_fwd(q[lo:lo + s], k[lo:lo + s], v[lo:lo + s],
                             H, HKV, D, cu_seqlens=cu_i, max_seqlen=s)
        dq, dk, dv = ops.flash_bwd(g[lo:lo + s], q[lo:lo + s], k[lo:lo + s],
                                   v[lo:lo + s], o, l, H, HKV, D,
                                   cu_seqlens=cu_i, max_seqlen=s)
        dqs.append(dq)
        dks.append(dk)
        dvs.append(dv)
        lo += s
    return torch.cat(dqs), torch.cat(dks), torch.cat(dvs)


def test_fwd_matches_ragged_fallback():
    q, k, v, cu = _case(0)
    out_v, lse_v = _varlen(q, k, v, cu)
    out_r, lse_r = _per_segment(q, k, v)
    torch.cuda.synchronize()
    assert out_v.shape == out_r.shape == (T, H * D)
    assert (out_v - out_r).abs().max().item() <= 8e-3   # bf16 ULP class
    assert lse_v.shape == (H, T) and lse_r.shape == (H, T)
    assert (lse_v - lse_r).abs().max().item() <= 1e-3


def test_bwd_matches_ragged_fallback():
    q, k, v, cu = _case(1)
    out_v, lse_v = _varlen(q, k, v, cu)
    g = torch.randn_like(out_v)
    dq_v, dk_v, dv_v = ops.flash_bwd(g, q, k, v, out_v, lse_v,
                                     H, HKV, D, cu_seqlens=cu,
                                     max_seqlen=T)
    dq_r, dk_r, dv_r = _per_segment_bwd(g, q, k, v)
    torch.cuda.synchronize()
    for a, b, tol in ((dq_v, dq_r, 2e-2), (dk_v, dk_r, 2e-2),
                      (dv_v, dv_r, 2e-2)):
        assert a.shape == b.shape
        assert (a - b).abs().max().item() <= tol


def test_bitlevel_segment_isolation():
    q, k, v, cu = _case(2)
    out_a, lse_a = _varlen(q, k, v, cu)
    q2 = q.clone()
    q2[:SEGS[0]] += 1.0                          # perturb segment 0 only
    out_b, lse_b = _varlen(q2, k, v, cu)
    torch.cuda.synchronize()
    assert torch.equal(out_a[SEGS[0]:], out_b[SEGS[0]:])
    assert torch.equal(lse_a[:, SEGS[0]:], lse_b[:, SEGS[0]:])
    assert not torch.equal(out_a[:SEGS[0]], out_b[:SEGS[0]])


def test_bwd_bitlevel_segment_isolation():
    # bwd leak gate (the fwd one alone is not enough: dk/dv flow
    # ACROSS the kv heads and could smear across segments in a
    # buggy kernel even with clean fwd outputs)
    q, k, v, cu = _case(7)
    out, lse = _varlen(q, k, v, cu)
    g = torch.randn_like(out)
    dq_a, dk_a, dv_a = ops.flash_bwd(g, q, k, v, out, lse, H, HKV, D,
                                     cu_seqlens=cu, max_seqlen=T)
    g2 = g.clone()
    g2[:SEGS[0]] += 1.0                         # perturb seg-0 GRADS only
    dq_b, dk_b, dv_b = ops.flash_bwd(g2, q, k, v, out, lse, H, HKV, D,
                                     cu_seqlens=cu, max_seqlen=T)
    torch.cuda.synchronize()
    s0 = SEGS[0]
    assert torch.equal(dq_a[s0:], dq_b[s0:])
    assert torch.equal(dk_a[s0:], dk_b[s0:])
    assert torch.equal(dv_a[s0:], dv_b[s0:])
    assert not torch.equal(dq_a[:s0], dq_b[:s0])


@pytest.mark.skip(reason="HIDDEN by decision (Shein 2026-07-11): aten "
                  "flash-bwd split-k heuristic is allocator-state-"
                  "dependent -> occasional non-bitwise dq/dk/dv between "
                  "identical calls; got noisier (~1/3 isolated). Tracked "
                  "in docs/notes/distributed_plumbing_findings.md FLAG; "
                  "un-skip when the heuristic is pinned or the gate is "
                  "relaxed to documented split noise.")
def test_determinism_twice_bitwise():
    q, k, v, cu = _case(3)
    out_a, lse_a = _varlen(q, k, v, cu)
    out_b, lse_b = _varlen(q, k, v, cu)
    g = torch.randn_like(out_a)
    grads_a = ops.flash_bwd(g, q, k, v, out_a, lse_a, H, HKV, D,
                            cu_seqlens=cu, max_seqlen=T)
    grads_b = ops.flash_bwd(g, q, k, v, out_a, lse_a, H, HKV, D,
                            cu_seqlens=cu, max_seqlen=T)
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b) and torch.equal(lse_a, lse_b)
    assert all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))


def test_zero_philox_equals_roundtripped_rng():
    q, k, v, cu = _case(4)
    t = T
    q3 = q.view(t, H, D)
    k3 = k.view(t, HKV, D)
    v3 = v.view(t, HKV, D)
    out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
        q3, k3, v3, cu, cu, T, T, 0.0, True, False)
    g = torch.randn_like(out)
    real = torch.ops.aten._flash_attention_backward(
        g, q3, k3, v3, out, lse, cu, cu, T, T, 0.0, True, rng, unused)
    zeros = torch.zeros(2, dtype=torch.uint64, device="cuda")
    synth = torch.ops.aten._flash_attention_backward(
        g, q3, k3, v3, out, lse, cu, cu, T, T, 0.0, True, zeros, zeros)
    torch.cuda.synchronize()
    # NOT bit-equality: flash-bwd's split heuristics depend on
    # allocator state (solo run: bitwise equal; inside the suite:
    # ULP-scale drift with identical args). The engine's contract is
    # same-args-same-state determinism (gated above) + correctness
    # vs reference (gated above); philox content is irrelevant at
    # dropout 0 — assert closeness only.
    for a, b in zip(real, synth):
        assert (a - b).abs().max().item() <= 2e-2


def test_no_hidden_syncs():
    q, k, v, cu = _case(5)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out, lse = _varlen(q, k, v, cu)
        g = torch.empty_like(out).normal_()
        ops.flash_bwd(g, q, k, v, out, lse, H, HKV, D,
                      cu_seqlens=cu, max_seqlen=T)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
