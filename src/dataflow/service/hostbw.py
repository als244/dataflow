"""Boot-time host-bandwidth probe: measures the memory lanes every
ideal-time estimate needs — pinned host memcpy, the bf16 in-place add
(the collective reduce op, exactly as the comm runs it), and D2H/H2D
over PCIe. Results are saved on the server and exposed via
engine_status, next to the peer plane's per-link wire probes — so
idle-gap math uses MEASURED numbers end to end.
"""
from __future__ import annotations

import time


def measure_host_bw(mib: int = 256) -> dict:
    """Payload GB/s per lane (best of 3). ~0.3 s at the default size;
    mib=0 disables (returns {})."""
    if mib <= 0:
        return {}
    import torch

    n = mib << 20
    out = {}
    a = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    b = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    a.fill_(1)
    b.fill_(2)
    out["host_copy_gbs"] = best_gbs(CopyOp(b, a), n)
    x = a.view(torch.bfloat16)
    y = b.view(torch.bfloat16)
    x.add_(y)                       # warm the kernel + pages
    out["host_bf16_add_gbs"] = best_gbs(AddOp(x, y), n)
    if torch.cuda.is_available():
        d = torch.empty(n, dtype=torch.uint8, device="cuda")
        d.fill_(3)
        torch.cuda.synchronize()
        out["h2d_gbs"] = best_gbs(CudaCopyOp(d, a), n)
        out["d2h_gbs"] = best_gbs(CudaCopyOp(a, d), n)
    return out


def best_gbs(op, payload_bytes: int, reps: int = 3) -> float:
    best = 0.0
    for rep in range(reps):
        t0 = time.monotonic()
        op()
        dt = max(time.monotonic() - t0, 1e-9)
        best = max(best, payload_bytes / dt / 1e9)
    return round(best, 2)


class CopyOp:
    def __init__(self, dst, src):
        self.dst = dst
        self.src = src

    def __call__(self):
        self.dst.copy_(self.src)


class AddOp:
    def __init__(self, acc, other):
        self.acc = acc
        self.other = other

    def __call__(self):
        self.acc.add_(self.other)


class CudaCopyOp:
    def __init__(self, dst, src):
        self.dst = dst
        self.src = src

    def __call__(self):
        import torch

        self.dst.copy_(self.src, non_blocking=True)
        torch.cuda.synchronize()
