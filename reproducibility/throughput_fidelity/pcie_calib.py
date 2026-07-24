#!/usr/bin/env python
"""Measure real pinned host<->device bandwidth on this GPU, so the sim's
slow<->fast (pcie_gbs) seed reflects della's actual link instead of the
generic 55 GB/s default. Prints `PCIE_GBS=<n>` (min of sustained H2D/D2H at
the largest size — the conservative seed for transfer-bound cells).
"""
import time

import torch


def bw_gbs(direction, mb, iters=20):
    n = mb * 1024 * 1024 // 4  # fp32 elements
    dev = torch.device("cuda")
    if direction == "h2d":
        src = torch.empty(n, pin_memory=True)
        dst = torch.empty(n, device=dev)
    else:
        src = torch.empty(n, device=dev)
        dst = torch.empty(n, pin_memory=True)
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return (n * 4) / 1e9 / dt  # decimal GB/s


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    peak = {}
    for direction in ("h2d", "d2h"):
        for mb in (16, 64, 256, 1024):
            g = bw_gbs(direction, mb)
            print(f"  {direction} {mb:>5} MB: {g:6.1f} GB/s")
            peak[direction] = max(peak.get(direction, 0.0), g)
    seed = min(peak["h2d"], peak["d2h"])
    print(f"peak H2D={peak['h2d']:.1f}  peak D2H={peak['d2h']:.1f} GB/s")
    print(f"PCIE_GBS={seed:.1f}")


if __name__ == "__main__":
    main()
