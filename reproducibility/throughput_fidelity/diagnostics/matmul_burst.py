#!/usr/bin/env python
"""Pure bf16 matmul burst — the 'what does full-power sustained compute look
like' reference for the block_bwd power question."""
import sys
import time

import torch

n = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
end = time.perf_counter() + secs
k = 0
while time.perf_counter() < end:
    for _ in range(50):
        a = a @ b
        a = a * 0.5 + 0.1
    torch.cuda.synchronize()
    k += 50
print(f"matmul {n}x{n}: {k} gemms in {secs}s")
