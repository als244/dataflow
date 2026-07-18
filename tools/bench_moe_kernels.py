"""Per-op MoE kernel bench: our registry impls vs flextrain's kernels.

Acceptance gate: each default impl within ~10% of its
flextrain counterpart at the target shapes, or understood/justified.
flextrain's kernel file is loaded DIRECTLY from refs/flextrain (it only
needs torch+triton — no flextrain package import), so this runs in the
dataflow env.

    python tools/bench_moe_kernels.py [--shape olmoe|qwen35moe|both]

Shapes (bs16 s1k round): t=16384, K=8 -> rows=131072, d=2048;
olmoe E=64 F=1024; qwen35moe E=256 F=512.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import time

import torch

from dataflow_training.kernels import KernelCtx, resolve_kernels

ROOT = pathlib.Path(__file__).resolve().parent.parent
FLEX_KERNELS = ROOT / "refs/flextrain/flextrain/ops/_kernels/moe.py"


def load_flex():
    spec = importlib.util.spec_from_file_location("flex_moe_kernels", FLEX_KERNELS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(fn, n=20, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3  # ms


def row(name, ours_ms, theirs_ms=None, note=""):
    if theirs_ms is None:
        print(f"  {name:<34} ours {ours_ms*1e3:9.1f} us   {note}")
        return
    ratio = ours_ms / theirs_ms if theirs_ms else float("inf")
    flag = "" if ratio <= 1.10 else "  <-- INVESTIGATE (>1.10x)"
    print(
        f"  {name:<34} ours {ours_ms*1e3:9.1f} us   flextrain {theirs_ms*1e3:9.1f} us"
        f"   ratio {ratio:5.2f}x{flag}{('  ' + note) if note else ''}"
    )


def run_shape(label: str, e: int, f: int, flex) -> None:
    t, k, d = 16384, 8, 2048
    rows = t * k
    print(f"\n== {label}: t={t} K={k} rows={rows} d={d} E={e} F={f} ==")
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    K = resolve_kernels()
    kctx = KernelCtx(0, None)

    logits = torch.randn(t, e, generator=g, device="cuda", dtype=torch.float32).bfloat16()
    route_w = torch.empty(t, k, dtype=torch.bfloat16, device="cuda")
    route_ids = torch.empty(t, k, dtype=torch.int32, device="cuda")
    ours = bench(lambda: K.moe_topk_softmax(
        kctx, logits, route_w, route_ids, top_k=k, mode="topk_then_softmax"))
    theirs = bench(lambda: flex.flextrain_fused_topk_softmax(
        logits, k, route_ids, route_w, mode="topk_then_softmax"))
    row("topk_softmax (mode0)", ours, theirs)

    order = torch.empty(rows, dtype=torch.int32, device="cuda")
    offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    ours = bench(lambda: K.moe_sort(kctx, route_ids, order, offsets, n_experts=e))
    theirs = bench(lambda: flex.flextrain_moe_sort(route_ids, e))
    row("sort (argsort+bincount vs custom)", ours, theirs)

    x = torch.randn(t, d, generator=g, device="cuda", dtype=torch.float32).bfloat16()
    xp = torch.empty(rows, d, dtype=torch.bfloat16, device="cuda")
    K.moe_sort(kctx, route_ids, order, offsets, n_experts=e)
    ours = bench(lambda: K.moe_dispatch_fwd(kctx, x, order, xp, top_k=k))
    # flextrain's counterpart is scatter-form (same bytes moved)
    flex_map, _counts = flex.flextrain_moe_sort(route_ids, e)
    theirs = bench(lambda: flex.flextrain_moe_scatter(x, flex_map, out=xp))
    row("dispatch gather vs flex scatter", ours, theirs, "(same bytes)")

    slot_of = torch.empty(rows, dtype=torch.int32, device="cuda")
    slot_of.scatter_(0, order.long(),
                     torch.arange(rows, dtype=torch.int32, device="cuda"))
    slot_of = slot_of.view(t, k)
    yp = torch.randn(rows, d, generator=g, device="cuda", dtype=torch.float32).bfloat16()
    resid = x.clone()
    out = torch.empty(t, d, dtype=torch.bfloat16, device="cuda")
    ours = bench(lambda: K.moe_combine_fwd(kctx, yp, slot_of, route_w, resid, out))
    theirs = bench(lambda: flex.flextrain_moe_gather(yp, flex_map, route_w, resid, out))
    row("combine (weighted+resid)", ours, theirs)
    eag = resolve_kernels(overrides={"moe_combine_fwd": "eager"})
    row("combine EAGER (for reference)",
        bench(lambda: eag.moe_combine_fwd(kctx, yp, slot_of, route_w, resid, out)))

    dh2 = torch.empty(t, d, dtype=torch.float32, device="cuda")
    ours = bench(lambda: K.moe_dispatch_bwd(kctx, yp, slot_of, dh2))
    theirs = bench(lambda: flex.flextrain_moe_gather(yp, flex_map, None, None, None))
    row("dispatch_bwd (unweighted sum)", ours, theirs)

    h13 = torch.randn(rows, 2 * f, generator=g, device="cuda", dtype=torch.float32).bfloat16()
    sact = torch.empty(rows, f, dtype=torch.bfloat16, device="cuda")
    ours = bench(lambda: K.swiglu_packed_fwd(kctx, h13, sact))
    theirs = bench(lambda: flex.flextrain_swiglu_moe_fwd(h13, out=sact))
    row("swiglu_packed fwd", ours, theirs, "(their packing reversed; same bytes)")

    # grouped GEMM: ours (F.grouped_mm) vs per-expert cuBLAS loop at
    # host-known balanced sizes = flextrain's default backend WITHOUT its
    # per-layer host sync (a lower bound for it)
    per = rows // e
    boffs = (torch.arange(0, e + 1, dtype=torch.int32, device="cuda") * per)
    w13 = torch.randn(e, d, 2 * f, generator=g, device="cuda", dtype=torch.float32).bfloat16()
    h13o = torch.empty(rows, 2 * f, dtype=torch.bfloat16, device="cuda")
    ours = bench(lambda: K.moe_grouped_mm_fwd(kctx, xp, w13, boffs, h13o))

    def loop():
        for exp in range(e):
            lo = exp * per
            torch.matmul(xp[lo:lo + per], w13[exp], out=h13o[lo:lo + per])
    theirs = bench(loop)
    row("grouped mm fwd vs cuBLAS loop", ours, theirs, "(loop needs host sync IRL)")

    dw13 = torch.empty(e, d, 2 * f, dtype=torch.bfloat16, device="cuda")
    ours = bench(lambda: K.moe_grouped_mm_wgrad(
        kctx, xp, h13, boffs, dw13, accumulate=True))

    def loop_wgrad():
        for exp in range(e):
            lo = exp * per
            dw13[exp].add_(xp[lo:lo + per].t() @ h13[lo:lo + per])
    theirs = bench(loop_wgrad)
    row("grouped mm wgrad vs cuBLAS loop", ours, theirs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=("olmoe", "qwen35moe", "both"), default="both")
    args = ap.parse_args()
    flex = load_flex()
    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    if args.shape in ("olmoe", "both"):
        run_shape("olmoe", e=64, f=1024, flex=flex)
    if args.shape in ("qwen35moe", "both"):
        run_shape("qwen35moe", e=256, f=512, flex=flex)


if __name__ == "__main__":
    main()
