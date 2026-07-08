"""A/B microbench: fla chunk_gated_delta_rule + causal_conv1d at qwen3.5-9B shapes.

Motivated by the qwen35 linear-attention perf gap. Two
candidate explanations for our lin-attn tasks costing more than flextrain's:

1. varlen invocation overhead — we call the kernels as (1, B*T) +
   cu_seqlens; the handoff hypothesized flextrain calls them dense (B, T).
   (Reading flextrain/nn/blocks/linear_attn.py shows it ALSO uses the
   varlen path, so this axis should measure ~neutral — verify.)
2. launch-size parallelism — flextrain's solver packs 32,768 tokens into
   one kernel launch ("2 chunks x 32 layers" at 65,536 tok/step); our
   bs8 rounds launch 8,192. Fewer/larger launches amortize fixed cost and
   fill SMs during the inter-chunk recurrent stages.

This times fwd+bwd for both kernels at matched per-token work across
launch shapes, mirroring src/dataflow/tasks/models/qwen35_blocks.py invocations
exactly (contiguity contract included). Run in the `dataflow` env with the
GPU otherwise idle:

    python tools/bench_qwen35_kernels.py --out artifacts/qwen35-kernel-ab/bench.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

# Qwen3.5-9B linear-attention shapes (models/Qwen3.5-9B config.json).
HK, HV, K, V, W = 16, 32, 128, 128, 4
KEY_DIM = HK * K              # 2048
VALUE_DIM = HV * V            # 4096
CONV_D = 2 * KEY_DIM + VALUE_DIM  # 8192
SEQ = 1024
SCALE = K ** -0.5


def _cu(n_seqs: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.modules.conv.triton.ops import prepare_chunk_indices

    cu = torch.arange(n_seqs + 1, device=device, dtype=torch.int64) * SEQ
    return cu, prepare_chunk_indices(cu, 64)


def _inputs(b: int, t: int, device: str, seed: int = 0) -> dict:
    """Random inputs at realistic magnitudes; q/k unit-normalized like the
    block's l2norm stage."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape):
        return (torch.randn(*shape, generator=gen) * 0.02).to(torch.bfloat16)

    q = torch.nn.functional.normalize(
        torch.randn(b, t, HK, K, generator=gen), dim=-1
    ).to(torch.bfloat16).to(device).contiguous()
    k = torch.nn.functional.normalize(
        torch.randn(b, t, HK, K, generator=gen), dim=-1
    ).to(torch.bfloat16).to(device).contiguous()
    v = r(b, t, HV, V).to(device).contiguous()
    a_raw = r(b, t, HV).to(device).contiguous()          # pre-softplus gate input
    beta = torch.sigmoid(r(b, t, HV).float()).to(torch.bfloat16).to(device).contiguous()
    do = r(b, t, HV, V).to(device).contiguous()
    A_log = torch.log(
        1.0 + 15.0 * torch.rand(HV, generator=gen)
    ).float().to(device)
    dt_bias = (torch.randn(HV, generator=gen) * 0.02).float().to(device)
    conv_x = r(b * t, CONV_D).to(device).contiguous()    # token-major, like our ops
    conv_dy = r(b * t, CONV_D).to(device).contiguous()
    conv_w = r(CONV_D, W).to(device).contiguous()
    return dict(
        q=q, k=k, v=v, a_raw=a_raw, beta=beta, do=do, A_log=A_log,
        dt_bias=dt_bias, conv_x=conv_x, conv_dy=conv_dy, conv_w=conv_w,
    )


def _time(fn, warmup: int, iters: int) -> float:
    """Median CUDA-event milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def bench_config(name: str, b: int, t: int, varlen: bool, device: str,
                 warmup: int, iters: int) -> dict:
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_fwd,
    )
    import fla.modules.conv.triton.ops as fops

    x = _inputs(b, t, device)
    tokens = b * t
    if varlen:
        assert b == 1 and t % SEQ == 0
        cu, ci = _cu(t // SEQ, device)
    else:
        cu, ci = None, None

    def fla_fwd():
        return chunk_gated_delta_rule_fwd(
            x["q"], x["k"], x["v"], x["a_raw"], x["beta"],
            scale=SCALE, initial_state=None, output_final_state=False,
            cu_seqlens=cu, chunk_indices=ci,
            use_gate_in_kernel=True, A_log=x["A_log"], dt_bias=x["dt_bias"],
        )

    g_post, o, A_int, _fs, _is, _gi = fla_fwd()

    def fla_bwd():
        return chunk_gated_delta_rule_bwd(
            q=x["q"], k=x["k"], v=x["v"], g=g_post, beta=x["beta"], A=A_int,
            scale=SCALE, initial_state=None, do=x["do"], dht=None,
            cu_seqlens=cu, chunk_indices=ci,
            use_gate_in_kernel=True, g_input=x["a_raw"],
            A_log=x["A_log"], dt_bias=x["dt_bias"],
        )

    # Conv mirrors our registry wrapper: token-major (t, D) unsqueezed to
    # (1, t, D) + cu_seqlens; the dense variant feeds a true (B, T, D).
    cx = x["conv_x"] if varlen else x["conv_x"].view(b, t, CONV_D)
    cdy = x["conv_dy"] if varlen else x["conv_dy"].view(b, t, CONV_D)
    cx3 = cx.unsqueeze(0) if varlen else cx
    cdy3 = cdy.unsqueeze(0) if varlen else cdy

    def conv_fwd():
        return fops.causal_conv1d_fwd(
            cx3, x["conv_w"], None, None, activation="silu", cu_seqlens=cu,
        )

    def conv_bwd():
        return fops.causal_conv1d_bwd(
            cx3, cdy3, None, weight=x["conv_w"], bias=None, residual=None,
            initial_state=None, activation="silu", cu_seqlens=cu,
        )

    row = {"config": name, "batch": b, "tokens_per_launch": tokens}
    for label, fn in (("fla_fwd", fla_fwd), ("fla_bwd", fla_bwd),
                      ("conv_fwd", conv_fwd), ("conv_bwd", conv_bwd)):
        ms = _time(fn, warmup, iters)
        row[f"{label}_ms"] = round(ms, 4)
        row[f"{label}_ns_per_tok"] = round(ms * 1e6 / tokens, 2)
    row["total_ns_per_tok"] = round(
        sum(row[f"{l}_ns_per_tok"] for l in ("fla_fwd", "fla_bwd", "conv_fwd", "conv_bwd")), 2
    )
    del x, g_post, o, A_int
    torch.cuda.empty_cache()
    return row


CONFIGS = (
    # name, B, T, varlen
    ("dense-8x1024", 8, 1024, False),      # the handoff's hypothesized flextrain mode
    ("varlen-8k", 1, 8192, True),          # OUR current launch (bs8 rounds)
    ("dense-32x1024", 32, 1024, False),    # separates varlen-tax from launch-size at 32k
    ("varlen-32k", 1, 32768, True),        # flextrain's ACTUAL launch (2 chunks/step)
    ("varlen-64k", 1, 65536, True),        # bs64ga1 single-launch scenario
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    import fla

    meta = {
        "torch": torch.__version__,
        "fla": getattr(fla, "__version__", "?"),
        "gpu": torch.cuda.get_device_name(args.device),
        "shapes": dict(HK=HK, HV=HV, K=K, V=V, conv_dim=CONV_D, conv_w=W, seq=SEQ),
        "warmup": args.warmup, "iters": args.iters,
    }
    print(f"# {meta['gpu']}  torch {meta['torch']}  fla {meta['fla']}")

    rows = []
    for name, b, t, varlen in CONFIGS:
        row = bench_config(name, b, t, varlen, args.device, args.warmup, args.iters)
        rows.append(row)
        print(
            f"{name:>14}  tok/launch={row['tokens_per_launch']:>6}  "
            f"fla_fwd={row['fla_fwd_ms']:>8.3f}ms  fla_bwd={row['fla_bwd_ms']:>8.3f}ms  "
            f"conv_fwd={row['conv_fwd_ms']:>7.3f}ms  conv_bwd={row['conv_bwd_ms']:>7.3f}ms  "
            f"total={row['total_ns_per_tok']:>7.1f} ns/tok"
        )

    base = next(r for r in rows if r["config"] == "varlen-8k")
    print("\n# speedup vs varlen-8k (our current launch), total lin-kernel ns/token")
    for r in rows:
        rel = base["total_ns_per_tok"] / r["total_ns_per_tok"]
        print(f"{r['config']:>14}  {r['total_ns_per_tok']:>7.1f} ns/tok  x{rel:.3f}")

    if args.out:
        import os

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
