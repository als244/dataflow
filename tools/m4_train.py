"""M4 gate: memory-constrained multi-step llama3 training, swept vs the sim.

Per budget: plan on MEASURED costs (profiled unique tasks + measured PCIe
bidi bandwidth), run N optimizer steps on the 5090, report steady-state
tokens/s against the simulator's prediction plus the replay-fidelity gap
(pure scheduling overhead, cost-model error factored out).

Also (--baseline) times the plain eager-torch golden model on a config that
fits in VRAM, vs our runtime at a generous budget on the same config —
grounding absolute throughput. The 8B headline config is NOT plain-torch
trainable on 32 GB (params+grads+Adam ≈ 60 GB): that's the thesis.

Usage:
    python tools/m4_train.py --config 8b --budgets 12,16,20 --steps 4
    python tools/m4_train.py --config baseline1b --budgets 24 --steps 4 --baseline
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from dataflow.core import save_program
from dataflow.core.convert import to_webapp_program
from dataflow.runtime.device.cuda import CudaBackend
from dataflow.tasks.llama3_blocks import build_resolver
from dataflow.training.llama3_lowering import dims_of, lower_llama3
from dataflow.training.planning import plan_program
from dataflow.training.profiling import apply_measured_costs, profile_program
from dataflow.training.replay import replay_gap_pct
from dataflow.training.shaped_llama3 import ShapedLlamaConfig
from dataflow.training.train_loop import train

GIB = 1024**3

CONFIGS = {
    "8b": ShapedLlamaConfig.llama3_8b(),
    "baseline1b": ShapedLlamaConfig(
        n_layers=16, d_model=2048, n_heads=16, n_kv_heads=4, d_ff=8192,
        vocab_size=32768, seq_len=4096, batch=1,
    ),
    "mini": ShapedLlamaConfig(
        n_layers=4, d_model=1024, n_heads=8, n_kv_heads=2, d_ff=4096,
        vocab_size=16384, seq_len=1024, batch=1,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), default="8b")
    parser.add_argument("--budgets", type=str, required=True, help="GiB list, e.g. 12,16,20")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="also time the plain-torch golden model")
    parser.add_argument("--out", type=Path, default=Path("artifacts/m4"))
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    dims = dims_of(cfg)
    tokens_per_step = float(cfg.tokens * cfg.grad_accum_rounds)
    backend = CudaBackend()
    pcie = backend.measure_pcie()
    print(f"PCIe GB/s: uni {pcie.uni_h2d/1e3:.1f}/{pcie.uni_d2h/1e3:.1f}  "
          f"bidi {pcie.bidi_h2d/1e3:.1f}/{pcie.bidi_d2h/1e3:.1f}")

    def build_raw(levels=None):
        return replace(
            lower_llama3(cfg, recompute_levels=levels),
            bandwidth_from_slow=pcie.bidi_h2d,
            bandwidth_to_slow=pcie.bidi_d2h,
        )

    program = build_raw()
    t0 = time.perf_counter()
    profiles = profile_program(program, build_resolver(dims), backend)
    if args.recompute:
        # recompute variants contain distinct signatures: block_fwd without a
        # context output, and block_recompute tasks — profile them too
        rc_all = {rw.object_id: 1 for rw in program.recompute_rewrites}
        profiles.update(profile_program(build_raw(rc_all), build_resolver(dims), backend))
    measured = apply_measured_costs(program, profiles)
    print(f"profiled {len(profiles)} unique task signatures in {time.perf_counter()-t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for gib in [float(x) for x in args.budgets.split(",")]:
        cap = int(gib * GIB)
        planned = plan_program(
            measured, fast_memory_capacity=cap, recompute=args.recompute,
            build_variant=(
                lambda levels: apply_measured_costs(build_raw(levels), profiles)
            ) if args.recompute else None,
        )
        sim_tok_s = tokens_per_step / (planned.makespan_us / 1e6)
        print(f"\n=== budget {gib:g} GiB: sim predicts {planned.makespan_us/1e3:.1f} ms/step "
              f"({sim_tok_s:.0f} tok/s), peak {planned.peak_fast_bytes/GIB:.2f} GiB, "
              f"recompute {sum(1 for v in planned.recompute_levels.values() if v)}/"
              f"{len(planned.recompute_levels)} ===")

        torch.cuda.empty_cache()  # release prior budget's torch scratch
        report = train(planned.program, cfg, backend, steps=args.steps, seed=11)
        steady_us = report.steady_state_makespan_us
        real_tok_s = tokens_per_step / (steady_us / 1e6)
        gap = replay_gap_pct(planned.program, report.last_trace, report.step_makespan_us[-1])
        row = {
            "budget_gib": gib,
            "sim_ms_per_step": planned.makespan_us / 1e3,
            "real_ms_per_step_steady": steady_us / 1e3,
            "sim_tokens_per_s": sim_tok_s,
            "real_tokens_per_s": real_tok_s,
            "real_vs_sim_pct": (real_tok_s / sim_tok_s - 1) * 100,
            "replay_fidelity_gap_pct": gap,
            "losses": report.losses,
            "peak_fast_gib": report.peak_fast_bytes / GIB,
            "step_makespans_ms": [m / 1e3 for m in report.step_makespan_us],
            "step_wall_s": report.step_wall_s,
            "step_slab_overflows": report.step_slab_overflows,
            "recompute_chosen": sum(1 for v in planned.recompute_levels.values() if v),
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "losses"}, indent=2))
        print("losses:", [round(x, 4) for x in report.losses])

        stem = f"{planned.program.name}-{gib:g}gib"
        save_program(planned.program, args.out / f"{stem}.annotated.json")
        (args.out / f"{stem}.webapp.json").write_text(
            json.dumps(to_webapp_program(measured), indent=2) + "\n"
        )

    result = {"config": args.config, "steps": args.steps, "pcie": pcie.__dict__, "sweep": rows}

    if args.baseline:
        from dataflow.models.llama3_reference import GoldenLlama3
        from dataflow.runtime.device.base import Buffer  # noqa: F401
        from dataflow.tasks.interop import torch_view
        from dataflow.training.llama3_lowering import initial_values

        planned = plan_program(measured, fast_memory_capacity=int(24 * GIB))
        values = initial_values(planned.program, cfg, backend, seed=11)

        def pinned(name):
            buf = values[name]
            return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

        golden = GoldenLlama3.from_packed_bytes(
            dims, cfg.n_layers, pinned("W_embed"),
            [pinned(f"W_{i}") for i in range(cfg.n_layers)], pinned("W_head"),
        )
        gen = torch.Generator().manual_seed(12)
        toks = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen).cuda()
        tgts = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen).cuda()
        golden.train_step(toks, tgts)  # warm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            golden.train_step(toks, tgts)
        torch.cuda.synchronize()
        base_ms = (time.perf_counter() - t0) / 3 * 1e3
        result["baseline_plain_torch_ms_per_step"] = base_ms
        result["baseline_tokens_per_s"] = tokens_per_step / (base_ms / 1e3)
        print(f"\nplain-torch golden baseline: {base_ms:.1f} ms/step "
              f"({result['baseline_tokens_per_s']:.0f} tok/s)")

    tag = args.budgets.replace(",", "_") + ("-rc" if args.recompute else "")
    (args.out / f"m4-{args.config}-{tag}.summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}/m4-{args.config}-{tag}.summary.json")


if __name__ == "__main__":
    main()
