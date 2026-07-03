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
from dataflow.training.profiling import apply_measured_costs, cached_pcie, load_or_profile
from dataflow.training.replay import replay_gap_pct
from dataflow.training.shaped_llama3 import ShapedLlamaConfig
from dataflow.training.train_loop import train

GIB = 1024**3

CONFIGS = {
    "8b": ShapedLlamaConfig.llama3_8b(),
    # same effective 65,536 tokens/step, different chunking — amortizes the
    # per-step optimizer/transfer overhead over more math
    "8b-bs4ga4": ShapedLlamaConfig.llama3_8b(batch=4, grad_accum_rounds=4),
    "8b-ga16": ShapedLlamaConfig.llama3_8b(batch=1, grad_accum_rounds=16),
    # seq-1024 family: 64 sequences/step (65,536 tokens/step) chunked four ways
    "8b-s1k-bs2ga32": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=2, grad_accum_rounds=32),
    "8b-s1k-bs4ga16": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=4, grad_accum_rounds=16),
    "8b-s1k-bs8ga8": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=8, grad_accum_rounds=8),
    "8b-s1k-bs16ga4": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=16, grad_accum_rounds=4),
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
    parser.add_argument("--budgets", type=str, default=None,
                        help="LEDGER budgets in GiB (planned object bytes), e.g. 12,16,20. "
                             "Physical usage exceeds this by the geometry tax + op "
                             "scratch + CUDA context — use --device-gib for a hard "
                             "device envelope instead.")
    parser.add_argument(
        "--device-gib", type=str, default=None,
        help="HARD device envelope(s) in GiB: everything — placed extent, "
             "measured op scratch, CUDA context — must fit inside. The ledger "
             "budget is derived (envelope - fixed - scratch, shaved until the "
             "packing fits). Actual device usage <= this number.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--embed-host", action="store_true",
        help="embed-on-host: table lives on CPU (sparse gather/scatter + "
             "host optimizer); removes W/O/dW_embed and embed tasks from "
             "the chain",
    )
    parser.add_argument(
        "--placement", choices=["static", "dynamic"], default="static",
        help="static (default): offsets packed offline from dry-run lifetimes, "
             "fragmentation impossible; dynamic: online slab+arena — required "
             "for shape-unstable programs (variable-length sequences)",
    )
    parser.add_argument(
        "--probe-max", action="store_true",
        help="replace the budget list with the LARGEST placement-feasible "
             "budget for this config: probe one plan, scale by its measured "
             "geometry tax, verify by packing (CPU-only), step down 0.25 GiB "
             "on miss",
    )
    parser.add_argument(
        "--extent-budget", action="store_true",
        help="make the budget bound the PHYSICAL footprint: iteratively shave "
             "the planning budget until the packed extent (geometry tax "
             "included) fits within the nominal budget",
    )
    parser.add_argument("--baseline", action="store_true", help="also time the plain-torch golden model")
    parser.add_argument(
        "--backing-gib", type=float, default=None,
        help="cap pinned-host bytes; sim verification then rejects plans whose "
             "offload footprint exceeds host RAM (essential at high grad-accum: "
             "save-all context across rounds can exceed system memory)",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/m4"))
    parser.add_argument(
        "--annotated", type=Path, default=None,
        help="skip profiling/planning and train a SAVED annotated program "
             "(exact replay of a prior plan; --config must still match its "
             "shapes for token generation and the resolver)",
    )
    parser.add_argument("--refresh-profiles", action="store_true",
                        help="ignore the profile cache and re-measure")
    args = parser.parse_args()

    from dataflow.tasks.kernels import resolve_kernels

    cfg = CONFIGS[args.config]
    if args.embed_host:
        from dataclasses import replace as _replace

        cfg = _replace(cfg, embed_on_host=True)
    dims = dims_of(cfg)
    tokens_per_step = float(cfg.tokens * cfg.grad_accum_rounds)

    if args.annotated is not None:
        from dataflow.core import load_program
        from dataflow.training.planning import simulate_program

        program = load_program(args.annotated)
        log = simulate_program(program)
        sim_us = max(iv.end for iv in log.task_intervals)
        sim_tok_s = tokens_per_step / (sim_us / 1e6)
        print(f"replaying saved plan {args.annotated.name}: "
              f"sim {sim_us / 1e3:.1f} ms/step ({sim_tok_s:.0f} tok/s)")
        backend = CudaBackend()
        report = train(program, cfg, backend, steps=args.steps, seed=11)
        real_tok_s = tokens_per_step / (report.steady_state_makespan_us / 1e6)
        print(json.dumps({
            "annotated": str(args.annotated),
            "sim_tokens_per_s": sim_tok_s,
            "real_tokens_per_s": real_tok_s,
            "real_vs_sim_pct": (real_tok_s / sim_tok_s - 1) * 100,
            "placement_escapes": report.placement_escapes,
            "losses": [round(x, 4) for x in report.losses],
        }, indent=2))
        return

    kernel_set = resolve_kernels().describe()
    impls = sorted(set(kernel_set.values()))
    print(f"kernel set: {impls} ({len(kernel_set)} registry ops)")
    backend = CudaBackend()
    pcie = cached_pcie(backend)
    print(f"PCIe GB/s: uni {pcie.uni_h2d/1e3:.1f}/{pcie.uni_d2h/1e3:.1f}  "
          f"bidi {pcie.bidi_h2d/1e3:.1f}/{pcie.bidi_d2h/1e3:.1f}")

    def build_raw(levels=None):
        return replace(
            lower_llama3(cfg, recompute_levels=levels),
            bandwidth_from_slow=pcie.bidi_h2d,
            bandwidth_to_slow=pcie.bidi_d2h,
            backing_memory_capacity=int(args.backing_gib * GIB) if args.backing_gib else None,
        )

    program = build_raw()
    t0 = time.perf_counter()
    profiles = load_or_profile(
        program, build_resolver(dims), backend, refresh=args.refresh_profiles,
    )
    if args.recompute:
        # recompute variants contain distinct signatures: block_fwd without a
        # context output, and block_recompute tasks — profile them too
        rc_all = {rw.object_id: 1 for rw in program.recompute_rewrites}
        profiles.update(load_or_profile(
            build_raw(rc_all), build_resolver(dims), backend,
            refresh=args.refresh_profiles,
        ))
    measured = apply_measured_costs(program, profiles)
    print(f"profiled {len(profiles)} unique task signatures in {time.perf_counter()-t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)

    def _plan(budget: int):
        return plan_program(
            measured, fast_memory_capacity=budget, recompute=args.recompute,
            build_variant=(
                lambda levels: apply_measured_costs(build_raw(levels), profiles)
            ) if args.recompute else None,
        )

    if (args.budgets is None) == (args.device_gib is None):
        parser.error("exactly one of --budgets / --device-gib is required")

    if args.device_gib is not None:
        # device-envelope mode: derive the ledger budget so that
        #   fixed (context etc.) + max task scratch + packed extent <= envelope
        import torch as _torch

        from dataflow.runtime import Engine
        from dataflow.runtime.device.fake import FakeBackend
        from dataflow.runtime.placement import PlacementRecorder, compute_placement

        _torch.cuda.empty_cache()  # drop profiling's cache pages: 'fixed'
        free_b, total_b = _torch.cuda.mem_get_info()  # = context + residents
        fixed = total_b - free_b
        scratch = max(p.workspace_bytes for p in profiles.values()) + (256 << 20)

        def extent_of(planned_prog) -> int:
            rec = PlacementRecorder()
            Engine(FakeBackend()).execute(planned_prog, record_placement=rec).close()
            return compute_placement(rec, physical_limit_bytes=2**62).extent_bytes

        device_rows = []
        for env_gib in [float(x) for x in args.device_gib.split(",")]:
            envelope = int(env_gib * GIB)
            avail = envelope - fixed - scratch
            if avail <= 0:
                print(f"device envelope {env_gib:g} GiB < fixed+scratch "
                      f"({(fixed + scratch) / GIB:.2f} GiB) — skipping")
                continue
            eff = avail
            for _ in range(6):
                ext = extent_of(_plan(eff).program)
                print(f"  envelope {env_gib:g}: ledger {eff / GIB:.2f} -> "
                      f"extent {ext / GIB:.2f} (avail {avail / GIB:.2f})")
                if ext <= avail:
                    break
                eff = int(eff * avail / ext)
            device_rows.append((env_gib, eff / GIB, fixed, scratch))
        budget_list = [r[1] for r in device_rows]
        device_meta = {r[1]: r for r in device_rows}
        print(f"device mode: fixed {fixed / GIB:.2f} GiB, scratch reserve "
              f"{scratch / GIB:.2f} GiB; ledger budgets {budget_list}")
    else:
        budget_list = [float(x) for x in args.budgets.split(",")]
        device_meta = {}

    if args.probe_max:
        from dataflow.runtime import Engine
        from dataflow.runtime.device.fake import FakeBackend
        from dataflow.runtime.placement import PlacementRecorder, compute_placement

        PHYS = 27 * GIB

        def extent_at(gib_probe: float) -> int:
            rec = PlacementRecorder()
            Engine(FakeBackend()).execute(
                _plan(int(gib_probe * GIB)).program, record_placement=rec
            ).close()
            return compute_placement(rec, physical_limit_bytes=2**62).extent_bytes

        probe = 24.0
        ext = extent_at(probe)
        cand = min(27.0, probe * PHYS / ext)          # linear extent scaling
        cand = int(cand * 4) / 4                       # quarter-GiB floor
        print(f"probe: extent({probe:g}) = {ext / GIB:.2f} GiB "
              f"(tax x{ext / (probe * GIB):.3f}) -> candidate {cand:g} GiB")
        while cand > 20:
            e = extent_at(cand)
            print(f"probe: extent({cand:g}) = {e / GIB:.2f} GiB "
                  f"{'<= ' if e <= PHYS else '> '}{PHYS / GIB:.0f} GiB limit")
            if e <= PHYS:
                break
            cand -= 0.25
        budget_list = [cand]
        print(f"largest feasible budget: {cand:g} GiB")

    rows = []
    for gib in budget_list:
        cap = int(gib * GIB)

        def plan_at(budget: int):
            return _plan(budget)

        planned = plan_at(cap)
        eff = cap
        placement = None
        if args.placement == "static" and args.extent_budget:
            # shave the planning budget until the packed extent fits the
            # nominal budget: "N GiB" then bounds what the device physically
            # holds, not just the ledger. The geometry tax is paid out of the
            # budget instead of out of headroom above it.
            from dataflow.runtime import Engine
            from dataflow.runtime.device.fake import FakeBackend
            from dataflow.runtime.placement import PlacementRecorder, compute_placement

            for attempt in range(6):
                recorder = PlacementRecorder()
                Engine(FakeBackend()).execute(
                    planned.program, record_placement=recorder
                ).close()
                placement = compute_placement(recorder, physical_limit_bytes=2**62)
                if placement.extent_bytes <= cap:
                    break
                eff = int(eff * cap / placement.extent_bytes)
                print(f"  extent {placement.extent_bytes / GIB:.2f} GiB > budget; "
                      f"re-planning at {eff / GIB:.2f} GiB")
                planned = plan_at(eff)
            else:
                print(f"  extent-budget search did not converge at {gib:g} GiB — skipping")
                continue

        sim_tok_s = tokens_per_step / (planned.makespan_us / 1e6)
        print(f"\n=== budget {gib:g} GiB: sim predicts {planned.makespan_us/1e3:.1f} ms/step "
              f"({sim_tok_s:.0f} tok/s), peak {planned.peak_fast_bytes/GIB:.2f} GiB, "
              f"recompute {sum(1 for v in planned.recompute_levels.values() if v)}/"
              f"{len(planned.recompute_levels)} ===")

        torch.cuda.empty_cache()  # release prior budget's torch scratch
        torch.cuda.reset_peak_memory_stats()  # scratch peak of THIS run only
        from dataflow.runtime.placement import PlacementError

        try:
            report = train(
                planned.program, cfg, backend, steps=args.steps, seed=11,
                placement_mode=args.placement, placement=placement,
            )
        except PlacementError as exc:
            # not packable on this device: keep the SIM side of the row so the
            # table stays comprehensive, and say why the real side is absent
            print(f"placement infeasible at {gib:g} GiB: {exc}")
            rows.append({
                "budget_gib": gib,
                "planned_budget_gib": eff / GIB,
                "placement_mode": args.placement,
                "extent_budget": args.extent_budget,
                "sim_ms_per_step": planned.makespan_us / 1e3,
                "sim_tokens_per_s": sim_tok_s,
                "recompute_chosen": sum(1 for v in planned.recompute_levels.values() if v),
                "status": f"placement_infeasible: {exc}",
            })
            continue
        steady_us = report.steady_state_makespan_us
        real_tok_s = tokens_per_step / (steady_us / 1e6)
        gap = replay_gap_pct(planned.program, report.last_trace, report.step_makespan_us[-1])
        if gib in device_meta:
            # verify the envelope claim against reality: base extent (ours,
            # non-torch) + torch cache peak + fixed context
            actual = (device_meta[gib][2] + report.placement_extent_bytes
                      + torch.cuda.max_memory_reserved())
            env_b = int(device_meta[gib][0] * GIB)
            print(f"device envelope check: actual peak {actual / GIB:.2f} GiB "
                  f"{'<=' if actual <= env_b else 'EXCEEDS'} "
                  f"envelope {device_meta[gib][0]:g} GiB")
        row = {
            "budget_gib": gib,
            "planned_budget_gib": eff / GIB,
            **({"device_envelope_gib": device_meta[gib][0],
                "fixed_overhead_gib": device_meta[gib][2] / GIB,
                "scratch_reserve_gib": device_meta[gib][3] / GIB,
                "actual_device_peak_gib": (device_meta[gib][2]
                    + report.placement_extent_bytes
                    + torch.cuda.max_memory_reserved()) / GIB} if gib in device_meta else {}),
            "placement_mode": args.placement,
            "extent_budget": args.extent_budget,
            "sim_ms_per_step": planned.makespan_us / 1e3,
            "real_ms_per_step_steady": steady_us / 1e3,
            "sim_tokens_per_s": sim_tok_s,
            "real_tokens_per_s": real_tok_s,
            "real_vs_sim_pct": (real_tok_s / sim_tok_s - 1) * 100,
            "replay_fidelity_gap_pct": gap,
            "losses": report.losses,
            "peak_fast_gib": report.peak_fast_bytes / GIB,
            "peak_backing_gib": report.peak_backing_bytes / GIB,
            "pinned_host_gib": report.pinned_host_bytes / GIB,
            "step_makespans_ms": [m / 1e3 for m in report.step_makespan_us],
            "step_wall_s": report.step_wall_s,
            "step_slab_overflows": report.step_slab_overflows,
            "placement_extent_gib": report.placement_extent_bytes / GIB,
            "placement_escapes": report.placement_escapes,
            "placement_overhead": report.placement_overhead,
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

    result = {"config": args.config, "steps": args.steps, "pcie": pcie.__dict__,
              "kernel_set": kernel_set, "sweep": rows}

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

    tag = "_".join(f"{b:g}" for b in budget_list) + ("-rc" if args.recompute else "")
    (args.out / f"m4-{args.config}-{tag}.summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}/m4-{args.config}-{tag}.summary.json")


if __name__ == "__main__":
    main()
