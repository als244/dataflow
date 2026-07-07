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
import os
import time
from dataclasses import replace
from pathlib import Path

# Expandable segments measured strictly better on both families (A/B,
# artifacts/m5/alloc-ab-*): torch reserved collapses to the single-task
# allocated floor (qwen35 bs32: 6.95 -> 5.55 GiB vs 5.39 floor), device
# peak -1.3..-2.0 GiB, wall +1.2%. Env wins if the caller sets it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from dataflow.core import save_program
from dataflow.core.convert import to_webapp_program
from dataflow.runtime.device.cuda import CudaBackend
from dataflow.training.families import resolve_family
from dataflow.training.planning import plan_program
from dataflow.training.profiling import apply_measured_costs, cached_pcie, load_or_profile
from dataflow.training.replay import replay_gap_pct
from dataflow.training.llama3 import ShapedLlamaConfig
from dataflow.training.olmoe import ShapedOlmoeConfig
from dataflow.training.qwen3 import ShapedQwen3Config
from dataflow.training.qwen35 import ShapedQwen35Config
from dataflow.training.qwen35moe import ShapedQwen35MoeConfig
from dataflow.training.qwen3moe import ShapedQwen3MoeConfig
from dataflow.training.dsv3 import ShapedDsv3Config
from dataflow.training.dsv32 import ShapedDsv32Config
from dataflow.training.train_loop import train

GIB = 1024**3

CONFIGS = {
    "8b": ShapedLlamaConfig.llama3_8b(),
    # same effective 65,536 tokens/step, different chunking — amortizes the
    # per-step optimizer/transfer overhead over more math
    "8b-bs4ga4": ShapedLlamaConfig.llama3_8b(batch=4, grad_accum_rounds=4),
    "8b-ga16": ShapedLlamaConfig.llama3_8b(batch=1, grad_accum_rounds=16),
    # seq-1024 family: 64 sequences/step (65,536 tokens/step) chunked four ways
    # frontier edge probes (Shein): single-sequence rounds / single-round batch
    "8b-s1k-bs1ga64": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=1, grad_accum_rounds=64),
    "8b-s1k-bs64ga1": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=64, grad_accum_rounds=1),
    "8b-s1k-bs2ga32": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=2, grad_accum_rounds=32),
    "8b-s1k-bs4ga16": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=4, grad_accum_rounds=16),
    "8b-s1k-bs8ga8": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=8, grad_accum_rounds=8),
    "8b-s1k-bs16ga4": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=16, grad_accum_rounds=4),
    # 2-round shape, unlocked by the fused head_loss (the old lowering's
    # monolithic logits at bs32 was 8 GiB/round of ledger)
    "8b-s1k-bs32ga2": ShapedLlamaConfig.llama3_8b(seq_len=1024, batch=32, grad_accum_rounds=2),
    "baseline1b": ShapedLlamaConfig(
        n_layers=16, d_model=2048, n_heads=16, n_kv_heads=4, d_ff=8192,
        vocab_size=32768, seq_len=4096, batch=1,
    ),
    "mini": ShapedLlamaConfig(
        n_layers=4, d_model=1024, n_heads=8, n_kv_heads=2, d_ff=4096,
        vocab_size=16384, seq_len=1024, batch=1,
    ),
    # Qwen3-dense family (qk-norm; 36L, head_dim 128, ff 12288, vocab 151936)
    "qwen3-8b": ShapedQwen3Config.qwen3_8b(),
    "qwen3-8b-s1k-bs8ga8": ShapedQwen3Config.qwen3_8b(seq_len=1024, batch=8, grad_accum_rounds=8),
    "qwen3-8b-s1k-bs2ga32": ShapedQwen3Config.qwen3_8b(seq_len=1024, batch=2, grad_accum_rounds=32),
    # Qwen3.5-dense family (hybrid: 3x DeltaNet + 1x gated attention per 4
    # layers; untied embed/head per the 9B config, ~8.96B params)
    "qwen35-9b": ShapedQwen35Config.qwen35_9b(),
    "qwen35-9b-s1k-bs8ga8": ShapedQwen35Config.qwen35_9b(
        seq_len=1024, batch=8, grad_accum_rounds=8,
    ),
    # fewer/larger rounds: same 65,536 tok/step, 2-4x less per-step weight
    # re-streaming — the M5.2 findings put the s1k config h2d-bound
    "qwen35-9b-s1k-bs16ga4": ShapedQwen35Config.qwen35_9b(
        seq_len=1024, batch=16, grad_accum_rounds=4,
    ),
    "qwen35-9b-s1k-bs32ga2": ShapedQwen35Config.qwen35_9b(
        seq_len=1024, batch=32, grad_accum_rounds=2,
    ),
    # OLMoE-7B-A1B (first MoE family): 16L, E=64 top-8, F=1024, ~6.92B
    # params (~52 GiB pinned W+dW+O — full scale fits this box; bf16
    # weights 13.4 GiB even fit VRAM at generous budgets). The regime under
    # study: ~1.2B active params vs the full stack streamed per round.
    "olmoe-7b": ShapedOlmoeConfig.olmoe_7b(),
    "olmoe-7b-s1k-bs8ga8": ShapedOlmoeConfig.olmoe_7b(
        seq_len=1024, batch=8, grad_accum_rounds=8,
    ),
    "olmoe-7b-s1k-bs16ga4": ShapedOlmoeConfig.olmoe_7b(
        seq_len=1024, batch=16, grad_accum_rounds=4,
    ),
    "olmoe-7b-s1k-bs32ga2": ShapedOlmoeConfig.olmoe_7b(
        seq_len=1024, batch=32, grad_accum_rounds=2,
    ),
    # single-round edge: the expert stack streams ONCE per step — the
    # restreaming-minimal shape for the MoE weights>>compute regime
    "olmoe-7b-s1k-bs64ga1": ShapedOlmoeConfig.olmoe_7b(
        seq_len=1024, batch=64, grad_accum_rounds=1,
    ),
    "olmoe-7b-s1k-bs4ga16": ShapedOlmoeConfig.olmoe_7b(
        seq_len=1024, batch=4, grad_accum_rounds=16,
    ),
    # Qwen3.5-MoE: the faithful 35B-A3B needs ~277 GB pinned W+dW+O —
    # PLANNING/LOWERING VALIDATION ONLY on this 188 GB box (train would
    # exhaust host RAM allocating initial values). Perf rows use the 20L
    # variant (~17.8B, ~143 GB pinned — near the ceiling, watch OS
    # pressure): 15 lin + 5 full, E=256, shared expert, 248k vocab.
    "qwen35moe-35b": ShapedQwen35MoeConfig.qwen35moe_35b(),
    "qwen35moe-20l": ShapedQwen35MoeConfig.qwen35moe_20l(),
    "qwen35moe-20l-s1k-bs16ga4": ShapedQwen35MoeConfig.qwen35moe_20l(
        seq_len=1024, batch=16, grad_accum_rounds=4,
    ),
    "qwen35moe-20l-s1k-bs32ga2": ShapedQwen35MoeConfig.qwen35moe_20l(
        seq_len=1024, batch=32, grad_accum_rounds=2,
    ),
    "qwen35moe-20l-s1k-bs64ga1": ShapedQwen35MoeConfig.qwen35moe_20l(
        seq_len=1024, batch=64, grad_accum_rounds=1,
    ),
    # oracle's dev-12 pick (bs32/bs16 infeasible there; the curve's low end)
    "qwen35moe-20l-s1k-bs4ga16": ShapedQwen35MoeConfig.qwen35moe_20l(
        seq_len=1024, batch=4, grad_accum_rounds=16,
    ),
    # qwen3moe (Qwen3-30B-A3B family): full 48L is ~183 GiB pinned — over
    # this host; planning/validation only. 24L (~15.6B, ~94 GiB) is the
    # perf config (qwen35moe-20l precedent).
    "qwen3moe-30b": ShapedQwen3MoeConfig.qwen3moe_30b(),
    "qwen3moe-30b-24l": ShapedQwen3MoeConfig.qwen3moe_30b_24l(),
    "qwen3moe-30b-24l-s1k-bs16ga4": ShapedQwen3MoeConfig.qwen3moe_30b_24l(
        seq_len=1024, batch=16, grad_accum_rounds=4,
    ),
    "qwen3moe-30b-24l-s1k-bs32ga2": ShapedQwen3MoeConfig.qwen3moe_30b_24l(
        seq_len=1024, batch=32, grad_accum_rounds=2,
    ),
    "qwen3moe-30b-24l-s1k-bs64ga1": ShapedQwen3MoeConfig.qwen3moe_30b_24l(
        seq_len=1024, batch=64, grad_accum_rounds=1,
    ),
    # dsv3 (DeepSeek-V3): 671b is the big-machine target (1.22 TiB W —
    # lowering/planning only anywhere); dsv3-mini (12.7B, ~77 GiB pinned)
    # is THIS box's perf config.
    "dsv3-671b": ShapedDsv3Config.dsv3_671b(),
    "dsv3-mini": ShapedDsv3Config.dsv3_mini(),
    "dsv3-mini-s1k-bs16ga4": ShapedDsv3Config.dsv3_mini(
        seq_len=1024, batch=16, grad_accum_rounds=4,
    ),
    "dsv3-mini-s1k-bs32ga2": ShapedDsv3Config.dsv3_mini(
        seq_len=1024, batch=32, grad_accum_rounds=2,
    ),
    "dsv3-mini-s1k-bs64ga1": ShapedDsv3Config.dsv3_mini(
        seq_len=1024, batch=64, grad_accum_rounds=1,
    ),
    # s4k shapes (65,536 tok/step = 16 seqs): the dsv32 sparsity regime
    # (k=1024 < 4096) + dsv3 baselines for the DSA-overhead comparison
    "dsv3-mini-s4k-bs16ga1": ShapedDsv3Config.dsv3_mini(
        seq_len=4096, batch=16, grad_accum_rounds=1,
    ),
    "dsv3-mini-s4k-bs8ga2": ShapedDsv3Config.dsv3_mini(
        seq_len=4096, batch=8, grad_accum_rounds=2,
    ),
    "dsv3-mini-s4k-bs4ga4": ShapedDsv3Config.dsv3_mini(
        seq_len=4096, batch=4, grad_accum_rounds=4,
    ),
    "dsv32-671b": ShapedDsv32Config.dsv32_671b(),
    "dsv32-mini": ShapedDsv32Config.dsv32_mini(),
    "dsv32-mini-s4k-bs16ga1": ShapedDsv32Config.dsv32_mini(
        seq_len=4096, batch=16, grad_accum_rounds=1,
    ),
    "dsv32-mini-s4k-bs8ga2": ShapedDsv32Config.dsv32_mini(
        seq_len=4096, batch=8, grad_accum_rounds=2,
    ),
    "dsv32-mini-s4k-bs4ga4": ShapedDsv32Config.dsv32_mini(
        seq_len=4096, batch=4, grad_accum_rounds=4,
    ),
    # dense warm-up phase (M-H3): frozen main, indexer-only training
    "dsv32-mini-s4k-dense-bs16ga1": ShapedDsv32Config.dsv32_mini(
        seq_len=4096, batch=16, grad_accum_rounds=1, sparse_mode=False,
    ),
    "dsv32-mini-s4k-dense-bs8ga2": ShapedDsv32Config.dsv32_mini(
        seq_len=4096, batch=8, grad_accum_rounds=2, sparse_mode=False,
    ),
    # third-party shapes on our families (big-machine planning targets;
    # HF-config-verified 2026-07-07; K2.5/2.6/2.7 shape-identical to K2,
    # GLM-5.1 shape-identical to GLM-5)
    "kimi-k2": ShapedDsv3Config.kimi_k2(),
    "glm5": ShapedDsv32Config.glm5(),
    "glm5-dense-warmup": ShapedDsv32Config.glm5(sparse_mode=False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), default="8b")
    parser.add_argument("--budgets", type=str, default=None,
                        help="LEDGER budgets in GiB (planned object bytes), e.g. 12,16,20. "
                             "Physical usage exceeds this by the geometry tax + op "
                             "scratch + CUDA context. QUOTING CONVENTION: headline "
                             "sweeps use --device-gib instead, so a quoted 'N GiB' "
                             "means verified device usage <= N GiB; --budgets remains "
                             "for internal/ledger-space experiments.")
    parser.add_argument(
        "--device-gib", type=str, default=None,
        help="HARD device envelope(s) in GiB: everything — placed extent, "
             "measured op scratch, CUDA context — must fit inside. The ledger "
             "budget is derived (envelope - fixed - scratch, shaved until the "
             "packing fits). Actual device usage <= this number.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--recompute", action=argparse.BooleanOptionalAction, default=True,
        help="recompute planning (simulator-verified greedy selection over "
             "the rewrite table). ON by default since M5.2 — the qwen35 s1k "
             "sweeps ran without it and shipped a save-all plan that was "
             "h2d-bound (docs/notes/m52-perf-gap-findings.md); the planner "
             "chooses 0 recompute when save-all genuinely wins, so leaving "
             "this on costs only the recompute-variant profiling pass. "
             "--no-recompute restores the bare save-all lowering.",
    )
    parser.add_argument(
        "--force-recompute", choices=["all"], default=None,
        help="bypass the recompute planner and pin every rewrite to its "
             "recompute level (M5.2 contention probe: the greedy planner "
             "prices transfers at uncontended profiled costs, so it "
             "underestimates what recompute buys on transfer-heavy plans)",
    )
    parser.add_argument(
        "--placement", choices=["static", "dynamic", "vmm"], default="static",
        help="static (default): offsets packed offline from dry-run lifetimes, "
             "fragmentation impossible; dynamic: online slab+arena — required "
             "for shape-unstable programs (variable-length sequences); "
             "vmm: per-object stable VAs over pooled physical extents — no "
             "packing, no extent tax, physical == ledger by construction",
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
    parser.add_argument(
        "--optimizer", choices=["interleaved", "tail"], default="interleaved",
        help="optimizer task placement in the lowered chain: interleaved "
             "(default; each optimizer fires at its gradient's final mutation, "
             "state streaming overlaps the last backward round) or tail "
             "(legacy; all optimizers after all rounds — drains transfers "
             "into a GPU-idle PCIe phase)",
    )
    parser.add_argument(
        "--preplace", choices=["task0", "greedy"], default="task0",
        help="PressureFit t=0 placement: task0 (default; only task 0's "
             "inputs pre-placed, the rest arrive as planned prefetches) or "
             "greedy (legacy; fills spare capacity — the runtime then pays "
             "the whole set as a synchronous upload before each step)",
    )
    args = parser.parse_args()

    from dataflow.tasks.kernels import resolve_kernels

    cfg = replace(CONFIGS[args.config], optimizer_placement=args.optimizer)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
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
        wall_tail = report.step_wall_s[1:] or report.step_wall_s  # steps=1: no steady tail
        steady_wall = sum(wall_tail) / len(wall_tail)
        wall_tok_s = tokens_per_step / steady_wall
        print(json.dumps({
            "annotated": str(args.annotated),
            "sim_tokens_per_s": sim_tok_s,
            "real_tokens_per_s": real_tok_s,
            "wall_tokens_per_s": wall_tok_s,
            "real_vs_sim_pct": (real_tok_s / sim_tok_s - 1) * 100,
            "placement_escapes": report.placement_escapes,
            "pressure_evictions": report.pressure_evictions,
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
            fam.lower(cfg, recompute_levels=levels),
            bandwidth_from_slow=pcie.bidi_h2d,
            bandwidth_to_slow=pcie.bidi_d2h,
            backing_memory_capacity=int(args.backing_gib * GIB) if args.backing_gib else None,
        )

    program = build_raw()
    t0 = time.perf_counter()
    profiles = load_or_profile(
        program, fam.build_resolver(dims), backend, refresh=args.refresh_profiles,
    )
    if args.recompute or args.force_recompute:
        # recompute variants contain distinct signatures: block_fwd without a
        # context output, and block_recompute tasks — profile them too
        rc_all = {rw.object_id: 1 for rw in program.recompute_rewrites}
        profiles.update(load_or_profile(
            build_raw(rc_all), fam.build_resolver(dims), backend,
            refresh=args.refresh_profiles,
        ))
    measured = apply_measured_costs(program, profiles)
    print(f"profiled {len(profiles)} unique task signatures in {time.perf_counter()-t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)

    def _plan(budget: int):
        if args.force_recompute == "all":
            levels = {rw.object_id: 1 for rw in program.recompute_rewrites}
            forced = apply_measured_costs(build_raw(levels), profiles)
            planned = plan_program(
                forced, fast_memory_capacity=budget, preplace=args.preplace,
            )
            return replace(planned, recompute_levels=levels)
        return plan_program(
            measured, fast_memory_capacity=budget, recompute=args.recompute,
            build_variant=(
                lambda levels: apply_measured_costs(build_raw(levels), profiles)
            ) if args.recompute else None,
            preplace=args.preplace,
        )

    if (args.budgets is None) == (args.device_gib is None):
        parser.error("exactly one of --budgets / --device-gib is required")

    # fixed device overhead (CUDA context + resident pages) — measured for
    # EVERY mode so each row can report its true device peak:
    #   actual peak = fixed + placed extent (our cudaMalloc slab, non-torch)
    #                 + torch allocator reserved peak (task scratch)
    torch.cuda.empty_cache()  # drop profiling's cache pages first
    _free_b, _total_b = torch.cuda.mem_get_info()
    fixed = _total_b - _free_b

    if args.device_gib is not None:
        # device-envelope mode: derive the ledger budget so that
        #   fixed (context etc.) + max task scratch + packed extent <= envelope
        from dataflow.runtime import Engine
        from dataflow.runtime.device.fake import FakeBackend
        from dataflow.runtime.placement import PlacementRecorder, compute_placement

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
            if args.placement == "vmm":
                # no packing geometry: physical = ledger + arena headroom,
                # so the ledger budget follows from the envelope DIRECTLY.
                # honor the same env override the arena reads, or the
                # derivation and the pool disagree and the envelope leaks
                import os

                from dataflow.runtime.device.vmm import VmmArena

                headroom = VmmArena.headroom_bytes
                env = os.environ.get("DATAFLOW_VMM_HEADROOM_GIB")
                if env is not None:
                    headroom = int(float(env) * GIB)
                eff = avail - headroom
                print(f"  envelope {env_gib:g}: ledger {eff / GIB:.2f} "
                      f"(vmm: avail {avail / GIB:.2f} - headroom)")
            else:
                eff = avail
                try:
                    for _ in range(6):
                        ext = extent_of(_plan(eff).program)
                        print(f"  envelope {env_gib:g}: ledger {eff / GIB:.2f} -> "
                              f"extent {ext / GIB:.2f} (avail {avail / GIB:.2f})")
                        if ext <= avail:
                            break
                        eff = int(eff * avail / ext)
                except Exception as exc:
                    # infeasible at this envelope (e.g. PressureFit cannot
                    # reduce): skip the envelope, keep the rest of the sweep
                    print(f"  envelope {env_gib:g}: derivation infeasible — {exc}")
                    continue
            device_rows.append((env_gib, eff / GIB, fixed, scratch))
        budget_list = [r[1] for r in device_rows]
        device_meta = {r[1]: r for r in device_rows}
        print(f"device mode: fixed {fixed / GIB:.2f} GiB, scratch reserve "
              f"{scratch / GIB:.2f} GiB; ledger budgets {budget_list}")
    else:
        budget_list = [float(x) for x in args.budgets.split(",")]
        device_meta = {}
        print("NOTE: --budgets quotes LEDGER budgets (device usage runs higher "
              "by geometry tax + scratch + context). Headline sweeps use "
              "--device-gib so 'N GiB' means device usage <= N GiB.")

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
    # initial pinned state (weights + zeroed opt/grad, tens of GB at 8B
    # scale) is budget-INDEPENDENT: create once for the whole sweep instead
    # of alloc+fill+free per row (~1-2 min/row of pure pinned churn).
    # NOTE training mutates it, so later rows start from the previous row's
    # trained state — wall/device/fidelity are unaffected (same shapes and
    # kernels); per-row LOSSES are no longer fresh-seed comparable.
    shared_values = fam.initial_values(build_raw(), cfg, backend, seed=11)

    for gib in budget_list:
        # QUOTING CONVENTION (Shein, 2026-07-03): sweeps are quoted in DEVICE
        # budget — "28 GiB" means verified device usage <= 28 GiB. In
        # --device-gib mode budget_gib is therefore the ENVELOPE; the derived
        # ledger number stays in planned_budget_gib. --budgets (ledger) rows
        # are labeled by their ledger number, as all pre-2026-07-03 results
        # were; budget_semantics says which reading applies.
        quoted_gib = device_meta[gib][0] if gib in device_meta else gib
        semantics = "device" if gib in device_meta else "ledger"
        cap = int(gib * GIB)

        def plan_at(budget: int):
            return _plan(budget)

        try:
            planned = plan_at(cap)
        except Exception as exc:
            # PressureFit/recompute ANNOTATION infeasible at this budget (a
            # known near-capacity corner, e.g. llama interleaved @24) — record
            # the cell and keep sweeping the config's remaining budgets
            print(f"planning infeasible at {gib:g} GiB: {exc}")
            rows.append({
                "budget_gib": quoted_gib,
                "budget_semantics": semantics,
                "planned_budget_gib": gib,
                "placement_mode": args.placement,
                "status": f"planning_infeasible: {exc}",
            })
            continue
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
                values=shared_values,
            )
        except PlacementError as exc:
            # not packable on this device: keep the SIM side of the row so the
            # table stays comprehensive, and say why the real side is absent
            print(f"placement infeasible at {gib:g} GiB: {exc}")
            rows.append({
                "budget_gib": quoted_gib,
                "budget_semantics": semantics,
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
        wall_tail = report.step_wall_s[1:] or report.step_wall_s  # steps=1: no steady tail
        steady_wall = sum(wall_tail) / len(wall_tail)
        wall_tok_s = tokens_per_step / steady_wall
        # replay-fidelity is a DIAGNOSTIC: imposing real start times on the
        # sim's own transfer model can transiently overflow its
        # reserve-at-start ledger accounting even though the real engine's
        # admission was clean (first hit: olmoe bs32ga2 heavy-recompute
        # plans — MoE's huge-W/tiny-A shape tightens the timing coupling).
        # A failed diagnostic must not kill a successful measurement.
        try:
            gap = replay_gap_pct(planned.program, report.last_trace, report.step_makespan_us[-1])
        except Exception as exc:  # noqa: BLE001
            print(f"replay-fidelity diagnostic infeasible (non-fatal): {exc}")
            gap = None
        # measured device peak for EVERY row: fixed context + placed extent
        # (our cudaMalloc slab, outside torch) + torch allocator reserved
        # peak (task scratch; reset per budget). Slab overflows (0 in
        # healthy runs) would add unmeasured backend allocations.
        torch_scratch_peak = torch.cuda.max_memory_reserved()
        actual = fixed + report.placement_extent_bytes + torch_scratch_peak
        print(f"device peak (measured): {actual / GIB:.2f} GiB = "
              f"fixed {fixed / GIB:.2f} + extent "
              f"{report.placement_extent_bytes / GIB:.2f} + torch scratch "
              f"{torch_scratch_peak / GIB:.2f}")
        if gib in device_meta:
            env_b = int(device_meta[gib][0] * GIB)
            print(f"device envelope check: actual peak {actual / GIB:.2f} GiB "
                  f"{'<=' if actual <= env_b else 'EXCEEDS'} "
                  f"envelope {device_meta[gib][0]:g} GiB")
        row = {
            "budget_gib": quoted_gib,
            "budget_semantics": semantics,
            "planned_budget_gib": eff / GIB,
            **({"vmm": report.vmm_stats} if report.vmm_stats else {}),
            "actual_device_peak_gib": actual / GIB,
            "fixed_overhead_gib": fixed / GIB,
            "torch_scratch_peak_gib": torch_scratch_peak / GIB,
            **({"device_envelope_gib": device_meta[gib][0],
                "scratch_reserve_gib": device_meta[gib][3] / GIB} if gib in device_meta else {}),
            "placement_mode": args.placement,
            "extent_budget": args.extent_budget,
            "sim_ms_per_step": planned.makespan_us / 1e3,
            "real_ms_per_step_steady": steady_us / 1e3,
            "sim_tokens_per_s": sim_tok_s,
            "real_tokens_per_s": real_tok_s,
            "wall_tokens_per_s": wall_tok_s,
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
            "pressure_evictions": report.pressure_evictions,
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
              "optimizer_placement": args.optimizer, "preplace": args.preplace,
              "kernel_set": kernel_set, "sweep": rows}

    if args.baseline:
        from dataflow.runtime.device.base import Buffer  # noqa: F401
        from dataflow.tasks.interop import torch_view

        planned = plan_program(measured, fast_memory_capacity=int(24 * GIB))
        values = fam.initial_values(planned.program, cfg, backend, seed=11)

        def pinned(name):
            buf = values[name]
            return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

        golden = fam.golden().from_packed_bytes(
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

    quoted_budgets = [device_meta[b][0] if b in device_meta else b for b in budget_list]
    tag = "_".join(f"{b:g}" for b in quoted_budgets) \
        + ("dev" if device_meta else "") + ("-rc" if args.recompute else "") \
        + (f"-{args.placement}" if args.placement != "static" else "")
    (args.out / f"m4-{args.config}-{tag}.summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}/m4-{args.config}-{tag}.summary.json")


if __name__ == "__main__":
    main()
