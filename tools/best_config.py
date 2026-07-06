"""Pick the best bs/ga combo per device budget — WITHOUT training steps.

For a family + tokens/step, enumerates bs/ga candidates, plans each at each
device envelope (measured profiles from cache; GPU touched only for missing
shapes), and ranks by simulated tok/s. Validated against the measured
matrix (artifacts/m5): raw sim ranking agreed with wall ranking at every
tested envelope, and wall/sim calibration is stable per config
(llama bs8/16/32: 0.947/0.922/0.925, stdev <1%; qwen35 bs32: 0.893).

Margins under ~2% are within calibration noise — the tool flags those
cells; verify the top-2 with real runs (m4_train --device-gib) if it
matters.

Usage:
  python tools/best_config.py --family llama3-8b --device-gib 16,18,20,24
  python tools/best_config.py --family qwen35-9b --device-gib 18,24 --bs 8,16,32
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

GIB = 1024**3

# wall/sim calibration measured on the m5 matrix (see module docstring);
# families fall back to DEFAULT_CALIB for unmeasured shapes
CALIB = {
    ("llama3-8b", 8): 0.947, ("llama3-8b", 16): 0.922, ("llama3-8b", 32): 0.925,
    ("llama3-8b", 4): 1.037, ("llama3-8b", 2): 1.191,
    ("qwen35-9b", 8): 0.995, ("qwen35-9b", 16): 0.901, ("qwen35-9b", 32): 0.893,
}
DEFAULT_CALIB = 0.93


def make_cfg(family: str, seq_len: int, batch: int, rounds: int):
    if family == "llama3-8b":
        from dataflow.training.llama3 import ShapedLlamaConfig

        return ShapedLlamaConfig.llama3_8b(
            seq_len=seq_len, batch=batch, grad_accum_rounds=rounds)
    if family == "qwen35-9b":
        from dataflow.training.qwen35 import ShapedQwen35Config

        return ShapedQwen35Config.qwen35_9b(
            seq_len=seq_len, batch=batch, grad_accum_rounds=rounds)
    raise SystemExit(f"unknown family {family!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True, choices=["llama3-8b", "qwen35-9b"])
    ap.add_argument("--tokens-per-step", type=int, default=65536)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--bs", default="2,4,8,16,32",
                    help="candidate batch sizes (ga derived from tokens/step)")
    ap.add_argument("--device-gib", required=True,
                    help="device envelopes to rank at, e.g. 16,18,24")
    ap.add_argument("--calibrated", action=argparse.BooleanOptionalAction, default=True,
                    help="scale sim tok/s by the measured wall/sim factor")
    args = ap.parse_args()

    import torch  # noqa: F401  (backend init below)

    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.placement import PlacementRecorder, compute_placement
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program
    from dataflow.training.profiling import (
        apply_measured_costs, cached_pcie, load_or_profile,
    )

    backend = CudaBackend()
    pcie = cached_pcie(backend)

    import torch as _torch
    _torch.cuda.empty_cache()
    free_b, total_b = _torch.cuda.mem_get_info()
    fixed = total_b - free_b

    candidates = []
    for bs in [int(x) for x in args.bs.split(",")]:
        per_round = bs * args.seq_len
        if args.tokens_per_step % per_round:
            print(f"  skip bs{bs}: tokens/step not divisible")
            continue
        rounds = args.tokens_per_step // per_round
        candidates.append((bs, rounds))

    def extent_of(prog) -> int:
        rec = PlacementRecorder()
        Engine(FakeBackend()).execute(prog, record_placement=rec).close()
        return compute_placement(rec, physical_limit_bytes=2**62).extent_bytes

    # profile every candidate shape once (cache hits are free)
    plans = {}
    for bs, rounds in candidates:
        cfg = make_cfg(args.family, args.seq_len, bs, rounds)
        fam = resolve_family(cfg)
        dims = fam.dims_of(cfg)

        def build(levels=None, cfg=cfg, fam=fam):
            return replace(
                fam.lower(cfg, recompute_levels=levels),
                bandwidth_from_slow=pcie.bidi_h2d, bandwidth_to_slow=pcie.bidi_d2h,
            )

        t0 = time.perf_counter()
        profiles = load_or_profile(build(), fam.build_resolver(dims), backend)
        base = build()
        rc_all = {rw.object_id: 1 for rw in base.recompute_rewrites}
        profiles.update(load_or_profile(build(rc_all), fam.build_resolver(dims), backend))
        scratch = max(p.workspace_bytes for p in profiles.values()) + (256 << 20)
        plans[(bs, rounds)] = (build, profiles, scratch)
        print(f"  bs{bs}ga{rounds}: profiles ready ({time.perf_counter()-t0:.0f}s), "
              f"scratch reserve {scratch/GIB:.2f} GiB")

    envs = [float(x) for x in args.device_gib.split(",")]
    print(f"\n{args.family} @ {args.tokens_per_step} tok/step — predicted tok/s "
          f"({'calibrated' if args.calibrated else 'raw sim'}):")
    header = "device | " + " | ".join(f"bs{bs}ga{r}" for bs, r in candidates)
    print(header)
    print("-" * len(header))
    for env_gib in envs:
        cells = []
        for bs, rounds in candidates:
            build, profiles, scratch = plans[(bs, rounds)]
            avail = int(env_gib * GIB) - fixed - scratch
            if avail <= 0:
                cells.append((bs, rounds, None, None, "no-fit"))
                continue
            eff = avail
            try:
                planned = None
                for _ in range(6):
                    planned = plan_program(
                        apply_measured_costs(build(), profiles),
                        recompute=True, fast_memory_capacity=eff,
                        build_variant=lambda lv, b=build, p=profiles:
                            apply_measured_costs(b(lv), p),
                    )
                    ext = extent_of(planned.program)
                    if ext <= avail:
                        break
                    eff = int(eff * avail / ext)
                sim = args.tokens_per_step / (planned.makespan_us / 1e6)
                calib = CALIB.get((args.family, bs), DEFAULT_CALIB) if args.calibrated else 1.0
                cells.append((bs, rounds, sim * calib,
                              sum(1 for v in planned.recompute_levels.values() if v), None))
            except Exception as e:
                cells.append((bs, rounds, None, None, type(e).__name__))
        best = max((c for c in cells if c[2]), key=lambda c: c[2], default=None)
        parts = []
        for bs, rounds, tps, rc, err in cells:
            if tps is None:
                parts.append(f"{err:>10s}")
                continue
            mark = " *" if best and (bs, rounds) == best[:2] else "  "
            parts.append(f"{tps:7,.0f}{mark}")
        line = f"dev-{env_gib:<3g}| " + " | ".join(parts)
        if best:
            runners = sorted((c[2] for c in cells if c[2]), reverse=True)
            margin = (runners[0] / runners[1] - 1) * 100 if len(runners) > 1 else 99
            note = "" if margin > 2 else f"   <-- margin {margin:.1f}% ~ noise: verify top-2"
            line += f"   best: bs{best[0]}ga{best[1]}{note}"
        print(line)


if __name__ == "__main__":
    main()
