"""Estimate tok/s for EVERY bs/ga combo at each device budget — simulator only.

Input: a model (family + preset and/or arbitrary config fields), seq length,
total sequences per step, and device envelopes.
Output: predicted tok/s for every divisor pair (bs x ga = seqs/step) at every
envelope, the best combo per envelope, and optional JSON for automation.
No training steps run: each cell is profile-load -> plan (recompute planner +
PressureFit) -> simulate. GPU is touched only when a shape's profiles are not
cached yet (~30-60 s per new shape, once).

Validated against the measured llama3-8B frontier (29 real cells): ranking
agreed at every envelope; the one inside-noise call (dev-14, 1.0% margin) was
flagged and settled by a real run. With contended profiles (the default since
2026-07-06) raw sim is CONSERVATIVE: measured wall ran +2.6..+6.6% above it,
tightening as budgets grow. Margins under ~2% are within that noise — the
tool flags them; settle top-2 with `bench_train --device-gib`.

The model is specified the way the registry thinks about it: --family names
a REGISTERED FAMILY (llama3, qwen3, qwen35 — the grammar/lowering/kernels),
and the concrete model is a CONFIG of that family: --preset picks a named
classmethod (8b, 9b, tiny, ...) and/or --model-json supplies arbitrary
config fields (n_layers, d_model, n_heads, n_kv_heads, d_ff, vocab_size,
...) overriding the preset/defaults. Adding a new architecture = one family
module (docs/extending.md); every config of a registered family works here
with zero tool changes.

Usage:
  python tools/best_config.py --family llama3 --preset 8b --seq-len 1024 \
      --seqs-per-step 64 --device-gib 10,12,16,24,30
  python tools/best_config.py --family qwen35 --preset 9b --seq-len 1024 \
      --seqs-per-step 64 --device-gib 18,24 --json picks.json
  python tools/best_config.py --family llama3 \
      --model-json '{"n_layers": 16, "d_model": 2048, "n_heads": 16,
                     "n_kv_heads": 4, "d_ff": 8192, "vocab_size": 32000}' \
      --seq-len 2048 --seqs-per-step 32 --device-gib 8,12
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time

# match bench_train's allocator convention: the big-round MoE shapes (bs64ga1:
# many ~2 GiB tail temporaries) fragment the default allocator and OOM the
# profiling launch well under capacity — expandable segments fixed exactly
# this class in the alloc A/B study
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# stream results as they compute: sweeps run backgrounded/redirected, and
# block-buffered stdout hides the table until exit
print = functools.partial(print, flush=True)
from dataclasses import replace

GIB = 1024**3

# measured wall/sim on contended profiles (2026-07-06 frontier validation);
# raw sim is already conservative — calibration only sharpens absolutes
CONTENDED_CALIB_DEFAULT = 1.04  # +-3%; ranking is the load-bearing output


def config_factory(family_name: str, preset: str | None, model_json: str | None):
    """Return make(batch, rounds, seq_len) -> shaped config for the family.

    Base fields come from the preset classmethod (tried as `{preset}` then
    `{family}_{preset}`), else the config class defaults; --model-json fields
    override either. batch/ga/seq_len are injected per enumerated combo."""
    import dataclasses

    from dataflow.training.families import family as family_named

    fam = family_named(family_name)
    cls = fam.config_type
    if preset:
        method = getattr(cls, preset, None) or getattr(cls, f"{family_name}_{preset}", None)
        if method is None:
            names = [n for n in dir(cls)
                     if not n.startswith("_") and callable(getattr(cls, n))
                     and isinstance(cls.__dict__.get(n), classmethod)]
            raise SystemExit(f"family {family_name!r} has no preset {preset!r} "
                             f"(available: {names})")
        base = method()
    else:
        base = cls()
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    if model_json:
        overrides = json.loads(model_json)
        unknown = set(overrides) - set(fields)
        if unknown:
            raise SystemExit(f"unknown config fields for {cls.__name__}: {sorted(unknown)}")
        fields.update(overrides)

    def make(batch: int, rounds: int, seq_len: int):
        d = dict(fields)
        d.update(batch=batch, grad_accum_rounds=rounds, seq_len=seq_len)
        return cls(**d)

    return make, cls.__name__


def divisor_combos(seqs: int, cap: int | None) -> list[tuple[int, int]]:
    out = []
    b = 1
    while b <= seqs:
        if seqs % b == 0 and (cap is None or b <= cap):
            out.append((b, seqs // b))
        b *= 2
    if seqs & (seqs - 1):  # non-power-of-two: include all divisors
        out = [(b, seqs // b) for b in range(1, seqs + 1)
               if seqs % b == 0 and (cap is None or b <= cap)]
    return out


def main() -> None:
    from dataflow.training.families import load_plugins

    ap = argparse.ArgumentParser(
        description="simulate tok/s for every bs/ga combo per device budget")
    ap.add_argument("--family", required=True,
                    help="registered model family: llama3, qwen3, qwen35")
    ap.add_argument("--preset", default=None,
                    help="named config of the family (8b, 9b, tiny, ...)")
    ap.add_argument("--model-json", default=None,
                    help="JSON dict of config fields overriding preset/defaults")
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--seqs-per-step", type=int, required=True,
                    help="total sequences per optimizer step (= bs x ga)")
    ap.add_argument("--device-gib", required=True,
                    help="device envelopes, e.g. 10,12,16,24")
    ap.add_argument("--max-bs", type=int, default=None,
                    help="optional cap on batch size candidates")
    ap.add_argument("--skip-slow-bs", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="prune small-bs columns that provably cannot win: a "
                         "combo's compute-only ceiling (sum of measured task "
                         "runtimes; no plan can beat it) is checked against the "
                         "row leader, sweeping bs and envelopes DESCENDING. "
                         "Near-misses (ceiling within 10%% of the leader) get a "
                         "seeds-only reduced search, auto-promoted to a full "
                         "search if they top the row. Full table: ~tens of "
                         "seconds instead of tens of minutes")
    ap.add_argument("--calibrated", action=argparse.BooleanOptionalAction,
                    default=False,
                    help=f"scale raw sim by {CONTENDED_CALIB_DEFAULT} (measured "
                         "wall/sim on contended profiles); default reports the "
                         "conservative raw sim")
    ap.add_argument("--json", type=str, default=None,
                    help="write results as JSON (for automated selection)")
    ap.add_argument("--plugin", action="append", default=None,
                   help="external family plugin module(s); installed "
                        "dataflow.families entry points load automatically")
    args = ap.parse_args()
    load_plugins(explicit=[m for arg in (args.plugin or [])
                           for m in arg.split(",")])

    import torch  # noqa: F401

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

    make_config, cfg_name = config_factory(args.family, args.preset, args.model_json)
    tokens_per_step = args.seq_len * args.seqs_per_step
    combos = divisor_combos(args.seqs_per_step, args.max_bs)
    combos.sort(reverse=True)  # largest bs first: leaders compute cheapest
    print(f"{args.family} ({cfg_name}"
          + (f", preset {args.preset}" if args.preset else "")
          + (", +overrides" if args.model_json else "")
          + f"): seq {args.seq_len} x {args.seqs_per_step} seqs/step "
          f"= {tokens_per_step:,} tok/step; combos: "
          + ", ".join(f"bs{b}ga{g}" for b, g in combos))

    def extent_of(prog) -> int:
        rec = PlacementRecorder()
        Engine(FakeBackend()).execute(prog, record_placement=rec).close()
        return compute_placement(rec, physical_limit_bytes=2**62).extent_bytes

    plans = {}
    for bs, rounds in combos:
        cfg = make_config(bs, rounds, args.seq_len)
        fam = resolve_family(cfg)
        dims = fam.dims_of(cfg)

        def build(levels=None, cfg=cfg, fam=fam):
            return replace(
                fam.lower(cfg, recompute_levels=levels),
                bandwidth_from_slow=pcie.bidi_h2d, bandwidth_to_slow=pcie.bidi_d2h,
            )

        t0 = time.perf_counter()
        try:
            profiles = load_or_profile(build(), fam.build_resolver(dims), backend)
            base = build()
            rc_all = {rw.object_id: 1 for rw in base.recompute_rewrites}
            # ...and BETWEEN the base and rc-variant passes of one shape:
            # the base pass's cached segments (peak-signature sized) stack
            # under the rc pass's new size classes (qwen35moe bs64 OOM'd
            # here while its base pass succeeded)
            import torch as _torch

            _torch.cuda.empty_cache()
            profiles.update(
                load_or_profile(build(rc_all), fam.build_resolver(dims), backend))
        except Exception as e:
            print(f"  bs{bs}ga{rounds}: profiling failed ({type(e).__name__}: {e})")
            continue
        finally:
            # torch's caching allocator hoards segments ACROSS shapes in this
            # one-process loop; the next shape's raw cudaMalloc profiling
            # buffers then starve (bs16 OOM'd after bs64+bs32 profiled)
            import torch

            torch.cuda.empty_cache()
        scratch = max(p.workspace_bytes for p in profiles.values()) + (256 << 20)
        measured = apply_measured_costs(base, profiles)
        compute_us = sum(t.runtime_us for t in measured.tasks)
        ceiling = tokens_per_step / (compute_us / 1e6)  # no plan can beat this
        plans[(bs, rounds)] = (build, profiles, scratch, ceiling)
        dt = time.perf_counter() - t0
        note = "cached" if dt < 5 else f"profiled {dt:.0f}s"
        print(f"  bs{bs}ga{rounds}: ready ({note}), scratch {scratch/GIB:.2f} GiB, "
              f"compute-only ceiling {ceiling:,.0f} tok/s")

    envs = sorted((float(x) for x in args.device_gib.split(",")), reverse=True)
    calib = CONTENDED_CALIB_DEFAULT if args.calibrated else 1.0
    results = []
    print(f"\npredicted tok/s ({'calibrated x' + str(calib) if args.calibrated else 'raw sim, conservative'}):")
    header = "device | " + " | ".join(f"{f'bs{b}ga{g}':>9s}" for b, g in combos)
    print(header)
    print("-" * len(header))
    dead_columns: dict = {}  # combo -> value at the largest envelope it ran

    def plan_cell(build, profiles, avail, *, reduced=False):
        eff = avail
        planned = None
        for _ in range(1 if reduced else 6):
            planned = plan_program(
                apply_measured_costs(build(), profiles),
                recompute=True, fast_memory_capacity=eff,
                # reduced: seeds-only search, two inbound schedules
                # (packed-fifo = congestion winner, latest-safe = the
                # pressure SURVIVOR — dropping it caused false
                # infeasibility), no extent-shave: a fast LOWER estimate
                # for columns that are not contending (auto-promoted to a
                # full search if one tops the row)
                max_iters=0 if reduced else 8,
                pressurefit_schedules=(
                    ("packed-fifo", "latest-safe") if reduced else None),
                build_variant=lambda lv, b=build, p=profiles:
                    apply_measured_costs(b(lv), p),
            )
            if reduced:
                break
            ext = extent_of(planned.program)
            if ext <= avail:
                break
            eff = int(eff * avail / ext)
        return planned, eff

    for env_gib in envs:
        cells = []
        row_best = 0.0
        for bs, rounds in combos:
            entry = plans.get((bs, rounds))
            if entry is None:
                cells.append(dict(bs=bs, ga=rounds, status="no-profiles"))
                continue
            build, profiles, scratch, ceiling = entry
            avail = int(env_gib * GIB) - fixed - scratch
            if avail <= 0:
                cells.append(dict(bs=bs, ga=rounds, status="no-fit"))
                continue
            if args.skip_slow_bs:
                # tok/s is monotone in budget: a column that already lost at a
                # LARGER envelope can never win a smaller one
                prior = dead_columns.get((bs, rounds))
                bound = min(ceiling, prior) if prior is not None else ceiling
                if bound <= row_best:
                    cells.append(dict(bs=bs, ga=rounds, status="skip:bounded",
                                      bound=round(bound)))
                    continue
                reduced = ceiling <= row_best * 1.10
            else:
                reduced = False
            try:
                try:
                    planned, eff = plan_cell(build, profiles, avail, reduced=reduced)
                except Exception:
                    if not reduced:
                        raise
                    # belt-and-braces: if the cheap search still fails,
                    # confirm with the full one before declaring infeasible
                    planned, eff = plan_cell(build, profiles, avail, reduced=False)
                    reduced = False
                tps = tokens_per_step / (planned.makespan_us / 1e6) * calib
                if reduced and tps > row_best:
                    # a seeds-only estimate is a LOWER bound; if it tops the
                    # row, redo it with the full search before believing it
                    planned, eff = plan_cell(build, profiles, avail)
                    tps = tokens_per_step / (planned.makespan_us / 1e6) * calib
                    reduced = False
                row_best = max(row_best, tps)
                if args.skip_slow_bs:
                    prior = dead_columns.get((bs, rounds))
                    dead_columns[(bs, rounds)] = max(prior or 0, tps)
                cells.append(dict(
                    bs=bs, ga=rounds, tok_s=round(tps),
                    ledger_gib=round(eff / GIB, 2),
                    recompute=sum(1 for v in planned.recompute_levels.values() if v),
                    rewrites=len(planned.recompute_levels),
                    **({"reduced_search": True} if reduced else {}),
                ))
            except Exception as e:
                cells.append(dict(bs=bs, ga=rounds,
                                  status=f"infeasible:{type(e).__name__}"))
        ok = [c for c in cells if "tok_s" in c]
        best = max(ok, key=lambda c: c["tok_s"], default=None)
        parts = []
        for c in cells:
            if "tok_s" not in c:
                label = "skip" if c["status"].startswith("skip") else c["status"][:9]
                parts.append(f"{label:>9s}")
            else:
                mark = "*" if best and c is best else ("~" if c.get("reduced_search") else " ")
                parts.append(f"{c['tok_s']:8,}{mark}")
        line = f"dev-{env_gib:<3g}| " + " | ".join(parts)
        margin = None
        if best and len(ok) > 1:
            second = sorted((c["tok_s"] for c in ok), reverse=True)[1]
            margin = (best["tok_s"] / second - 1) * 100
        if best:
            line += f"   best: bs{best['bs']}ga{best['ga']}"
            if margin is not None and margin <= 2:
                line += f"  (margin {margin:.1f}% ~ noise: verify top-2 with bench_train)"
        print(line)
        results.append(dict(
            device_gib=env_gib,
            cells=cells,
            best=(dict(bs=best["bs"], ga=best["ga"], tok_s=best["tok_s"],
                       margin_pct=None if margin is None else round(margin, 2))
                  if best else None),
        ))

    if args.json:
        payload = dict(
            family=args.family, preset=args.preset,
            model_overrides=json.loads(args.model_json) if args.model_json else None,
            seq_len=args.seq_len,
            seqs_per_step=args.seqs_per_step, tokens_per_step=tokens_per_step,
            calibrated=args.calibrated, calibration=calib,
            note="raw sim on contended profiles is conservative: measured wall "
                 "ran +2.6..+6.6% above it on the validated frontier; margins "
                 "<=2% are noise — verify top-2 with real runs",
            envelopes=results,
        )
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
