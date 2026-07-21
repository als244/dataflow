#!/usr/bin/env python
"""Pretraining orchestration: reference-vs-engine parity + scaling runs.

Subcommands:
  engine     engine-only long run (checkpoints/resume, doc-aware feed, profile)
  reference  pure-torch twin long run (same recipe/feed conventions)
  smoke      tiny real-vocab model, reference vs service engine — the infra gate
  parity     one preset, reference + engine at N device budgets -> curves + report
  scaling    the ladder, one backend/budget -> loss curves for the scaling study
  peek       read an in-flight run's loss curve from its newest checkpoint

(`tools/train/eval_checkpoint.py` computes a checkpoint's held-out
val loss — the published-curve axis.)

All backends share one deterministic data pipeline, the cosine recipe, and a
byte-identical seeded init (the reference bridges the engine's packed bytes).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from dataflow_training.run import parity, presets as P
from dataflow_training.run.driver import (
    daemon_client,
    init_model,
    run_engine,
    run_reference,
)
from dataflow_training.data.pipeline import (
    legacy_block_pipeline,
    pipeline_from_args,
)
from dataflow_training.run.recipe import Recipe

RESULTS = _ROOT / "results" / "pretrain"


def _recipe(steps: int, *, peak_lr: float = 3e-4,
            muon_lr: float | None = None) -> Recipe:
    return Recipe(peak_lr=peak_lr, min_lr=peak_lr / 10,
                  warmup_steps=max(1, steps // 10), total_steps=steps,
                  muon_lr=muon_lr)


def _pipeline(cfg, args):
    """Build the run's data pipeline from the --data spec + packing
    flags (data.pipeline_from_args rejects the retired block/doc
    literals with a migration hint)."""
    return pipeline_from_args(
        cfg, args.data, policy=args.packing_policy,
        allow_round_split=args.allow_round_split,
        capture=args.capture)


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _init_bytes_identical(cfg, client, seed: int) -> bool:
    """The daemon's init program (already run, objects in the store) must
    byte-match the in-process initial_values(seed) the reference bridged
    from. Call AFTER init_model, BEFORE any run()."""
    import numpy as np
    import torch

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family

    fam = resolve_family(cfg)
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=seed)
    try:
        for oid in ("W_embed", "W_0", "W_head"):
            ref = torch_view(values[oid], (values[oid].size_bytes,),
                             torch.uint8).clone().numpy()
            got = np.frombuffer(client.get_object(oid), dtype=np.uint8)
            if not np.array_equal(ref, got):
                return False
        return True
    finally:
        for buf in values.values():
            backend.free(buf)


def cmd_smoke(args) -> int:
    cfg = P.smoke_preset()
    steps = args.steps
    recipe = _recipe(steps)
    feed = legacy_block_pipeline(cfg)
    lnV = math.log(cfg.vocab_size)
    _log(f"SMOKE: {cfg.n_layers}L d{cfg.d_model} vocab{cfg.vocab_size} "
         f"seq{cfg.seq_len}x{cfg.batch} ga{cfg.grad_accum_rounds} "
         f"steps{steps}  ln(V)={lnV:.3f}")

    ref = run_reference(cfg, recipe, feed, steps, seed=11, log=_log)
    with daemon_client(slab_gib=args.slab, log=_log) as client:
        # seed the store, then check the daemon's init byte-matches the
        # reference's (before any run() mutates the weights)
        init_model(client, "llama3", P.cfg_dict(cfg), seed=11)
        identical = _init_bytes_identical(cfg, client, seed=11)
        _log(f"init byte-identity (daemon vs reference): {identical}")
        eng = run_engine(client, cfg, recipe, feed, steps,
                         budget_gib=args.budget, seed=11, log=_log)

    RESULTS.mkdir(parents=True, exist_ok=True)
    ref.save(RESULTS / "smoke_reference.json")
    eng.save(RESULTS / "smoke_engine.json")

    okr, msgr = parity.curves_healthy(ref.losses, expect_start=lnV, min_drop=0.2)
    oke, msge = parity.curves_healthy(eng.losses, expect_start=lnV, min_drop=0.2)
    _log(f"reference health: {okr} ({msgr})")
    _log(f"engine    health: {oke} ({msge})")
    rep = parity.compare(ref.losses, eng.losses, a_label="reference",
                         b_label=f"engine@{args.budget}GiB")
    _log(rep.summary())
    passed = identical and okr and oke and rep.passed
    _log(f"SMOKE {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


def cmd_parity(args) -> int:
    cfg = P.resolve_preset(args.preset)
    steps = args.steps
    recipe = _recipe(steps)
    feed = legacy_block_pipeline(cfg)
    budgets = [float(b) for b in args.budgets.split(",")]
    lnV = math.log(cfg.vocab_size)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _log(f"PARITY {args.preset}: steps{steps} budgets{budgets} "
         f"tok/step={P.tokens_per_step(cfg)}")

    ref = run_reference(cfg, recipe, feed, steps, seed=11,
                        grad_checkpoint=args.grad_checkpoint, log=_log)
    ref.save(RESULTS / f"{args.preset}_reference.json")
    # the reference and the engine share this process's CUDA context:
    # release the reference's cached allocations before the daemon
    # claims its slab/pool (a 1.9B reference otherwise starves it)
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    engs = {}
    with daemon_client(slab_gib=args.slab, log=_log) as client:
        for b in budgets:
            eng = run_engine(client, cfg, recipe, feed, steps,
                             budget_gib=b, seed=11, log=_log)
            eng.save(RESULTS / f"{args.preset}_engine_{b:g}gib.json")
            engs[b] = eng

    ok = True
    okr, msgr = parity.curves_healthy(ref.losses, expect_start=lnV)
    _log(f"reference health: {okr} ({msgr})")
    for b, eng in engs.items():
        rep = parity.compare(ref.losses, eng.losses, a_label="reference",
                             b_label=f"engine@{b:g}GiB")
        _log(rep.summary())
        ok = ok and rep.passed
    # budget-invariance: engine curves agree with each other
    if len(budgets) == 2:
        b0, b1 = budgets
        rep = parity.compare(engs[b0].losses, engs[b1].losses,
                             a_label=f"engine@{b0:g}GiB",
                             b_label=f"engine@{b1:g}GiB")
        _log("budget-invariance " + rep.summary())
    return 0 if ok else 1


def cmd_scaling(args) -> int:
    steps = args.steps
    recipe = _recipe(steps)
    presets = args.presets.split(",")
    RESULTS.mkdir(parents=True, exist_ok=True)
    _log(f"SCALING presets{presets} backend={args.backend} steps{steps}")
    if args.backend == "reference":
        for name in presets:
            cfg = P.preset(name)
            feed = legacy_block_pipeline(cfg)
            r = run_reference(cfg, recipe, feed, steps, seed=11,
                              grad_checkpoint=args.grad_checkpoint, log=_log)
            r.meta["preset"] = name
            r.meta["params"] = P.param_counts(cfg)
            r.save(RESULTS / f"scaling_{name}_reference.json")
    else:
        with daemon_client(slab_gib=args.slab, log=_log) as client:
            for name in presets:
                cfg = P.preset(name)
                feed = legacy_block_pipeline(cfg)
                r = run_engine(client, cfg, recipe, feed, steps,
                               budget_gib=args.budget, seed=11, log=_log)
                r.meta["preset"] = name
                r.meta["params"] = P.param_counts(cfg)
                r.save(RESULTS / f"scaling_{name}_engine.json")
                client.wipe("all", force=True)   # reset store for the next size
    return 0


def cmd_reference(args) -> int:
    """Reference-ONLY run: the pure-torch twin trained end to end, the
    loss curve saved as a yardstick (e.g. the muon-recipe curve the
    engine leg must later match)."""
    from dataclasses import replace

    cfg = P.resolve_preset(args.preset)
    overrides = {}
    if args.opt:
        overrides["opt_policy"] = args.opt
    if args.ga_rounds:
        overrides["grad_accum_rounds"] = args.ga_rounds
    if overrides:
        cfg = replace(cfg, **overrides)
    recipe = _recipe(args.steps, peak_lr=args.peak_lr)
    feed = _pipeline(cfg, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ck_dir = None
    partial = None
    if args.checkpoint_every:
        ck_dir = RESULTS / "checkpoints" / out.stem
        ck_dir.mkdir(parents=True, exist_ok=True)
        partial = out.with_name(out.stem + "_partial.json")
    _log(f"REFERENCE-ONLY: {args.preset} opt={getattr(cfg, 'opt_policy', 'adamw')} "
         f"steps={args.steps} data={args.data} "
         f"grad_checkpoint={args.grad_checkpoint} "
         f"tokens/step={cfg.seq_len * cfg.batch * cfg.grad_accum_rounds} "
         f"ckpt={args.checkpoint_every or 'off'}")
    res = run_reference(cfg, recipe, feed, args.steps,
                        grad_checkpoint=args.grad_checkpoint,
                        checkpoint_every=args.checkpoint_every,
                        checkpoint_dir=ck_dir, resume=args.resume,
                        partial_out=partial, log=_log)
    res.save(out)
    _log(f"saved {out} (final loss {res.losses[-1]:.4f})")
    return 0


def cmd_engine(args) -> int:
    """Engine-ONLY run (one daemon, one GPU): the service engine
    trained end to end, the curve saved for comparison against a
    reference yardstick (same seed/feed/recipe conventions).
    Long runs checkpoint host-locally and resume with --resume."""
    from dataclasses import replace

    cfg = P.resolve_preset(args.preset)
    overrides = {}
    if args.opt:
        overrides["opt_policy"] = args.opt
    if args.ga_rounds:
        overrides["grad_accum_rounds"] = args.ga_rounds
    if args.batch:
        overrides["batch"] = args.batch
    if overrides:
        cfg = replace(cfg, **overrides)
    recipe = _recipe(args.steps, peak_lr=args.peak_lr,
                     muon_lr=args.muon_lr)
    feed = _pipeline(cfg, args)
    ck_dir = None
    if args.checkpoint_every:
        ck_dir = RESULTS / "checkpoints" / Path(args.out).stem
        ck_dir.mkdir(parents=True, exist_ok=True)
    _log(f"ENGINE-ONLY: {args.preset} opt={getattr(cfg, 'opt_policy', 'adamw')} "
         f"steps={args.steps} budget={args.budget}GiB data={args.data} "
         f"tokens/step={cfg.seq_len * cfg.batch * cfg.grad_accum_rounds} "
         f"ckpt={args.checkpoint_every or 'off'}")
    with daemon_client(slab_gib=args.slab, log=_log) as client:
        profile = None
        if args.profile:
            profile = {"start": args.profile_start_before_step,
                       "stop": args.profile_stop_after_step}
        res = run_engine(client, cfg, recipe, feed, args.steps,
                         budget_gib=args.budget, seed=11, log=_log,
                         measured=args.measured, profile=profile,
                         checkpoint_every=args.checkpoint_every,
                         checkpoint_dir=ck_dir, resume=args.resume)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.save(out)
    _log(f"saved {out} (final loss {res.losses[-1]:.4f})")
    return 0


CKPTS = _ROOT / "results" / "pretrain" / "checkpoints"


def newest_checkpoint(run: str, step: int | None):
    """The run's checkpoint dir at ``step`` (or newest complete), or None."""
    run_dir = CKPTS / run
    if step is not None:
        ck = run_dir / f"step_{step:06d}"
        return ck if (ck / "manifest.json").is_file() else None
    manifests = sorted(run_dir.glob("step_*/manifest.json"))
    return manifests[-1].parent if manifests else None


def cmd_peek(args) -> int:
    """Newest checkpoint's embedded loss curve -> summary + partial json."""
    import json

    ck = newest_checkpoint(args.run, None)
    if ck is None:
        print(f"no complete checkpoints under {CKPTS / args.run}",
              file=sys.stderr)
        return 1
    manifest = json.loads((ck / "manifest.json").read_text())
    meta = manifest.get("client_meta", {})
    losses = [float(x) for x in meta.get("losses", [])]
    if not losses:
        print(f"{ck} carries no loss curve", file=sys.stderr)
        return 1
    ema_v = losses[0]
    for x in losses:
        ema_v = args.ema * ema_v + (1 - args.ema) * x
    out = _ROOT / "results" / "pretrain" / f"{args.run}_partial.json"
    out.write_text(json.dumps({
        "backend": "engine", "partial_through_step": int(meta["step"]),
        "losses": losses, "meta": {"source": str(ck / "manifest.json")},
    }, indent=2))
    print(f"{args.run}: {len(losses)} steps recorded "
          f"(through step {meta['step']})")
    print(f"  last loss {losses[-1]:.4f}   EMA({args.ema}) {ema_v:.4f}   "
          f"min {min(losses):.4f}")
    print(f"  partial curve -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("engine")
    e.add_argument("--preset", default="l3_1b")
    e.add_argument("--steps", type=int, default=P.TRAIN_STEPS)
    e.add_argument("--opt", choices=["adamw", "muon"], default=None)
    e.add_argument("--peak-lr", type=float, default=3e-4)
    e.add_argument("--muon-lr", type=float, default=None,
                   help="muon params' own PEAK lr (rides the same "
                        "warmup+cosine shape; default: share --peak-lr)")
    e.add_argument("--budget", type=float, default=14.0)
    e.add_argument("--slab", type=float, default=100.0)
    e.add_argument("--batch", type=int, default=None,
                   help="override cfg.batch (tokens/round scales)")
    e.add_argument("--ga-rounds", type=int, default=None,
                   help="override grad-accum rounds (tokens/step = "
                        "seq*batch*rounds; 64 -> 512K at the locked "
                        "8192-token round)")
    e.add_argument("--checkpoint-every", type=int, default=None)
    e.add_argument("--resume", action="store_true")
    e.add_argument("--profile", action="store_true",
                   help="bracket a step window with profiler_control "
                        "(cudaProfilerStart/Stop) — run under "
                        "tools/bench/nsys_profile.py to capture it")
    e.add_argument("--profile-start-before-step", type=int, default=3)
    e.add_argument("--profile-stop-after-step", type=int, default=6)
    e.add_argument("--measured", action="store_true",
                   help="plan with PROFILED task costs (disk-cached) — the "
                        "[plan] line's prediction becomes the true-profiling "
                        "simulator expectation")
    e.add_argument("--data", default=None,
                   help="data source spec (scheme:args,k=v — see "
                        "docs/data_feeds.md); default: the in-repo "
                        "shard corpus, per-document")
    e.add_argument("--packing-policy", choices=["ffd", "greedy"],
                   default="ffd")
    e.add_argument("--allow-round-split", action="store_true",
                   help="legacy exact-fill packing: sequences split at "
                        "round edges (with long_policy=whole this "
                        "reproduces the historical curves)")
    e.add_argument("--capture", default=None,
                   help="log every consumed sequence to this file "
                        "(replayable via --data capture:PATH)")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_engine)

    r = sub.add_parser("reference")
    r.add_argument("--preset", default="l3_1b")
    r.add_argument("--steps", type=int, default=P.TRAIN_STEPS)
    r.add_argument("--opt", choices=["adamw", "muon"], default=None,
                   help="override the preset's opt_policy ('muon' = the "
                        "hybrid recipe: muon matrices, adamw for "
                        "embed/head/norms)")
    r.add_argument("--peak-lr", type=float, default=3e-4)
    r.add_argument("--grad-checkpoint", action="store_true")
    r.add_argument("--ga-rounds", type=int, default=None,
                   help="override cfg.grad_accum_rounds (tokens/step scales)")
    r.add_argument("--checkpoint-every", type=int, default=None)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--data", default=None,
                   help="data source spec (scheme:args,k=v); default: "
                        "the in-repo shard corpus, per-document")
    r.add_argument("--packing-policy", choices=["ffd", "greedy"],
                   default="ffd")
    r.add_argument("--allow-round-split", action="store_true")
    r.add_argument("--capture", default=None)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_reference)

    s = sub.add_parser("smoke")
    s.add_argument("--steps", type=int, default=P.SMOKE_STEPS)
    s.add_argument("--budget", type=float, default=4.0)
    s.add_argument("--slab", type=float, default=8.0)
    s.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("parity")
    p.add_argument("--preset", default="l3_1b")
    p.add_argument("--steps", type=int, default=P.TRAIN_STEPS)
    p.add_argument("--budgets", default="6,14")
    p.add_argument("--slab", type=float, default=100.0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.set_defaults(fn=cmd_parity)

    sc = sub.add_parser("scaling")
    sc.add_argument("--presets", default="l3_125m,l3_350m,l3_760m,l3_1b")
    sc.add_argument("--steps", type=int, default=P.TRAIN_STEPS)
    sc.add_argument("--backend", choices=["reference", "engine"], default="engine")
    sc.add_argument("--budget", type=float, default=14.0)
    sc.add_argument("--slab", type=float, default=100.0)
    sc.add_argument("--grad-checkpoint", action="store_true")
    sc.set_defaults(fn=cmd_scaling)

    pk = sub.add_parser("peek")
    pk.add_argument("run", help="run name (the --out stem; a directory "
                                "under results/pretrain/checkpoints/)")
    pk.add_argument("--ema", type=float, default=0.98)
    pk.set_defaults(fn=cmd_peek)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
