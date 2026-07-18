#!/usr/bin/env python
"""Pretraining orchestration: reference-vs-engine parity + scaling runs.

Subcommands:
  smoke    tiny real-vocab model, reference vs service engine — the infra gate
  parity   one preset, reference + engine at N device budgets -> curves + report
  scaling  the ladder, one backend/budget -> loss curves for the scaling study

All backends share the deterministic fineweb stream, the cosine recipe, and a
byte-identical seeded init (the reference bridges the engine's packed bytes).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dataflow_training.run import parity, presets as P
from dataflow_training.run.driver import (
    daemon_client,
    run_engine,
    run_reference,
)
from dataflow_training.data.fineweb import make_stream
from dataflow_training.run.recipe import Recipe

RESULTS = _ROOT / "results" / "pretrain"


def _recipe(steps: int, *, peak_lr: float = 3e-4) -> Recipe:
    return Recipe(peak_lr=peak_lr, min_lr=peak_lr / 10,
                  warmup_steps=max(1, steps // 10), total_steps=steps)


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _init_bytes_identical(cfg, client, seed: int) -> bool:
    """The daemon's family_init_all(seed) (already materialized in the store)
    must byte-match the in-process initial_values(seed) the reference bridged
    from. Call AFTER materialize_group, BEFORE any run()."""
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
    stream = make_stream(cfg.tokens)
    lnV = math.log(cfg.vocab_size)
    _log(f"SMOKE: {cfg.n_layers}L d{cfg.d_model} vocab{cfg.vocab_size} "
         f"seq{cfg.seq_len}x{cfg.batch} ga{cfg.grad_accum_rounds} "
         f"steps{steps}  ln(V)={lnV:.3f}")

    ref = run_reference(cfg, recipe, stream, steps, seed=11, log=_log)
    with daemon_client(slab_gib=args.slab, log=_log) as client:
        # seed the store, then check the daemon's init byte-matches the
        # reference's (before any run() mutates the weights)
        client.materialize_group({"kind": "family_init_all", "family": "llama3",
                                  "cfg": P.cfg_dict(cfg), "seed": 11})
        identical = _init_bytes_identical(cfg, client, seed=11)
        _log(f"init byte-identity (daemon vs reference): {identical}")
        eng = run_engine(client, cfg, recipe, stream, steps,
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
    stream = make_stream(cfg.tokens)
    budgets = [float(b) for b in args.budgets.split(",")]
    lnV = math.log(cfg.vocab_size)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _log(f"PARITY {args.preset}: steps{steps} budgets{budgets} "
         f"tok/step={P.tokens_per_step(cfg)}")

    ref = run_reference(cfg, recipe, stream, steps, seed=11,
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
            eng = run_engine(client, cfg, recipe, stream, steps,
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
            stream = make_stream(cfg.tokens)
            r = run_reference(cfg, recipe, stream, steps, seed=11,
                              grad_checkpoint=args.grad_checkpoint, log=_log)
            r.meta["preset"] = name
            r.meta["params"] = P.param_counts(cfg)
            r.save(RESULTS / f"scaling_{name}_reference.json")
    else:
        with daemon_client(slab_gib=args.slab, log=_log) as client:
            for name in presets:
                cfg = P.preset(name)
                stream = make_stream(cfg.tokens)
                r = run_engine(client, cfg, recipe, stream, steps,
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
    if args.opt:
        cfg = replace(cfg, opt_policy=args.opt)
    recipe = _recipe(args.steps, peak_lr=args.peak_lr)
    stream = make_stream(cfg.tokens)
    _log(f"REFERENCE-ONLY: {args.preset} opt={getattr(cfg, 'opt_policy', 'adamw')} "
         f"steps={args.steps} grad_checkpoint={args.grad_checkpoint}")
    res = run_reference(cfg, recipe, stream, args.steps,
                        grad_checkpoint=args.grad_checkpoint, log=_log)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.save(out)
    _log(f"saved {out} (final loss {res.losses[-1]:.4f})")
    return 0


def cmd_engine(args) -> int:
    """Engine-ONLY run (one daemon, one GPU): the service engine
    trained end to end, the curve saved for comparison against a
    reference yardstick (same seed/stream/recipe conventions).
    Long runs checkpoint host-locally and resume with --resume."""
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
    stream = make_stream(cfg.tokens)
    ck_dir = None
    if args.checkpoint_every:
        ck_dir = RESULTS / "checkpoints" / Path(args.out).stem
        ck_dir.mkdir(parents=True, exist_ok=True)
    _log(f"ENGINE-ONLY: {args.preset} opt={getattr(cfg, 'opt_policy', 'adamw')} "
         f"steps={args.steps} budget={args.budget}GiB "
         f"tokens/step={cfg.seq_len * cfg.batch * cfg.grad_accum_rounds} "
         f"ckpt={args.checkpoint_every or 'off'}")
    with daemon_client(slab_gib=args.slab, log=_log) as client:
        res = run_engine(client, cfg, recipe, stream, args.steps,
                         budget_gib=args.budget, seed=11, log=_log,
                         checkpoint_every=args.checkpoint_every,
                         checkpoint_dir=ck_dir, resume=args.resume)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.save(out)
    _log(f"saved {out} (final loss {res.losses[-1]:.4f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("engine")
    e.add_argument("--preset", default="l3_1b")
    e.add_argument("--steps", type=int, default=P.TRAIN_STEPS)
    e.add_argument("--opt", choices=["adamw", "muon"], default=None)
    e.add_argument("--peak-lr", type=float, default=3e-4)
    e.add_argument("--budget", type=float, default=14.0)
    e.add_argument("--slab", type=float, default=100.0)
    e.add_argument("--ga-rounds", type=int, default=None,
                   help="override grad-accum rounds (tokens/step = "
                        "seq*batch*rounds; 64 -> 512K at the locked "
                        "8192-token round)")
    e.add_argument("--checkpoint-every", type=int, default=None)
    e.add_argument("--resume", action="store_true")
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

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
