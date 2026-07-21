#!/usr/bin/env python
"""THE training tool: one entry point for every world size.

Subcommands:
  train      engine training — world-1 by default (zero-config: no
             topology flags means one local child daemon); a topology
             group + per-rank round split makes it a DP fleet. zero1rs
             is the DP default at world > 1 (--opt-shard co is the
             co-responsible diagnostic lane). --profile wraps every
             launched daemon in the canonical nsys command.
  reference  pure-torch twin long run (daemon-less)
  smoke      tiny real-vocab model, reference vs engine — the infra gate
  parity     one preset, reference + engine at N budgets
  scaling    the preset ladder on one backend
  peek       read an in-flight run's loss curve from its checkpoints
  compare    overlay two finished run curves

Budget flags are memory-tier named and comma-separated per rank:
  --fast-budget    device fast memory GiB (one value per rank)
  --backing-budget host memory GiB per rank — drives the daemon's
                   pinned slab AND refuses plans whose backing peak
                   would not fit it

Checkpoints are manifest v2 at every world size (fleet.json:
responsibility save plan, launch record, per-rank planned programs);
resume with --resume auto|<step dir>.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataflow_training.data.pipeline import (  # noqa: E402
    legacy_block_pipeline,
    pipeline_from_args,
)
from dataflow_training.run import parity, presets as P  # noqa: E402
from dataflow_training.run.driver import (  # noqa: E402
    daemon_client,
    init_model,
    run_engine,
    run_reference,
)
from dataflow_training.run.recipe import Recipe  # noqa: E402

RESULTS = ROOT / "results" / "pretrain"
CKPTS = RESULTS / "checkpoints"


def log_line(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def recipe_for(steps: int, *, peak_lr: float = 3e-4,
               muon_lr: float | None = None) -> Recipe:
    return Recipe(peak_lr=peak_lr, min_lr=peak_lr / 10,
                  warmup_steps=max(1, steps // 10), total_steps=steps,
                  muon_lr=muon_lr)


def pipeline_for(cfg, args):
    return pipeline_from_args(
        cfg, args.data, policy=args.packing_policy,
        allow_round_split=args.allow_round_split,
        capture=args.capture)


def per_rank_floats(raw: str, world: int, flag: str) -> tuple:
    vals = [float(v) for v in raw.split(",")]
    if len(vals) == 1 and world > 1:
        vals = vals * world
    if len(vals) != world:
        raise SystemExit(f"{flag} got {len(vals)} value(s) for "
                         f"world {world}")
    return tuple(vals)


def cfg_with_overrides(args):
    cfg = P.resolve_preset(args.preset)
    overrides = {}
    if getattr(args, "opt", None):
        overrides["opt_policy"] = args.opt
    if getattr(args, "ga_rounds", None):
        overrides["grad_accum_rounds"] = args.ga_rounds
    if getattr(args, "batch", None):
        overrides["batch"] = args.batch
    return replace(cfg, **overrides) if overrides else cfg


# --------------------------------------------------------------- train
def cmd_train(args) -> int:
    from dataflow_training.distributed.fleet import (
        local_topology,
        run_fleet_dp,
    )
    from dataflow_training.distributed.topology import load_topology

    cfg = cfg_with_overrides(args)
    if args.topology or args.group:
        topo = load_topology(args.topology)
        group = args.group or "dp"
        world = len(topo.group(group).members)
        rank_rounds = (tuple(int(r) for r in args.rounds.split(","))
                       if args.rounds else None)
        if rank_rounds is None:
            raise SystemExit("fleet mode needs --rounds (per-rank "
                             "round split summing to ga_rounds)")
    else:
        topo, group, world = None, "local", 1
        rank_rounds = (cfg.grad_accum_rounds,)
    fast = per_rank_floats(args.fast_budget, world, "--fast-budget")
    backing = per_rank_floats(args.backing_budget, world,
                              "--backing-budget")
    if topo is None:
        topo = local_topology(budget_gib=fast[0], slab_gib=backing[0])

    recipe = recipe_for(args.steps, peak_lr=args.peak_lr,
                        muon_lr=args.muon_lr)
    feed = pipeline_for(cfg, args)
    profile = None
    if args.profile:
        profile = {"start": args.profile_start_before_step,
                   "stop": args.profile_stop_after_step}
    run_name = Path(args.out).stem
    log_line(f"TRAIN world={world} {args.preset} "
             f"opt={getattr(cfg, 'opt_policy', 'adamw')} "
             f"steps={args.steps} fast={fast} backing={backing} "
             f"data={args.data or 'default'} "
             f"ckpt={args.checkpoint_every or 'off'}")
    res = run_fleet_dp(
        cfg, recipe, feed, args.steps,
        rank_rounds=rank_rounds, budgets=fast, slabs=backing,
        topology=topo, group=group, seed=args.seed, log=log_line,
        profile=profile, backend=args.backend,
        opt_shard=args.opt_shard, tp_mlp=args.tp_mlp,
        execute_padding=args.execute_padding,
        launch_argv=sys.argv,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=str(CKPTS), run_name=run_name,
        checkpoint_redundancy=args.checkpoint_redundancy,
        checkpoint_keep_last=args.checkpoint_keep_last,
        resume=args.resume)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.save(out)
    log_line(f"saved {out} (final loss {res.losses[-1]:.4f})")
    return 0


# ----------------------------------------------------------- reference
def cmd_reference(args) -> int:
    cfg = cfg_with_overrides(args)
    recipe = recipe_for(args.steps, peak_lr=args.peak_lr)
    feed = pipeline_for(cfg, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ck_dir = None
    partial = None
    if args.checkpoint_every:
        ck_dir = CKPTS / out.stem
        ck_dir.mkdir(parents=True, exist_ok=True)
        partial = out.with_name(out.stem + "_partial.json")
    log_line(f"REFERENCE: {args.preset} "
             f"opt={getattr(cfg, 'opt_policy', 'adamw')} "
             f"steps={args.steps} data={args.data or 'default'}")
    res = run_reference(cfg, recipe, feed, args.steps,
                        grad_checkpoint=args.grad_checkpoint,
                        checkpoint_every=args.checkpoint_every,
                        checkpoint_dir=ck_dir, resume=args.resume,
                        partial_out=partial, log=log_line)
    res.save(out)
    log_line(f"saved {out} (final loss {res.losses[-1]:.4f})")
    return 0


# --------------------------------------------------------------- smoke
def init_bytes_identical(cfg, client, seed: int) -> bool:
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
    recipe = recipe_for(args.steps)
    feed = legacy_block_pipeline(cfg)
    lnV = math.log(cfg.vocab_size)
    log_line(f"SMOKE: {cfg.n_layers}L d{cfg.d_model} "
             f"vocab{cfg.vocab_size} steps{args.steps} ln(V)={lnV:.3f}")
    ref = run_reference(cfg, recipe, feed, args.steps, seed=11,
                        log=log_line)
    with daemon_client(slab_gib=float(args.backing_budget),
                       log=log_line) as client:
        init_model(client, "llama3", P.cfg_dict(cfg), seed=11)
        identical = init_bytes_identical(cfg, client, seed=11)
        log_line(f"init byte-identity (daemon vs reference): {identical}")
        eng = run_engine(client, cfg, recipe, feed, args.steps,
                         budget_gib=float(args.fast_budget), seed=11,
                         log=log_line)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ref.save(RESULTS / "smoke_reference.json")
    eng.save(RESULTS / "smoke_engine.json")
    okr, msgr = parity.curves_healthy(ref.losses, expect_start=lnV,
                                      min_drop=0.2)
    oke, msge = parity.curves_healthy(eng.losses, expect_start=lnV,
                                      min_drop=0.2)
    log_line(f"reference health: {okr} ({msgr})")
    log_line(f"engine    health: {oke} ({msge})")
    rep = parity.compare(ref.losses, eng.losses, a_label="reference",
                         b_label="engine")
    log_line(rep.summary())
    passed = identical and okr and oke and rep.passed
    log_line(f"SMOKE {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


# -------------------------------------------------------------- parity
def cmd_parity(args) -> int:
    cfg = P.resolve_preset(args.preset)
    recipe = recipe_for(args.steps)
    feed = legacy_block_pipeline(cfg)
    budgets = [float(b) for b in args.fast_budget.split(",")]
    lnV = math.log(cfg.vocab_size)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_line(f"PARITY {args.preset}: steps{args.steps} "
             f"budgets{budgets}")
    ref = run_reference(cfg, recipe, feed, args.steps, seed=11,
                        grad_checkpoint=args.grad_checkpoint,
                        log=log_line)
    ref.save(RESULTS / f"{args.preset}_reference.json")
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    engs = {}
    with daemon_client(slab_gib=float(args.backing_budget),
                       log=log_line) as client:
        for b in budgets:
            eng = run_engine(client, cfg, recipe, feed, args.steps,
                             budget_gib=b, seed=11, log=log_line)
            eng.save(RESULTS / f"{args.preset}_engine_{b:g}gib.json")
            engs[b] = eng
    ok, msgr = parity.curves_healthy(ref.losses, expect_start=lnV)
    log_line(f"reference health: {ok} ({msgr})")
    for b, eng in engs.items():
        rep = parity.compare(ref.losses, eng.losses,
                             a_label="reference",
                             b_label=f"engine@{b:g}GiB")
        log_line(rep.summary())
        ok = ok and rep.passed
    if len(budgets) == 2:
        b0, b1 = budgets
        rep = parity.compare(engs[b0].losses, engs[b1].losses,
                             a_label=f"engine@{b0:g}GiB",
                             b_label=f"engine@{b1:g}GiB")
        log_line("budget-invariance " + rep.summary())
    return 0 if ok else 1


# ------------------------------------------------------------- scaling
def cmd_scaling(args) -> int:
    recipe = recipe_for(args.steps)
    names = args.presets.split(",")
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_line(f"SCALING presets{names} backend={args.backend} "
             f"steps{args.steps}")
    if args.backend == "reference":
        for name in names:
            cfg = P.preset(name)
            feed = legacy_block_pipeline(cfg)
            r = run_reference(cfg, recipe, feed, args.steps, seed=11,
                              grad_checkpoint=args.grad_checkpoint,
                              log=log_line)
            r.meta["preset"] = name
            r.meta["params"] = P.param_counts(cfg)
            r.save(RESULTS / f"scaling_{name}_reference.json")
    else:
        with daemon_client(slab_gib=float(args.backing_budget),
                           log=log_line) as client:
            for name in names:
                cfg = P.preset(name)
                feed = legacy_block_pipeline(cfg)
                r = run_engine(client, cfg, recipe, feed, args.steps,
                               budget_gib=float(args.fast_budget),
                               seed=11, log=log_line)
                r.meta["preset"] = name
                r.meta["params"] = P.param_counts(cfg)
                r.save(RESULTS / f"scaling_{name}_engine.json")
                client.wipe("all", force=True)
    return 0


# ---------------------------------------------------------------- peek
def cmd_peek(args) -> int:
    run_dir = CKPTS / args.run
    manifests = sorted(run_dir.glob("step_*/fleet.json"))
    if manifests:
        manifest = json.loads(manifests[-1].read_text())
        losses = [float(x) for x in manifest.get("losses", [])]
        step = manifest["step"]
        source = str(manifests[-1])
    else:
        # engine-local layout (reference legs, library runs)
        legacy = sorted(run_dir.glob("step_*/manifest.json"))
        if not legacy:
            print(f"no complete checkpoints under {run_dir}",
                  file=sys.stderr)
            return 1
        meta = json.loads(legacy[-1].read_text()).get("client_meta", {})
        losses = [float(x) for x in meta.get("losses", [])]
        step = meta.get("step")
        source = str(legacy[-1])
    if not losses:
        print(f"{source} carries no loss curve", file=sys.stderr)
        return 1
    ema_v = losses[0]
    for x in losses:
        ema_v = args.ema * ema_v + (1 - args.ema) * x
    out = RESULTS / f"{args.run}_partial.json"
    out.write_text(json.dumps({
        "backend": "engine", "partial_through_step": int(step),
        "losses": losses, "meta": {"source": source},
    }, indent=2))
    print(f"{args.run}: {len(losses)} steps recorded (through {step})")
    print(f"  last loss {losses[-1]:.4f}   EMA({args.ema}) {ema_v:.4f}"
          f"   min {min(losses):.4f}")
    print(f"  partial curve -> {out}")
    return 0


# ------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    rep = parity.compare(a["losses"], b["losses"],
                         a_label=Path(args.a).stem,
                         b_label=Path(args.b).stem)
    print(rep.summary())
    return 0 if rep.passed else 1


# ---------------------------------------------------------------- CLI
def add_data_flags(p) -> None:
    p.add_argument("--data", default=None,
                   help="data source spec (docs/data_feeds.md); "
                        "default: the in-repo shard corpus, per-doc")
    p.add_argument("--packing-policy", choices=["ffd", "greedy"],
                   default="ffd")
    p.add_argument("--allow-round-split", action="store_true")
    p.add_argument("--capture", default=None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="engine training, any world size")
    t.add_argument("--preset", default="gpt2_124m")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--peak-lr", type=float, default=3e-4)
    t.add_argument("--muon-lr", type=float, default=None)
    t.add_argument("--opt", choices=["adamw", "muon"], default=None)
    t.add_argument("--ga-rounds", type=int, default=None)
    t.add_argument("--batch", type=int, default=None)
    t.add_argument("--seed", type=int, default=11)
    t.add_argument("--fast-budget", default="8",
                   help="device fast GiB, comma per rank")
    t.add_argument("--backing-budget", default="8",
                   help="host GiB per rank (pinned slab + plan cap)")
    t.add_argument("--topology", default=None)
    t.add_argument("--group", default=None)
    t.add_argument("--rounds", default=None,
                   help="per-rank round split (fleet mode)")
    t.add_argument("--backend", default=None,
                   help="hostmem | nccl | auto (fleet)")
    t.add_argument("--opt-shard",
                   choices=["zero1", "zero1rs", "co"], default=None,
                   help="default: zero1rs at world>1; co = the "
                        "co-responsible diagnostic lane")
    t.add_argument("--tp-mlp", action="store_true")
    t.add_argument("--execute-padding", action="store_true")
    t.add_argument("--profile", action="store_true",
                   help="wrap every launched daemon in the canonical "
                        "nsys command; bracket with the step flags")
    t.add_argument("--profile-start-before-step", type=int, default=None)
    t.add_argument("--profile-stop-after-step", type=int, default=None)
    t.add_argument("--checkpoint-every", type=int, default=None)
    t.add_argument("--checkpoint-redundancy", type=int, default=1)
    t.add_argument("--checkpoint-keep-last", type=int, default=0)
    t.add_argument("--resume", default=None,
                   help="auto | <step dir>")
    add_data_flags(t)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("reference", help="pure-torch twin leg")
    r.add_argument("--preset", required=True)
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--peak-lr", type=float, default=3e-4)
    r.add_argument("--opt", choices=["adamw", "muon"], default=None)
    r.add_argument("--ga-rounds", type=int, default=None)
    r.add_argument("--grad-checkpoint", action="store_true")
    r.add_argument("--checkpoint-every", type=int, default=None)
    r.add_argument("--resume", action="store_true")
    add_data_flags(r)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_reference)

    s = sub.add_parser("smoke", help="reference-vs-engine infra gate")
    s.add_argument("--steps", type=int, default=30)
    s.add_argument("--fast-budget", default="4")
    s.add_argument("--backing-budget", default="6")
    s.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("parity", help="reference + engine at N budgets")
    p.add_argument("--preset", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--fast-budget", required=True,
                   help="comma list of device budgets to sweep")
    p.add_argument("--backing-budget", default="8")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.set_defaults(fn=cmd_parity)

    sc = sub.add_parser("scaling", help="preset ladder on one backend")
    sc.add_argument("--presets", required=True)
    sc.add_argument("--backend", choices=["reference", "engine"],
                    default="engine")
    sc.add_argument("--steps", type=int, default=300)
    sc.add_argument("--fast-budget", default="8")
    sc.add_argument("--backing-budget", default="8")
    sc.add_argument("--grad-checkpoint", action="store_true")
    sc.set_defaults(fn=cmd_scaling)

    pk = sub.add_parser("peek", help="in-flight curve from checkpoints")
    pk.add_argument("run")
    pk.add_argument("--ema", type=float, default=0.98)
    pk.set_defaults(fn=cmd_peek)

    c = sub.add_parser("compare", help="overlay two finished runs")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
