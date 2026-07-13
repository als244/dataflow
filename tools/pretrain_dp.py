"""Fleet DP pretraining: the data-parallel twin of
tools/pretrain_run.py. Trains a ladder preset across a topology
group's hosts (topology.toml — see topology.example.toml) with
weighted round distribution and compares the curve against a recorded
single-box run.

    python tools/pretrain_dp.py train --preset l3_1b --steps 1000 \
        --rounds 6,2 --out results/pretrain/l3_1b_dp.json
    python tools/pretrain_dp.py compare \
        --a results/pretrain/l3_1b_engine_14gib.json \
        --b results/pretrain/l3_1b_dp.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataflow.pretrain import parity
from dataflow.pretrain.driver import load_result
from dataflow.pretrain.fineweb import make_stream
from dataflow.pretrain.fleet import run_fleet_dp
from dataflow.pretrain.presets import preset
from dataflow.pretrain.recipe import Recipe
from dataflow.pretrain.topology import load_topology


def floats_or_none(raw: str):
    if not raw:
        return None
    return tuple(float(x) for x in raw.split(","))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--preset", default="l3_1b")
    tr.add_argument("--steps", type=int, default=1000)
    tr.add_argument("--topology", default=None,
                    help="topology.toml path (default: cwd, repo root)")
    tr.add_argument("--group", default="dp",
                    help="topology group to train across")
    tr.add_argument("--rounds", default="6,2",
                    help="per-rank round counts, one per group member "
                         "(rank order = member order)")
    tr.add_argument("--budgets", default="",
                    help="per-rank device budgets GiB "
                         "(default: topology budget_gib)")
    tr.add_argument("--slabs", default="",
                    help="per-rank host slabs GiB "
                         "(default: topology slab_gib)")
    tr.add_argument("--attach", action="append", default=[],
                    metavar="HOST=SOCK",
                    help="attach to a pre-launched daemon instead of "
                         "launching one (repeatable; lifecycle stays "
                         "with the caller)")
    tr.add_argument("--peak-lr", type=float, default=3e-4)
    tr.add_argument("--seed", type=int, default=11)
    tr.add_argument("--out", required=True)
    tr.add_argument("--backend", default=None,
                    help="group backend override: hostmem | nccl | "
                         "auto (default: the topology group's)")
    tr.add_argument("--tp-mlp", action="store_true",
                    help="tensor-parallel MLPs through the sharding "
                         "API: every rank runs the FULL batch with "
                         "w1/w3/w2 sharded over d_ff (rank_rounds "
                         "does not apply); correctness track, not a "
                         "throughput one on this pair")
    tr.add_argument("--checkpoint-every", type=int, default=None,
                    help="fleet checkpoint every N steps (per-rank "
                         "host-local snapshots + a conductor fleet "
                         "manifest written last as the completeness "
                         "marker)")
    tr.add_argument("--checkpoint-redundancy", type=int, default=1,
                    help="shared-artifact copies on distinct hosts")
    tr.add_argument("--checkpoint-keep-last", type=int, default=0,
                    help="prune all but the newest K checkpoints "
                         "(0 = keep everything)")
    tr.add_argument("--resume", default=None,
                    help="'auto' (newest complete checkpoint for this "
                         "--out run name) or a step directory path")
    tr.add_argument("--opt-shard", default=None,
                    help="optimizer-state sharding: 'zero1' (field-"
                         "snapped shards, per-bucket reduce+broadcast) "
                         "or 'zero1rs' (byte-equal shards, ONE "
                         "reduce_scatter + ONE all_gather per object — "
                         "bandwidth-optimal at any world; needs a "
                         "uniform adamw policy)")
    tr.add_argument("--dp-overlap", action="store_true",
                    help="EXPERIMENTAL, known-broken at scale: tail "
                         "optimizers on PRE-REDUCED grads (grad_reduce "
                         "tasks overlap the exchange with backward). "
                         "Bitwise-correct at one step; NaNs under memory "
                         "pressure — needs the completion-stream engine "
                         "extension (findings) before real use")
    tr.add_argument("--profile", action="store_true",
                    help="bracket steps with the vendor capture API; "
                         "launched daemons are wrapped in the canonical "
                         "nsys command and remote reports are fetched "
                         "back; training CONTINUES after the window")
    tr.add_argument("--profile-start-before-step", type=int, default=1)
    tr.add_argument("--profile-stop-after-step", type=int, default=3)
    cp = sub.add_parser("compare")
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)
    args = p.parse_args()

    if args.cmd == "train":
        cfg = preset(args.preset)
        if args.dp_overlap:
            from dataclasses import replace as dc_replace

            cfg = dc_replace(cfg, optimizer_placement="tail")
        recipe = Recipe(peak_lr=args.peak_lr, min_lr=args.peak_lr / 10,
                        warmup_steps=max(1, args.steps // 10),
                        total_steps=args.steps)
        rounds = tuple(int(x) for x in args.rounds.split(","))
        attach = {}
        for item in args.attach:
            name, sep, sock = item.partition("=")
            if not sep or not sock:
                p.error(f"--attach wants HOST=SOCK, got {item!r}")
            attach[name] = sock
        stream = make_stream(cfg.tokens)
        profile = None
        if args.profile:
            profile = {"start": args.profile_start_before_step,
                       "stop": args.profile_stop_after_step}
        res = run_fleet_dp(cfg, recipe, stream, args.steps,
                           rank_rounds=rounds,
                           budgets=floats_or_none(args.budgets),
                           slabs=floats_or_none(args.slabs),
                           topology=load_topology(args.topology),
                           group=args.group, attach=attach,
                           seed=args.seed, profile=profile,
                           dp_overlap=args.dp_overlap,
                           backend=args.backend,
                           opt_shard=args.opt_shard,
                           tp_mlp=args.tp_mlp,
                           checkpoint_every=args.checkpoint_every,
                           checkpoint_redundancy=args.checkpoint_redundancy,
                           checkpoint_keep_last=args.checkpoint_keep_last,
                           run_name=Path(args.out).stem,
                           resume=args.resume)
        res.save(args.out)
        print(f"saved {args.out} (final loss {res.losses[-1]:.4f}, "
              f"steady {res.steady_tok_per_s:.0f} tok/s)")
    else:
        a = load_result(args.a)
        b = load_result(args.b)
        rep = parity.compare(a.losses, b.losses)
        print(rep.summary())
        print("PASS" if rep.passed else "FAIL")


if __name__ == "__main__":
    main()
