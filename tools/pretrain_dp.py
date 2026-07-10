"""Fleet DP pretraining: the two-box data-parallel twin of
tools/pretrain_run.py. Trains a ladder preset across chicago+tubingen
with weighted round distribution and compares the curve against the
recorded single-box run.

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--preset", default="l3_1b")
    tr.add_argument("--steps", type=int, default=1000)
    tr.add_argument("--rounds", default="6,2",
                    help="per-rank round counts (chicago,tubingen)")
    tr.add_argument("--budgets", default="14,12")
    tr.add_argument("--slabs", default="60,30")
    tr.add_argument("--peak-lr", type=float, default=3e-4)
    tr.add_argument("--seed", type=int, default=11)
    tr.add_argument("--out", required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)
    args = p.parse_args()

    if args.cmd == "train":
        cfg = preset(args.preset)
        recipe = Recipe(peak_lr=args.peak_lr, min_lr=args.peak_lr / 10,
                        warmup_steps=max(1, args.steps // 10),
                        total_steps=args.steps)
        rounds = tuple(int(x) for x in args.rounds.split(","))
        budgets = tuple(float(x) for x in args.budgets.split(","))
        slabs = tuple(float(x) for x in args.slabs.split(","))
        stream = make_stream(cfg.tokens)
        res = run_fleet_dp(cfg, recipe, stream, args.steps,
                           rank_rounds=rounds, budgets=budgets,
                           slabs=slabs, seed=args.seed)
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
