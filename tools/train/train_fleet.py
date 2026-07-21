"""Fleet DP pretraining: the data-parallel twin of
tools/train/train_solo.py. Trains a ladder preset across a topology
group's hosts (topology.toml — see topology.example.toml) with
weighted round distribution and compares the curve against a recorded
single-box run.

    python tools/train/train_fleet.py train --preset l3_1b --steps 1000 \
        --rounds 6,2 --out results/pretrain/l3_1b_dp.json
    python tools/train/train_fleet.py compare \
        --a results/pretrain/l3_1b_engine_14gib.json \
        --b results/pretrain/l3_1b_dp.json
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataflow_training.run import parity
from dataflow_training.run.driver import daemon_client, load_result, run_engine
from dataflow_training.data.pipeline import legacy_block_pipeline
from dataflow_training.distributed.fleet import run_fleet_dp
from dataflow_training.run.presets import preset
from dataflow_training.run.recipe import Recipe
from dataflow_training.distributed.topology import load_topology


def floats_or_none(raw: str):
    if not raw:
        return None
    return tuple(float(x) for x in raw.split(","))


def parse_tokens(spec: str) -> list:
    out = []
    for part in spec.split(","):
        part = part.strip().upper()
        if part.endswith("K"):
            out.append(int(part[:-1]) * 1024)
        elif part.endswith("M"):
            out.append(int(part[:-1]) * 1024 * 1024)
        else:
            out.append(int(part))
    return out


def fmt_tokens(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def run_config(cfg_base, tokens_global: int, steps: int, budgets,
               backing, topo, seed: int, peak_lr: float,
               out_dir: Path) -> dict:
    t_round = cfg_base.seq_len * cfg_base.batch
    if tokens_global % t_round:
        raise ValueError(f"{tokens_global} tokens not divisible by the "
                         f"preset round of {t_round}")
    rounds = tokens_global // t_round
    if rounds % 4:
        raise ValueError(f"{rounds} rounds not divisible by 4 (needed "
                         f"for the 3:1 split)")
    cfg = replace(cfg_base, grad_accum_rounds=rounds)
    recipe = Recipe(peak_lr=peak_lr, min_lr=peak_lr / 10,
                    warmup_steps=max(1, steps // 10), total_steps=steps)
    tag = fmt_tokens(tokens_global)
    row = {"tokens": tokens_global, "rounds": rounds}

    single_path = out_dir / f"single_{tag}.json"
    if single_path.exists():
        print(f"[sweep] {tag}: single exists, skipping")
        res_s = load_result(single_path)
    else:
        print(f"[sweep] {tag}: SINGLE ({rounds} rounds x {t_round} tok)")
        feed = legacy_block_pipeline(cfg)
        t0 = time.monotonic()
        with daemon_client(slab_gib=backing[0]) as client:
            res_s = run_engine(client, cfg, recipe, feed, steps,
                               budget_gib=budgets[0], seed=seed,
                               log_every=max(1, steps // 2))
        res_s.meta["bringup_plus_train_wall_s"] = round(
            time.monotonic() - t0, 2)
        res_s.save(single_path)

    dist_path = out_dir / f"dist_{tag}.json"
    if dist_path.exists():
        print(f"[sweep] {tag}: dist exists, skipping")
        res_d = load_result(dist_path)
    else:
        split = (rounds * 3 // 4, rounds // 4)
        print(f"[sweep] {tag}: DIST rounds {split[0]}:{split[1]}")
        feed = legacy_block_pipeline(cfg)
        t0 = time.monotonic()
        res_d = run_fleet_dp(cfg, recipe, feed, steps,
                             rank_rounds=split, budgets=tuple(budgets),
                             slabs=tuple(backing), topology=topo,
                             seed=seed, log_every=max(1, steps // 2))
        res_d.meta["bringup_plus_train_wall_s"] = round(
            time.monotonic() - t0, 2)
        res_d.save(dist_path)

    row["single"] = summarize(res_s)
    row["dist"] = summarize(res_d)
    row["loss_max_abs_delta"] = round(
        max(abs(a - b) for a, b in zip(res_s.losses, res_d.losses)), 5)
    row["speedup"] = round(row["dist"]["steady_tok_s"]
                           / row["single"]["steady_tok_s"], 3)
    return row


def summarize(res) -> dict:
    walls = res.step_wall_s
    return {"steady_tok_s": round(res.steady_tok_per_s, 1),
            "train_wall_s": round(sum(walls), 2),
            "step_wall_s": [round(w, 3) for w in walls],
            "final_loss": round(res.losses[-1], 4),
            "losses": [round(x, 4) for x in res.losses]}


def print_table(rows, steps: int) -> None:
    print(f"\n=== single vs distributed (3:1), {steps} steps ===")
    hdr = (f"{'batch':>7} {'rounds':>6} | {'1gpu tok/s':>10} "
           f"{'1gpu wall':>9} | {'dist tok/s':>10} {'dist wall':>9} | "
           f"{'speedup':>7} {'max|dLoss|':>10} {'final 1gpu/dist':>17}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s, d = r["single"], r["dist"]
        print(f"{fmt_tokens(r['tokens']):>7} {r['rounds']:>6} | "
              f"{s['steady_tok_s']:>10.0f} {s['train_wall_s']:>8.1f}s | "
              f"{d['steady_tok_s']:>10.0f} {d['train_wall_s']:>8.1f}s | "
              f"{r['speedup']:>6.2f}x {r['loss_max_abs_delta']:>10.5f} "
              f"{s['final_loss']:>8.4f}/{d['final_loss']:<8.4f}")




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
    tr.add_argument("--execute-padding", action="store_true",
                    help="execute under-full rounds' buffer tails "
                         "(masked; REQUIRED with --tp-mlp when rounds "
                         "can be under-full)")
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
    sw = sub.add_parser("sweep", help="single-GPU vs distributed (3:1) "
                                      "comparison sweep")
    sw.add_argument("--preset", default="l3_1b")
    sw.add_argument("--global-tokens", default="32K,64K,128K,256K,512K")
    sw.add_argument("--steps", type=int, default=10)
    sw.add_argument("--budgets", default="16,16",
                    help="fast GiB: single+rank0, rank1")
    sw.add_argument("--backing", default="60,60",
                    help="host slab GiB: single+rank0, rank1")
    sw.add_argument("--peak-lr", type=float, default=3e-4)
    sw.add_argument("--seed", type=int, default=11)
    sw.add_argument("--topology", default=None)
    sw.add_argument("--out-dir", required=True)
    args = p.parse_args()

    if args.cmd == "sweep":
        import json as json_mod

        cfg_base = preset(args.preset)
        budgets = [float(x) for x in args.budgets.split(",")]
        backing = [float(x) for x in args.backing.split(",")]
        topo = load_topology(args.topology)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for tokens_global in parse_tokens(args.global_tokens):
            row = run_config(cfg_base, tokens_global, args.steps,
                             budgets, backing, topo, args.seed,
                             args.peak_lr, out_dir)
            rows.append(row)
            with open(out_dir / "sweep.json", "w") as fh:
                json_mod.dump({"preset": args.preset, "steps": args.steps,
                               "budgets": budgets, "backing": backing,
                               "rows": rows}, fh, indent=1)
            print_table(rows, args.steps)
        print(f"\nsaved {out_dir}/sweep.json")
        return

    if args.cmd == "train":
        cfg = preset(args.preset)
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
        feed = legacy_block_pipeline(cfg)
        profile = None
        if args.profile:
            profile = {"start": args.profile_start_before_step,
                       "stop": args.profile_stop_after_step}
        res = run_fleet_dp(cfg, recipe, feed, args.steps,
                           rank_rounds=rounds,
                           budgets=floats_or_none(args.budgets),
                           slabs=floats_or_none(args.slabs),
                           topology=load_topology(args.topology),
                           group=args.group, attach=attach,
                           seed=args.seed, profile=profile,
                           backend=args.backend,
                           opt_shard=args.opt_shard,
                           tp_mlp=args.tp_mlp,
                           execute_padding=args.execute_padding,
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
