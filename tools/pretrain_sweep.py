#!/usr/bin/env python
"""Single-GPU vs distributed comparison runner + sweeps.

For each config (preset, global batch tokens, steps, budgets,
backing) this runs the SAME training twice — once on the local
daemon, once data-parallel across the topology group with the 3:1
round split — and saves per-step throughput, walls, and the full
loss curve for both. T_round is held at the preset's per-round
geometry (seq_len x batch tokens); the global batch sets the ROUND
COUNT only.

    python tools/pretrain_sweep.py --preset l3_1b \
        --global-tokens 32K,64K,128K,256K,512K --steps 10 \
        --budgets 16,16 --backing 60,60 \
        --out-dir results/pretrain/sweeps/l3_1b_batch

Prints a comparison table over all configs at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dataflow.pretrain.driver import daemon_client, load_result, run_engine
from dataflow.pretrain.fineweb import make_stream
from dataflow.pretrain.fleet import run_fleet_dp
from dataflow.pretrain.presets import preset
from dataflow.pretrain.recipe import Recipe
from dataflow.pretrain.topology import load_topology


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
        stream = make_stream(cfg.tokens)
        t0 = time.monotonic()
        with daemon_client(slab_gib=backing[0]) as client:
            res_s = run_engine(client, cfg, recipe, stream, steps,
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
        stream = make_stream(cfg.tokens)
        t0 = time.monotonic()
        res_d = run_fleet_dp(cfg, recipe, stream, steps,
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="l3_1b")
    ap.add_argument("--global-tokens", default="32K,64K,128K,256K,512K")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--budgets", default="16,16",
                    help="fast GiB: single+rank0, rank1")
    ap.add_argument("--backing", default="60,60",
                    help="host slab GiB: single+rank0, rank1")
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--topology", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg_base = preset(args.preset)
    budgets = [float(x) for x in args.budgets.split(",")]
    backing = [float(x) for x in args.backing.split(",")]
    topo = load_topology(args.topology)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tokens_global in parse_tokens(args.global_tokens):
        row = run_config(cfg_base, tokens_global, args.steps, budgets,
                         backing, topo, args.seed, args.peak_lr,
                         out_dir)
        rows.append(row)
        with open(out_dir / "sweep.json", "w") as fh:
            json.dump({"preset": args.preset, "steps": args.steps,
                       "budgets": budgets, "backing": backing,
                       "rows": rows}, fh, indent=1)
        print_table(rows, args.steps)
    print(f"\nsaved {out_dir}/sweep.json")


if __name__ == "__main__":
    main()
