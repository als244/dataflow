#!/usr/bin/env python
"""Ladder-3 v2 measurement sweep: every family x {uniform, ragged},
envelope-free, reporting the dW-space gradient comparison (the sharp
instrument) alongside loss and final-param entries.

Per case: loss rel, worst gradient fields (rel_l2 + cosine), gradient
median, worst final-param fields. Ends with a summary matrix. Exit 0
always — measurement tool, not a gate.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

FAMILIES = ("gpt2", "llama3", "qwen3", "olmoe", "dsv3", "dsv32", "glm52",
            "qwen3moe", "qwen35", "qwen35moe")
SHAPES = ("uniform", "ragged")


def ragged_partition(cfg):
    t = cfg.seq_len * cfg.batch
    a = t // 2 + 3
    b = t // 4 + 1
    return (a, b, t - a - b)


def tiny_cfg(family: str):
    mod = importlib.import_module(f"dataflow_training.model_families.{family}")
    cfg_cls = next(v for k, v in vars(mod).items()
                   if k.startswith("Shaped") and k.endswith("Config"))
    return cfg_cls.tiny()


def stats_of(report):
    import statistics

    grads = {k[5:]: v for k, v in report.errors.items()
             if k.startswith("grad:")}
    gcos = {k[5:]: v for k, v in report.cosines.items()
            if k.startswith("grad:")}
    params = {k: v for k, v in report.errors.items()
              if not k.startswith("grad:") and k != "loss"}
    return {
        "loss": report.errors.get("loss", float("nan")),
        "grads": grads,
        "gcos": gcos,
        "gmed": statistics.median(grads.values()) if grads else float("nan"),
        "gworst": max(grads.items(), key=lambda kv: kv[1])
                  if grads else ("-", float("nan")),
        "cworst": min(gcos.items(), key=lambda kv: kv[1])
                  if gcos else ("-", float("nan")),
        "pworst": max(params.items(), key=lambda kv: kv[1])
                  if params else ("-", float("nan")),
    }


def main() -> int:
    from dataflow_training.testing.gradcheck import check_model_step

    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=None)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    fams = tuple(args.families.split(",")) if args.families else FAMILIES

    matrix = {}
    for family in fams:
        for shape in SHAPES:
            cfg = tiny_cfg(family)
            if shape == "ragged":
                cfg = replace(cfg, seq_lens=ragged_partition(cfg))
            label = f"{family}/{shape}"
            try:
                report = check_model_step(
                    cfg, fast_memory_capacity=96 * 1024 * 1024, tol=3e-2)
                s = stats_of(report)
                matrix[label] = s
                print(f"== {label}: loss_rel={s['loss']:.2e} "
                      f"grad median={s['gmed']:.3e}")
                worst = sorted(s["grads"].items(), key=lambda kv: -kv[1])
                for name, err in worst[:args.top]:
                    print(f"    grad {err:10.3e}  cos={s['gcos'][name]:.6f}"
                          f"  {name}")
                pn, pv = s["pworst"]
                print(f"    param worst {pv:10.3e}  {pn}")
            except Exception:
                print(f"== {label}: EXCEPTION")
                traceback.print_exc()
                matrix[label] = None
            sys.stdout.flush()

    print("\n==== summary matrix (grad space) ====")
    print(f"{'case':22s} {'loss_rel':>9s} {'grad_med':>9s} "
          f"{'grad_worst':>10s} {'worst_field':28s} {'min_cos':>8s}")
    for label, s in matrix.items():
        if s is None:
            print(f"{label:22s} {'EXCEPTION':>9s}")
            continue
        gn, gv = s["gworst"]
        cn, cv = s["cworst"]
        print(f"{label:22s} {s['loss']:9.2e} {s['gmed']:9.3e} "
              f"{gv:10.3e} {gn[:28]:28s} {cv:8.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
