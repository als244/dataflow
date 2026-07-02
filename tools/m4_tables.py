"""Consolidate results/m4/*.summary.json into comparison tables (markdown).

Usage: python tools/m4_tables.py [--dir results/m4] > results/m4/README.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("results/m4"))
    args = parser.parse_args()

    by_config: dict[str, list[dict]] = defaultdict(list)
    baselines: dict[str, dict] = {}
    for path in sorted(args.dir.glob("*.summary.json")):
        data = json.loads(path.read_text())
        for row in data.get("sweep", []):
            row["_config"] = data["config"]
            row["_steps"] = data.get("steps")
            by_config[data["config"]].append(row)
        if "baseline_plain_torch_ms_per_step" in data:
            baselines[data["config"]] = data

    print("# M4 results: llama3-8B full-bf16 training, budget vs throughput\n")
    print("RTX 5090 (31.3 GiB) · bf16 params+grads+AdamW state · seq 4096 ·")
    print("measured task costs · plans built on measured bidirectional PCIe ·")
    print("steady-state excludes the warm-up step · full methodology in docs/m4-report.md\n")

    pretty = {
        "8b": "bs=1, ga=1 (4,096 tokens/step)",
        "8b-bs4ga4": "bs=4, ga=4 (65,536 tokens/step)",
        "8b-ga16": "bs=1, ga=16 (65,536 tokens/step)",
        "baseline1b": "1B-class baseline config",
        "mini": "mini smoke config",
    }

    for config in ("8b", "8b-bs4ga4", "8b-ga16", "baseline1b", "mini"):
        rows = by_config.get(config)
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["budget_gib"])
        print(f"\n## {config} — {pretty.get(config, '')}\n")
        print("| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | losses |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
        for r in rows:
            losses = ", ".join(f"{x:.3f}" for x in r["losses"])
            print(
                f"| {r['budget_gib']:g} | {r['sim_ms_per_step']:.0f} | "
                f"{r['real_ms_per_step_steady']:.0f} | {r['sim_tokens_per_s']:.0f} | "
                f"{r['real_tokens_per_s']:.0f} | {r['real_vs_sim_pct']:+.1f}% | "
                f"{r['replay_fidelity_gap_pct']:+.2f}% | "
                f"{r.get('recompute_chosen', 0)}/{32 * max(1, _rounds(r))} | {losses} |"
            )

    if baselines:
        print("\n## Plain-torch baseline (configs that fit in VRAM)\n")
        for config, data in baselines.items():
            base_tok = data["baseline_tokens_per_s"]
            ours = max(data["sweep"], key=lambda r: r["budget_gib"])
            print(
                f"- **{config}**: plain eager torch {data['baseline_plain_torch_ms_per_step']:.1f} ms/step "
                f"({base_tok:.0f} tok/s); this runtime at {ours['budget_gib']:g} GiB: "
                f"{ours['real_tokens_per_s']:.0f} tok/s "
                f"(**{ours['real_tokens_per_s'] / base_tok * 100:.0f}%** of plain torch)."
            )

    # cross-config comparison at shared budgets
    shared = ("8b", "8b-bs4ga4", "8b-ga16")
    if all(c in by_config for c in shared):
        print("\n## Batching comparison (real tok/s by budget)\n")
        budgets = sorted({r["budget_gib"] for c in shared for r in by_config[c]})
        print("| budget (GiB) | " + " | ".join(pretty[c] for c in shared) + " |")
        print("|---:|" + "---:|" * len(shared))
        for b in budgets:
            cells = []
            for c in shared:
                match = [r for r in by_config[c] if r["budget_gib"] == b]
                cells.append(f"{match[0]['real_tokens_per_s']:.0f}" if match else "—")
            print(f"| {b:g} | " + " | ".join(cells) + " |")


def _rounds(row: dict) -> int:
    return {
        "8b": 1, "8b-bs4ga4": 4, "8b-ga16": 16, "baseline1b": 1, "mini": 1,
    }.get(row.get("_config", ""), 1)


if __name__ == "__main__":
    main()
