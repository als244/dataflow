"""Consolidate results/m4/*.summary.json into comparison tables (markdown).

Usage:
    python tools/m4_tables.py --dir results/m4/fused-v1 > .../README.md
    python tools/m4_tables.py --compare results/m4/eager-v1 results/m4/fused-v1
        > results/m4/README.md   # kernel-set A/B (real + sim per cell)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(directory: Path) -> dict[str, dict[float, dict]]:
    out: dict[str, dict[float, dict]] = defaultdict(dict)
    for path in sorted(directory.glob("*.summary.json")):
        data = json.loads(path.read_text())
        for row in data.get("sweep", []):
            out[data["config"]][row["budget_gib"]] = row
    return out


def compare(dir_a: Path, dir_b: Path) -> None:
    a_rows, b_rows = _load_rows(dir_a), _load_rows(dir_b)
    name_a, name_b = dir_a.name, dir_b.name
    pretty = {
        "8b": "bs=1, ga=1 (4,096 tokens/step)",
        "8b-bs4ga4": "bs=4, ga=4 (65,536 tokens/step)",
        "8b-ga16": "bs=1, ga=16 (65,536 tokens/step)",
    }
    print(f"# Kernel-set A/B: `{name_a}` vs `{name_b}`\n")
    print("Same programs, plans re-derived per set from re-measured task costs —")
    print("kernel changes move BOTH the sim prediction and the real run (and can")
    print("shift the planner's recompute choices). tok/s = real steady-state;")
    print("(sim NNNN) = the simulator's prediction for that set's plan.\n")
    for config in ("8b", "8b-bs4ga4", "8b-ga16"):
        if config not in a_rows and config not in b_rows:
            continue
        budgets = sorted(set(a_rows.get(config, {})) | set(b_rows.get(config, {})))
        print(f"\n## {config} — {pretty.get(config, '')}\n")
        print(f"| budget (GiB) | {name_a} tok/s | {name_b} tok/s | real speedup | "
              f"{name_a} recompute | {name_b} recompute |")
        print("|---:|---:|---:|---:|---:|---:|")
        for budget in budgets:
            ra = a_rows.get(config, {}).get(budget)
            rb = b_rows.get(config, {}).get(budget)

            def cell(r):
                if r is None:
                    return "—"
                return (f"{r['real_tokens_per_s']:.0f} "
                        f"(sim {r['sim_tokens_per_s']:.0f})")

            speedup = (
                f"{(rb['real_tokens_per_s'] / ra['real_tokens_per_s'] - 1) * 100:+.1f}%"
                if ra and rb else "—"
            )
            rc = lambda r: f"{r['recompute_chosen']}" if r else "—"
            print(f"| {budget:g} | {cell(ra)} | {cell(rb)} | {speedup} | "
                  f"{rc(ra)} | {rc(rb)} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("results/m4"))
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("DIR_A", "DIR_B"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    by_config: dict[str, list[dict]] = defaultdict(list)
    baselines: dict[str, dict] = {}
    kernel_sets: list[dict] = []
    for path in sorted(args.dir.glob("*.summary.json")):
        data = json.loads(path.read_text())
        if "kernel_set" in data:
            kernel_sets.append(data["kernel_set"])
        for row in data.get("sweep", []):
            row["_config"] = data["config"]
            row["_steps"] = data.get("steps")
            by_config[data["config"]].append(row)
        if "baseline_plain_torch_ms_per_step" in data:
            baselines[data["config"]] = data

    print("# M4 results: llama3-8B full-bf16 training, budget vs throughput\n")
    if kernel_sets and all(k == kernel_sets[0] for k in kernel_sets):
        impls = sorted(set(kernel_sets[0].values()))
        label = "fused-v1" if impls == ["triton"] else "+".join(impls)
        print(f"**Kernel set: `{label}`** — registry ops all resolved to "
              f"{impls} (aten flash-attention + cuBLAS GEMMs stay direct).")
        print("Costs are measured per task signature and feed the plans: kernel changes")
        print("move BOTH real throughput and the sim prediction.\n")
    else:
        print("**Kernel set: `eager-v1`** — row-chunked eager-torch ops (+ aten flash-")
        print("attention, cuBLAS GEMMs), predating the kernel registry. The fused set")
        print("moves BOTH columns: real throughput directly, sim via measured costs.\n")
    print("RTX 5090 (31.3 GiB) · bf16 params+grads+AdamW state · seq 4096 ·")
    print("measured task costs · plans built on measured bidirectional PCIe ·")
    print("static buffer placement (offsets packed offline from plan lifetimes,")
    print("validated against physical VRAM at planning time; 'geom. tax' = packed")
    print("extent / peak concurrent load, the price of contiguous placement) ·")
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
        print("| budget (GiB) | sim ms/step | real ms/step | sim tok/s | real tok/s | real vs sim | replay gap | recompute | placed extent | geom. tax | losses |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
        for r in rows:
            losses = ", ".join(f"{x:.3f}" for x in r["losses"])
            extent = r.get("placement_extent_gib")
            extent_s = f"{extent:.2f} GiB" if extent else "—"
            tax = r.get("placement_overhead")
            tax_s = f"×{tax:.2f}" if tax else "—"
            print(
                f"| {r['budget_gib']:g} | {r['sim_ms_per_step']:.0f} | "
                f"{r['real_ms_per_step_steady']:.0f} | {r['sim_tokens_per_s']:.0f} | "
                f"{r['real_tokens_per_s']:.0f} | {r['real_vs_sim_pct']:+.1f}% | "
                f"{r['replay_fidelity_gap_pct']:+.2f}% | "
                f"{r.get('recompute_chosen', 0)}/{_rewrite_count(r)} | "
                f"{extent_s} | {tax_s} | {losses} |"
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

    # correctness at very tight budgets (tools/m4_correctness.py output)
    corr_path = args.dir / "correctness.json"
    if corr_path.exists():
        corr = json.loads(corr_path.read_text())
        c = corr["config"]
        print("\n## Correctness under pressure: runtime vs plain-torch golden\n")
        print(f"Full train step through the real engine vs the golden eager-torch model "
              f"({c['n_layers']}L, d={c['d_model']}, seq {c['seq_len']}): relative-L2 of "
              f"loss + every final weight after the optimizer step. Memory pressure must "
              f"not perturb the math — errors stay at bf16 noise at every budget.\n")
        print("| budget (GiB) | recompute | worst tensor | worst rel-L2 | status |")
        print("|---:|:---:|:---|---:|:---|")
        for p in corr["points"]:
            if p["status"] not in ("ok", "FAILED"):
                print(f"| {p['budget_gib']:g} | {'all' if p['recompute'] else '—'} | — | — | {p['status'].split(':')[0]} |")
            else:
                print(
                    f"| {p['budget_gib']:g} | {'all' if p['recompute'] else '—'} | "
                    f"{p['worst_tensor']} | {p['worst_rel_l2']:.2e} | {p['status']} |"
                )


def _rewrite_count(row: dict) -> int:
    """Recompute-decision denominator = n_layers x grad-accum rounds."""
    layers, rounds = {
        "8b": (32, 1), "8b-bs4ga4": (32, 4), "8b-ga16": (32, 16),
        "baseline1b": (16, 1), "mini": (4, 1),
    }.get(row.get("_config", ""), (1, 1))
    return layers * rounds


if __name__ == "__main__":
    main()
