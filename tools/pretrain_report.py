#!/usr/bin/env python
"""Build the pretraining study reports (self-contained HTML) from the
saved run curves under results/pretrain/.

One report per study under results/pretrain/reports/ (llama3, qwen35,
dsv3_2b) plus index.html with the combined throughput table. Charts
are inline SVG; light+dark via prefers-color-scheme."""
from __future__ import annotations

import html
import math
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dataflow.pretrain import parity, scaling
from dataflow.pretrain.plot import PALETTE, Series, svg_line_chart

RESULTS = _ROOT / "results" / "pretrain"
REPORTS = RESULTS / "reports"


def table_html(rows: list[list[str]], header: list[str]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>"
                                   for c in r) + "</tr>"
                  for r in rows)
    return (f'<table><thead><tr>{th}</tr></thead>'
            f'<tbody>{trs}</tbody></table>')


def parity_section(R: dict, pkey: str, note: str = "") -> str | None:
    """reference-vs-engine parity block for one {pkey}_ run group."""
    ref = R.get(f"{pkey}_reference")
    engines = sorted((v for k, v in R.items()
                      if k.startswith(pkey + "_engine_")),
                     key=lambda e: e.budget_gib or 0)
    if ref is None or not engines:
        return None
    series = [Series(label="reference (pytorch)",
                     x=list(range(len(ref.losses))), y=ref.losses,
                     color=PALETTE[0], dashed=True)]
    prows = []
    for i, e in enumerate(engines):
        lbl = f"engine @ {e.budget_gib:g} GiB"
        series.append(Series(label=lbl, x=list(range(len(e.losses))),
                             y=e.losses,
                             color=PALETTE[(i + 1) % len(PALETTE)]))
        rep = parity.compare(ref.losses, e.losses, a_label="reference",
                             b_label=lbl)
        prows.append([lbl, f"{rep.step0_abs:.4f}", f"{rep.max_abs:.4f}",
                      f"{rep.final_abs:.4f}", f"{rep.ema_abs:.4f}",
                      "ALIGNED" if rep.passed else "DIVERGED"])
    if len(engines) == 2:
        a, b = engines
        rep = parity.compare(a.losses, b.losses,
                             a_label=f"{a.budget_gib:g}",
                             b_label=f"{b.budget_gib:g}")
        prows.append([f"engine {a.budget_gib:g} vs {b.budget_gib:g} "
                      f"(budget-invariance)",
                      f"{rep.step0_abs:.4f}", f"{rep.max_abs:.4f}",
                      f"{rep.final_abs:.4f}", f"{rep.ema_abs:.4f}",
                      "ALIGNED" if rep.passed else "DIVERGED"])
    svg = svg_line_chart(series, title=f"{pkey} — cross-entropy vs step",
                         xlabel="optimizer step", ylabel="mean CE loss")
    tbl = table_html(prows, ["comparison", "step0 Δ", "max Δ",
                             "final Δ", "ema Δ", "verdict"])
    return (f"<section><h2>{pkey} reference-vs-engine parity</h2>"
            f"{note}<div class='chart'>{svg}</div>{tbl}</section>")


def dp_section(R: dict) -> str | None:
    if "l3_1b_dp" not in R:
        return None
    dp = R["l3_1b_dp"]
    ref = R.get("l3_1b_reference")
    eng = R.get("l3_1b_engine_14gib")
    series = []
    prows = []
    if ref is not None:
        series.append(Series(label="reference (pytorch, single box)",
                             x=list(range(len(ref.losses))),
                             y=ref.losses, color=PALETTE[0], dashed=True))
    if eng is not None:
        series.append(Series(label="engine @ 14 GiB (single box)",
                             x=list(range(len(eng.losses))),
                             y=eng.losses, color=PALETTE[1]))
    rr = dp.meta.get("rank_rounds", ["?", "?"])
    series.append(Series(
        label=f"engine DP x2 (5090:{rr[0]} + 3090:{rr[1]} rounds)",
        x=list(range(len(dp.losses))), y=dp.losses, color=PALETTE[2]))
    for base, lbl in ((eng, "single-box engine"), (ref, "reference")):
        if base is None:
            continue
        rep = parity.compare(base.losses, dp.losses, a_label=lbl,
                             b_label="fleet DP")
        prows.append([f"fleet DP vs {lbl}", f"{rep.step0_abs:.4f}",
                      f"{rep.max_abs:.4f}", f"{rep.final_abs:.4f}",
                      f"{rep.ema_abs:.4f}",
                      "ALIGNED" if rep.passed else "DIVERGED"])
    svg = svg_line_chart(
        series,
        title="l3_1b — DATA-PARALLEL across two machines "
              "(weighted rounds, 25 GbE) vs the single-box runs",
        xlabel="optimizer step", ylabel="mean CE loss")
    tbl = table_html(prows, ["comparison", "step0 Δ", "max Δ",
                             "final Δ", "ema Δ", "verdict"])
    tokps = f"{dp.steady_tok_per_s:.0f}"
    note = ("<p>Two daemons (RTX 5090 + RTX 3090, direct 25 GbE), "
            "group collectives, global-denominator loss; the global "
            "batch and data order are IDENTICAL to the single-box "
            f"runs. Steady {tokps} tok/s at record time.</p>")
    return ("<section><h2>l3_1b distributed data-parallel parity</h2>"
            f"{note}<div class='chart'>{svg}</div>{tbl}</section>")


def scaling_section(R: dict) -> str | None:
    scale_results = {k: R[k] for k in R if k.startswith("scaling_")}
    if "l3_1b_engine_14gib" in R:
        top = R["l3_1b_engine_14gib"]
        if top.meta.get("params") is None:
            from dataflow.pretrain import presets as P

            top.meta["params"] = P.param_counts(P.preset("l3_1b"))
            top.meta["preset"] = "l3_1b"
        scale_results["scaling_l3_1b_engine"] = top
    if len(scale_results) < 2:
        return None
    series = []
    for k, r in sorted(scale_results.items(),
                       key=lambda kv: kv[1].meta.get("params", {})
                       .get("non_embedding", 0)):
        name = r.meta.get("preset", k)
        series.append(Series(label=name, x=list(range(len(r.losses))),
                             y=r.losses))
    svg_curves = svg_line_chart(series, title="scaling ladder — CE vs step",
                                xlabel="optimizer step",
                                ylabel="mean CE loss")
    fit = scaling.fit_scaling(scale_results)
    pts = [Series(label="final loss", x=[n for n, _, _ in fit.points],
                  y=[l for _, l, _ in fit.points], markers=True)]
    fit_series = []
    if not math.isnan(fit.a) and fit.points:
        ns = [fit.points[0][0], fit.points[-1][0]]
        fit_series = [Series(label=f"fit L={fit.a:.2f}-{fit.b:.2f}·log10(N)",
                             x=ns, y=[fit.predict(n) for n in ns],
                             dashed=True, color="#dc2626")]
    svg_fit = svg_line_chart(pts + fit_series,
                             title="final loss vs non-embedding params",
                             xlabel="non-embedding params",
                             ylabel="final CE", xlog=True)
    return (f"<section><h2>Scaling ladder</h2>"
            f"<div class='chart'>{svg_curves}</div>"
            f"<div class='chart'>{svg_fit}</div>"
            f"<p>log-linear fit R²={fit.r2:.3f}</p></section>")


def dsv3_balance_section(R: dict) -> str | None:
    """LBL-off vs paper-like balancing at the resident budget."""
    off = R.get("dsv3_2b_nolbl_engine_14gib")
    on = R.get("dsv3_2b_engine_14gib")
    if off is None or on is None:
        return None
    series = [Series(label="balancing OFF (engine @ 14 GiB)",
                     x=list(range(len(off.losses))), y=off.losses,
                     color=PALETTE[1]),
              Series(label="paper-like (bias 1e-3 + aux 1e-4)",
                     x=list(range(len(on.losses))), y=on.losses,
                     color=PALETTE[2])]
    svg = svg_line_chart(series,
                         title="dsv3_2b — load balancing on vs off",
                         xlabel="optimizer step", ylabel="mean CE loss")
    m_off = statistics.mean(off.losses[-100:])
    m_on = statistics.mean(on.losses[-100:])
    rows = [["balancing OFF", f"{off.losses[-1]:.4f}", f"{m_off:.4f}"],
            ["paper-like", f"{on.losses[-1]:.4f}", f"{m_on:.4f}"],
            ["delta (off - on)", f"{off.losses[-1] - on.losses[-1]:+.4f}",
             f"{m_off - m_on:+.4f}"]]
    note = ("<p>Same data, seeds, and budgets; only the router "
            "balancing differs (noaux bias update 1e-3 + seq-wise aux "
            "1e-4 vs both off). Routing did not collapse in 1000 "
            "steps even unbalanced; the paper-like configuration ends "
            "slightly ahead.</p>")
    return ("<section><h2>dsv3_2b load-balancing comparison</h2>"
            f"{note}<div class='chart'>{svg}</div>"
            + table_html(rows, ["configuration", "final loss",
                                "last-100 mean"]) + "</section>")


def throughput_section(R: dict, prefixes: tuple = ()) -> str | None:
    tput = scaling.throughput_table(R)
    if prefixes:
        tput = [t for t in tput if t.label.startswith(prefixes)]
    if not tput:
        return None
    rows = [[t.label, t.backend,
             ("-" if t.budget_gib is None else f"{t.budget_gib:g}"),
             f"{t.steady_tok_per_s:,.0f}", f"{t.final_loss:.3f}"]
            for t in sorted(tput, key=lambda t: t.label)]
    return ("<section><h2>Throughput</h2>"
            + table_html(rows, ["run", "backend", "budget GiB",
                                "tok/s", "final loss"])
            + "</section>")


HTML_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ max-width: 900px; margin: 2rem auto; padding: 0 1rem;
  font-family: ui-sans-serif, system-ui, sans-serif; line-height: 1.5;
  color: #1a1a1a; background: #fff; }}
@media (prefers-color-scheme: dark) {{ body {{ color: #e6e6e6; background: #14161a; }} }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
.chart {{ overflow-x: auto; margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.5rem 0; }}
th, td {{ border: 1px solid; border-color: color-mix(in srgb, currentColor 20%, transparent);
  padding: 4px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.sub {{ opacity: 0.75; font-size: 0.9rem; }}
nav a {{ margin-right: 1rem; }}
</style></head><body>
<nav><a href="index.html">index</a><a href="llama3.html">llama3</a>
<a href="qwen35.html">qwen35</a><a href="dsv3_2b.html">dsv3_2b</a></nav>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{body}
</body></html>
"""

COMMON_SUB = ("fineweb10B (gpt2, vocab 50304), 64K tokens/step, seq 2048, "
              "all-bf16 (AdamW m/v bf16, fp32 math), cosine LR peak 3e-4, "
              "1000 steps. Reference = independent pytorch nn.Module; "
              "engine = dataflowd service at a device budget. "
              "Byte-identical init + one deterministic data stream.")


def write_report(name: str, title: str, subtitle: str,
                 sections: list) -> None:
    body = "\n".join(s for s in sections if s) or "<p>no results yet</p>"
    out = REPORTS / f"{name}.html"
    out.write_text(HTML_SHELL.format(title=html.escape(title),
                                     subtitle=subtitle, body=body))
    print(f"wrote {out}")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    R = scaling.load_all(RESULTS)

    write_report(
        "llama3", "llama3 pretraining: parity, distributed DP, scaling",
        "llama3-shaped dense (l3_1b = 1.2B). " + COMMON_SUB,
        [parity_section(R, "l3_1b"), dp_section(R), scaling_section(R),
         throughput_section(R, ("l3_", "scaling_", "smoke"))])

    write_report(
        "qwen35", "qwen3.5 pretraining: parity",
        "qwen3.5 hybrid (delta-rule linear attention + full-attention "
        "layers). " + COMMON_SUB,
        [parity_section(R, "qwen35"), throughput_section(R, ("qwen35",))])

    dsv3_note = ("<p>dsv3_2b: 1.89B-total / ~0.5B-active MoE — 14 layers "
                 "(first 2 dense), d_model 1280, MLA (q_lora 640 / "
                 "kv_lora 320), 40 experts × d_ff 832, grouped top-4 "
                 "(8 groups pick 4), shared expert 2560. Budgets from "
                 "the plan oracle: natural peak 12.31 GiB → 4 GiB "
                 "(recompute + heavy PCIe) and 14 GiB (resident). "
                 "Reference runs gradient-checkpointed (1.89B "
                 "otherwise exceeds the 32 GB card).</p>")
    write_report(
        "dsv3_2b", "dsv3 (MoE) pretraining: parity + load balancing",
        "DeepSeek-V3-shaped MoE at 1.89B total params, two balancing "
        "configurations. " + COMMON_SUB,
        [parity_section(R, "dsv3_2b_nolbl",
                        note=dsv3_note + "<p><b>Balancing OFF</b> "
                        "(aux 0, bias 0):</p>"),
         parity_section(R, "dsv3_2b",
                        note="<p><b>Paper-like balancing</b> (noaux "
                        "router bias 1e-3 + seq-wise aux 1e-4):</p>"),
         dsv3_balance_section(R),
         throughput_section(R, ("dsv3_2b",))])

    write_report(
        "index", "Pretraining studies",
        "One report per study; combined throughput below.",
        ["<section><h2>Studies</h2><ul>"
         "<li><a href='llama3.html'>llama3 — parity, distributed DP, "
         "scaling ladder</a></li>"
         "<li><a href='qwen35.html'>qwen3.5 — parity</a></li>"
         "<li><a href='dsv3_2b.html'>dsv3 (MoE, 1.89B) — parity + "
         "load balancing</a></li></ul></section>",
         throughput_section(R)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
