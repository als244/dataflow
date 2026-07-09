#!/usr/bin/env python
"""Build the pretraining parity + scaling report (self-contained HTML) from
the saved run curves under results/pretrain/."""
from __future__ import annotations

import html
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dataflow.pretrain import parity, scaling
from dataflow.pretrain.plot import PALETTE, Series, svg_line_chart

RESULTS = _ROOT / "results" / "pretrain"


def _loss_series(results: dict, keys_labels_colors) -> list[Series]:
    out = []
    for i, (key, label, color) in enumerate(keys_labels_colors):
        r = results.get(key)
        if r is None:
            continue
        out.append(Series(label=label, x=list(range(len(r.losses))), y=r.losses,
                          color=color or PALETTE[i % len(PALETTE)],
                          dashed=("reference" in key)))
    return out


def _table(rows: list[list[str]], header: list[str]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
                  for r in rows)
    return f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def build(results_dir=RESULTS) -> str:
    R = scaling.load_all(results_dir)
    sections = []

    # ---- 1B parity: overlay reference vs engine budgets ----
    par_keys = [
        ("l3_1b_reference", "reference (pytorch)", PALETTE[0]),
        ("l3_1b_engine_6gib", "engine @ 6 GiB", PALETTE[1]),
        ("l3_1b_engine_14gib", "engine @ 14 GiB", PALETTE[2]),
    ]
    par_series = _loss_series(R, par_keys)
    if par_series:
        svg = svg_line_chart(par_series, title="1B — cross-entropy vs step",
                             xlabel="optimizer step", ylabel="mean CE loss")
        # parity metrics
        prows = []
        ref = R.get("l3_1b_reference")
        if ref:
            for key, label, _ in par_keys[1:]:
                e = R.get(key)
                if e:
                    rep = parity.compare(ref.losses, e.losses, a_label="reference",
                                         b_label=label)
                    prows.append([label, f"{rep.step0_abs:.4f}", f"{rep.max_abs:.4f}",
                                  f"{rep.final_abs:.4f}", f"{rep.ema_abs:.4f}",
                                  "ALIGNED" if rep.passed else "DIVERGED"])
            e6, e14 = R.get("l3_1b_engine_6gib"), R.get("l3_1b_engine_14gib")
            if e6 and e14:
                rep = parity.compare(e6.losses, e14.losses, a_label="6", b_label="14")
                prows.append(["engine 6 vs 14 (budget-invariance)", f"{rep.step0_abs:.4f}",
                              f"{rep.max_abs:.4f}", f"{rep.final_abs:.4f}",
                              f"{rep.ema_abs:.4f}", "ALIGNED" if rep.passed else "DIVERGED"])
        tbl = _table(prows, ["comparison", "step0 Δ", "max Δ", "final Δ", "ema Δ", "verdict"])
        sections.append(f"<section><h2>1B reference-vs-engine parity</h2>"
                        f"<div class='chart'>{svg}</div>{tbl}</section>")

    # ---- scaling ladder ----
    scale_keys = [k for k in R if k.startswith("scaling_")]
    scale_results = {k: R[k] for k in scale_keys}
    # include the 1B engine@14 parity run as the top of the ladder if present
    if "l3_1b_engine_14gib" in R and R["l3_1b_engine_14gib"].meta.get("params") is None:
        from dataflow.pretrain import presets as P
        R["l3_1b_engine_14gib"].meta["params"] = P.param_counts(P.preset("l3_1b"))
        R["l3_1b_engine_14gib"].meta["preset"] = "l3_1b"
    if "l3_1b_engine_14gib" in R:
        scale_results["scaling_l3_1b_engine"] = R["l3_1b_engine_14gib"]
    if len(scale_results) >= 2:
        series = []
        for i, (k, r) in enumerate(sorted(scale_results.items(),
                                          key=lambda kv: kv[1].meta.get("params", {}).get("non_embedding", 0))):
            name = r.meta.get("preset", k)
            series.append(Series(label=name, x=list(range(len(r.losses))), y=r.losses))
        svg_curves = svg_line_chart(series, title="scaling ladder — CE vs step",
                                    xlabel="optimizer step", ylabel="mean CE loss")
        fit = scaling.fit_scaling(scale_results)
        pts_series = [Series(label="final loss", x=[n for n, _, _ in fit.points],
                             y=[l for _, l, _ in fit.points], markers=True)]
        fit_series = []
        if not math.isnan(fit.a) and fit.points:
            ns = [fit.points[0][0], fit.points[-1][0]]
            fit_series = [Series(label=f"fit L={fit.a:.2f}-{fit.b:.2f}·log10(N)",
                                 x=ns, y=[fit.predict(n) for n in ns], dashed=True,
                                 color="#dc2626")]
        svg_fit = svg_line_chart(pts_series + fit_series,
                                 title="final loss vs non-embedding params",
                                 xlabel="non-embedding params", ylabel="final CE",
                                 xlog=True)
        sections.append(f"<section><h2>Scaling ladder</h2>"
                        f"<div class='chart'>{svg_curves}</div>"
                        f"<div class='chart'>{svg_fit}</div>"
                        f"<p>log-linear fit R²={fit.r2:.3f}</p></section>")

    # ---- throughput ----
    tput = scaling.throughput_table(R)
    if tput:
        rows = [[t.label, t.backend, ("-" if t.budget_gib is None else f"{t.budget_gib:g}"),
                 f"{t.steady_tok_per_s:,.0f}", f"{t.final_loss:.3f}"]
                for t in sorted(tput, key=lambda t: t.label)]
        sections.append("<section><h2>Throughput</h2>"
                        + _table(rows, ["run", "backend", "budget GiB", "tok/s", "final loss"])
                        + "</section>")

    body = "\n".join(sections) or "<p>no results yet</p>"
    return body


HTML_SHELL = """<style>
body {{ max-width: 900px; margin: 2rem auto; padding: 0 1rem;
  font-family: ui-sans-serif, system-ui, sans-serif; line-height: 1.5; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
.chart {{ overflow-x: auto; margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.5rem 0; }}
th, td {{ border: 1px solid; border-color: color-mix(in srgb, currentColor 20%, transparent);
  padding: 4px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
<h1>Pretraining: reference-vs-engine parity &amp; scaling</h1>
{body}
"""


def main() -> int:
    body = build()
    out = RESULTS / "report.html"
    out.write_text(HTML_SHELL.format(body=body))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
