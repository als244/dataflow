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

    # ---- parity groups: every {preset}_reference with matching engine budgets ----
    refs = {k[: -len("_reference")]: v for k, v in R.items()
            if k.endswith("_reference")}
    for pkey in sorted(refs):
        ref = refs[pkey]
        engines = sorted((v for k, v in R.items() if k.startswith(pkey + "_engine_")),
                         key=lambda e: e.budget_gib or 0)
        if not engines:
            continue
        series = [Series(label="reference (pytorch)",
                         x=list(range(len(ref.losses))), y=ref.losses,
                         color=PALETTE[0], dashed=True)]
        prows = []
        for i, e in enumerate(engines):
            lbl = f"engine @ {e.budget_gib:g} GiB"
            series.append(Series(label=lbl, x=list(range(len(e.losses))),
                                 y=e.losses, color=PALETTE[(i + 1) % len(PALETTE)]))
            rep = parity.compare(ref.losses, e.losses, a_label="reference", b_label=lbl)
            prows.append([lbl, f"{rep.step0_abs:.4f}", f"{rep.max_abs:.4f}",
                          f"{rep.final_abs:.4f}", f"{rep.ema_abs:.4f}",
                          "ALIGNED" if rep.passed else "DIVERGED"])
        if len(engines) == 2:
            a, b = engines
            rep = parity.compare(a.losses, b.losses, a_label=f"{a.budget_gib:g}",
                                 b_label=f"{b.budget_gib:g}")
            prows.append([f"engine {a.budget_gib:g} vs {b.budget_gib:g} (budget-invariance)",
                          f"{rep.step0_abs:.4f}", f"{rep.max_abs:.4f}",
                          f"{rep.final_abs:.4f}", f"{rep.ema_abs:.4f}",
                          "ALIGNED" if rep.passed else "DIVERGED"])
        svg = svg_line_chart(series, title=f"{pkey} — cross-entropy vs step",
                             xlabel="optimizer step", ylabel="mean CE loss")
        tbl = _table(prows, ["comparison", "step0 Δ", "max Δ", "final Δ", "ema Δ", "verdict"])
        sections.append(f"<section><h2>{pkey} reference-vs-engine parity</h2>"
                        f"<div class='chart'>{svg}</div>{tbl}</section>")

    # ---- distributed DP (two-box fleet) vs the single-box runs ----
    if "l3_1b_dp" in R:
        dp = R["l3_1b_dp"]
        ref = R.get("l3_1b_reference")
        eng = R.get("l3_1b_engine_14gib")
        series = []
        prows = []
        if ref is not None:
            series.append(Series(label="reference (pytorch, single box)",
                                 x=list(range(len(ref.losses))),
                                 y=ref.losses, color=PALETTE[0],
                                 dashed=True))
        if eng is not None:
            series.append(Series(label="engine @ 14 GiB (single box)",
                                 x=list(range(len(eng.losses))),
                                 y=eng.losses, color=PALETTE[1]))
        rr = dp.meta.get("rank_rounds", ["?", "?"])
        series.append(Series(
            label=f"engine DP x2 (5090:{rr[0]} + 3090:{rr[1]} rounds)",
            x=list(range(len(dp.losses))), y=dp.losses,
            color=PALETTE[2]))
        for base, lbl in ((eng, "single-box engine"),
                          (ref, "reference")):
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
                  "(weighted 6:2 rounds, 25 GbE) vs the single-box runs",
            xlabel="optimizer step", ylabel="mean CE loss")
        tbl = _table(prows, ["comparison", "step0 \u0394", "max \u0394",
                             "final \u0394", "ema \u0394", "verdict"])
        tokps = f"{dp.steady_tok_per_s:.0f}"
        note = ("<p>Two daemons (RTX 5090 + RTX 3090, direct 25 GbE), "
                "hostmem collectives, global-denominator loss; the "
                "global batch and data order are IDENTICAL to the "
                f"single-box runs. Steady {tokps} tok/s at "
                "10.7 s/step (socket-transport collectives, v1).</p>")
        sections.append("<section><h2>l3_1b distributed data-parallel "
                        f"parity</h2>{note}<div class='chart'>{svg}"
                        f"</div>{tbl}</section>")

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


HTML_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pretraining parity &amp; scaling</title>
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
</style></head><body>
<h1>Pretraining: reference-vs-engine parity &amp; scaling</h1>
<p class="sub">llama3-shaped, fineweb10B (gpt2, vocab 50304), 64K tokens/step,
seq 2048, all-bf16, AdamW + cosine LR (peak 3e-4), 1000 steps. Reference =
independent pytorch nn.Module; engine = dataflowd service at a device budget.
Byte-identical init + one deterministic data stream on both sides.</p>
{body}
</body></html>
"""


def main() -> int:
    body = build()
    out = RESULTS / "report.html"
    out.write_text(HTML_SHELL.format(body=body))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
