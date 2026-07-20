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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from dataflow_training.run import parity, scaling
from dataflow_training.run.plot import PALETTE, Series, svg_line_chart

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
            from dataflow_training.run import presets as P

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


LADDER_STAGES = [
    ("stage1_socket_perfield", "v1: socket lane, per-field exchange",
     "TCP frames, 9 posts/layer, CPU fp32 rank-ordered reduce"),
    ("stage2_zerocopy_native", "zero-copy staging + native-dtype reduce",
     "slab-extent staging (registered MR), one-pass bf16 CPU add"),
    ("stage3_rdma_fused", "rdma lane + fused per-layer exchange",
     "one-sided RDMA_WRITE, ONE 121.6 MB exchange per layer"),
    ("stage4_device_reduce", "device-side reduce",
     "peer bytes H2D'd and summed on-GPU; zero CPU passes"),
    ("stage5_nccl_backend", "NCCL group backend (tuned multi-socket)",
     "device-direct collectives, zero PCIe staging; ties the "
     "hostmem lane at this scale"),
    ("stage6_nccl_zero1", "+ ZeRO-1 optimizer sharding",
     "region reduce -> owned update -> W broadcast; per-rank "
     "optimizer state halves (2.01/2.38 GiB) for ~6% step cost"),
]

# coll_bench pattern measurements (cross-box, probed link 23.09 Gbit/s;
# recorded in the findings ledger the day they were run)
MICRO_ROWS = [
    ["one layer, 9 fields (socket, per-field)", "299 ms", "—",
     "profiled training gap: 200-300 ms"],
    ["one layer, 9 fields (rdma, CPU reduce)", "99 ms", "64 ms",
     "per-op tax ~4 ms x 8 extra ops"],
    ["one layer fused 121.6 MB (rdma, CPU reduce)", "67 ms", "64 ms",
     "wire 42 + CPU reduce 14 + PCIe 6"],
    ["one layer fused (rdma, DEVICE reduce)", "54 ms", "~48 ms",
     "wire 42 + PCIe 6; reduce off the wall"],
    ["one layer, 9 fields (rdma, device reduce)", "58 ms", "~48 ms",
     "per-op tax collapses to ~0.5 ms"],
    ["head+embed 412 MB (socket, per-field)", "761 ms", "—",
     "profiled training gap: ~800 ms"],
    ["head+embed 412 MB (rdma, device reduce)", "183 ms", "~164 ms",
     "wire 143 + PCIe 21"],
]


def distperf_sections(R: dict) -> list:
    from dataflow_training.run.driver import load_result

    sections = []
    prov = ("<section><h2>Measured floors, not guesses</h2><p>Every "
            "bound in this report composes from PROBED lanes: the "
            "connect-time link probe (rdma 23.09 Gbit/s — equal to the "
            "ib_write_bw ceiling — and socket ~14.3), the boot "
            "host-bandwidth probe (bf16 add 17.7 GB/s payload = 53 GB/s "
            "traffic = 79% of the dual-channel DDR5-4200 theoretical "
            "67.2), and PCIe H2D/D2H (35/44 GB/s). A layer's gradient "
            "is 121.6 MB bf16 (9 fields, contiguous); embed and head "
            "are untied 206 MB each.</p></section>")
    sections.append(prov)

    # ---- optimization ladder over the SAME 4-step 64K-token run ----
    rows = []
    series = []
    single64 = None
    sweep_dir = RESULTS / "sweeps" / "l3_1b_batch"
    if (sweep_dir / "single_64K.json").exists():
        single64 = load_result(sweep_dir / "single_64K.json")
    for i, (key, label, detail) in enumerate(LADDER_STAGES):
        path = RESULTS / "perf_ladder" / f"{key}.json"
        if not path.exists():
            continue
        r = load_result(path)
        walls = r.step_wall_s[1:] or r.step_wall_s
        wall = sum(walls) / len(walls)
        speed = ""
        if single64 is not None:
            speed = f"{r.steady_tok_per_s / single64.steady_tok_per_s:.2f}x"
        rows.append([label, detail, f"{wall:.2f} s",
                     f"{r.steady_tok_per_s:,.0f}", speed])
        series.append(Series(label=label,
                             x=list(range(len(r.step_wall_s))),
                             y=r.step_wall_s,
                             color=PALETTE[i % len(PALETTE)]))
    if rows:
        if single64 is not None:
            rows.append(["single GPU (RTX 5090), same 64K step", "",
                         f"{sum(single64.step_wall_s[1:]) / max(1, len(single64.step_wall_s) - 1):.2f} s",
                         f"{single64.steady_tok_per_s:,.0f}", "1.00x"])
        svg = svg_line_chart(series,
                             title="per-step wall by comm-plane stage "
                                   "(l3_1b, 64K tokens/step, 6:2 rounds)",
                             xlabel="step", ylabel="seconds")
        sections.append(
            "<section><h2>The optimization ladder</h2>"
            "<p>Same model, data, and 3:1 round split; only the "
            "collective plane changes. 10.9 s/step to 3.6 s/step in "
            "three landings (3.1x), each verified loss-banded against "
            "the recorded curves. The NCCL backend then ties the "
            "hostmem lane (its wire deficit on this fabric is bought "
            "back by zero PCIe staging), and ZeRO-1 sharding trades "
            "~6% step time for halved per-rank optimizer state — "
            "proven bitwise-equal to plain DP on the hostmem lane, "
            "and certified at full horizon by two 1000-step fineweb "
            "runs: NCCL + ZeRO-1 tracks the recorded DP flagship at "
            "max EMA |&Delta;| 0.0016 (final 4.9283 vs 4.9284) and "
            "the pure default path (auto&rarr;nccl, replicated "
            "optimizer) at 0.0022 (final 4.9289) — both tighter than "
            "the flagship sits to the single engine (0.0037).</p>"
            "<p>Tensor parallelism (MLP-sharded llama3 through the "
            "sharding API) is certified on an ARCH-HOMOGENEOUS pair: "
            "a 1000-step fineweb run with both ranks on one RTX 5090 "
            "tracks the recorded engine curve at max EMA |&Delta;| "
            "0.0023 (final 4.9331 vs 4.9326) with bitwise-identical "
            "replicas at every step. The heterogeneous finding: "
            "activation-level TP requires matching GPU architectures "
            "— sm86-vs-sm120 kernel differences measure 3&ndash;5e-4 "
            "nats/round at identical weights, and cross-arch TP sums "
            "a mixture of two model-variants every layer, degrading "
            "training deterministically (lane-independent; measured "
            "identically over nccl and hostmem). Data parallelism "
            "and ZeRO-1 are structurally immune: the gradient "
            "allreduce collapses all ranks onto one trajectory.</p>"
            f"<div class='chart'>{svg}</div>"
            + table_html(rows, ["stage", "what changed", "step wall",
                                "tok/s", "vs single GPU"])
            + "</section>")

    # ---- microbenches vs composed floors ----
    sections.append(
        "<section><h2>Exchange microbenches vs composed floors</h2>"
        "<p>coll_bench replays the exact per-layer / head+embed "
        "patterns through the real collective path (both ranks "
        "posting concurrently). The bench reproduced the profiled "
        "training gaps before the fixes and sits at the composed "
        "floor after them.</p>"
        + table_html(MICRO_ROWS, ["pattern", "measured", "composed floor",
                                  "notes"]) + "</section>")

    # ---- single vs distributed sweep ----
    sweep_rows = []
    s_single, s_dist = [], []
    for tag, tokens in (("32K", 32768), ("64K", 65536), ("128K", 131072),
                        ("256K", 262144), ("512K", 524288)):
        sp = sweep_dir / f"single_{tag}.json"
        dp = sweep_dir / f"dist_{tag}.json"
        if not (sp.exists() and dp.exists()):
            continue
        s = load_result(sp)
        d = load_result(dp)
        sweep_rows.append([tag, f"{s.steady_tok_per_s:,.0f}",
                           f"{d.steady_tok_per_s:,.0f}",
                           f"{d.steady_tok_per_s / s.steady_tok_per_s:.2f}x",
                           f"{sum(s.step_wall_s):.1f} s",
                           f"{sum(d.step_wall_s):.1f} s"])
        s_single.append((tokens, s.steady_tok_per_s))
        s_dist.append((tokens, d.steady_tok_per_s))
    if sweep_rows:
        series = [Series(label="single GPU (5090)",
                         x=[t for t, _ in s_single],
                         y=[v for _, v in s_single], color=PALETTE[0],
                         markers=True),
                  Series(label="distributed 3:1 (5090+3090)",
                         x=[t for t, _ in s_dist],
                         y=[v for _, v in s_dist], color=PALETTE[2],
                         markers=True)]
        svg = svg_line_chart(series,
                             title="throughput vs global batch "
                                   "(10 steps, budgets 16/16 GiB, "
                                   "T_round 8192)",
                             xlabel="global tokens per step",
                             ylabel="tok/s", xlog=True)
        sections.append(
            "<section><h2>Single GPU vs distributed: the batch sweep"
            "</h2><p>The exchange tail is FIXED per step (gradient "
            "bytes do not grow with batch), so distributed wins only "
            "once compute amortizes it: crossover lands between 128K "
            "and 256K global tokens, reaching 1.10x at 512K — 91% of "
            "this pair's measured ceiling (1.21x: the 3090 "
            "contributes ~7.3K tok/s against the 5090's ~22.9K "
            "in-fleet).</p>"
            f"<div class='chart'>{svg}</div>"
            + table_html(sweep_rows,
                         ["global batch", "single tok/s", "dist tok/s",
                          "speedup", "single wall (10 steps)",
                          "dist wall"]) + "</section>")

    # ---- split calibration + T_round ----
    cal_rows = []
    for fname, label in (("dist_512K.json", "48:16 (the 3:1 mandate)"),
                         ("dist_512K_45_19.json", "45:19"),
                         ("dist_512K_50_14.json", "50:14"),
                         ("dist_512K_T32K.json",
                          "48:16 at T_round=32K (12:4 rounds)")):
        path = sweep_dir / fname
        if not path.exists():
            continue
        r = load_result(path)
        walls = r.step_wall_s[1:] or r.step_wall_s
        cal_rows.append([label, f"{sum(walls) / len(walls):.2f} s",
                         f"{r.steady_tok_per_s:,.0f}"])
    if cal_rows:
        sections.append(
            "<section><h2>Round-split calibration & T_round</h2>"
            "<p>Three splits at 512K pin the per-round walls: 5090 "
            "~0.36 s in-fleet (0.33 solo + ~9% rank-0 overhead), 3090 "
            "~1.11 s — a 3.1:1 ratio, so the original 3:1 split was "
            "already optimal (45:19 puts the 3090 on the critical "
            "path; 50:14 just flips the 5090 onto it at the same "
            "height). T_round=32K is throughput-neutral: per-round "
            "overheads were already negligible, and the 4x activation "
            "footprint (11.1 to 15.8 GiB peak) only spends budget "
            "headroom.</p>"
            + table_html(cal_rows, ["configuration", "step wall",
                                    "tok/s"]) + "</section>")

    sections.append(
        "<section><h2>What remains on the table</h2><ul>"
        "<li><b>Exchange/backward overlap</b> (dedicated grad-reduce "
        "tasks + tail optimizers): prototyped and bitwise-correct at "
        "one step, then removed pending an engine completion-contract "
        "extension (output produced-ness); the design and measurements "
        "are recorded; projected ~2.9-3.1 s/step at 64K.</li>"
        "<li><b>Pipelined comm worker</b>: deterministic landing ring "
        "kills the RDY round-trip and the last ~0.5 ms/op tax for "
        "non-contiguous patterns.</li>"
        "<li><b>NCCL group backend</b> (in build): device-direct "
        "collectives, no PCIe staging at all — the production lane; "
        "hostmem remains the reference/fallback.</li></ul></section>")
    return sections


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
<nav><a href="index.html">index</a><a href="gpt2.html">gpt2</a>
<a href="llama3.html">llama3</a>
<a href="qwen35.html">qwen35</a><a href="dsv3_2b.html">dsv3_2b</a>
<a href="l3_1b_distperf.html">distributed perf</a></nav>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{body}
<p class="sub">charts: tap/click a legend entry to hide/show its series; drag a
box, scroll, or pinch on the plot to zoom; double-tap/-click resets.
(If charts are inert, the viewer stripped scripts — open the file
in a browser.)</p>
<script>
document.querySelectorAll("svg.ichart").forEach(function (svg) {{
  var fr = svg.dataset.frame.split(",").map(Number);
  var ml = fr[0], mt = fr[1], pw = fr[2], ph = fr[3];
  var plot = svg.querySelector("g.plot");
  var t = {{x: 0, y: 0, kx: 1, ky: 1}};
  svg.style.touchAction = "none";
  function apply() {{
    plot.setAttribute("transform",
      "translate(" + t.x + "," + t.y + ") scale(" + t.kx + "," + t.ky + ")");
    var zoomed = t.kx !== 1 || t.ky !== 1;
    svg.querySelectorAll("text.tick").forEach(function (e) {{
      e.setAttribute("fill-opacity", zoomed ? "0.15" : "0.7");
    }});
  }}
  function reset() {{ t = {{x: 0, y: 0, kx: 1, ky: 1}}; apply(); }}
  svg.querySelectorAll("g.lg").forEach(function (lg) {{
    lg.addEventListener("click", function (ev) {{
      ev.stopPropagation();
      var s = svg.querySelector('g.s[data-i="' + lg.dataset.i + '"]');
      if (!s) return;
      var hidden = s.style.display === "none";
      s.style.display = hidden ? "" : "none";
      lg.setAttribute("opacity", hidden ? "1" : "0.3");
    }});
  }});
  function toUser(ev) {{
    var r = svg.getBoundingClientRect();
    var vb = svg.viewBox.baseVal;
    return {{x: (ev.clientX - r.left) * vb.width / r.width,
             y: (ev.clientY - r.top) * vb.height / r.height}};
  }}
  function inFrame(p) {{
    return p.x >= ml && p.x <= ml + pw && p.y >= mt && p.y <= mt + ph;
  }}
  var pointers = new Map(), band = null, dragStart = null;
  var pinchStart = null, lastTap = 0;
  svg.addEventListener("pointerdown", function (ev) {{
    var p = toUser(ev);
    if (ev.target.closest && ev.target.closest("g.lg")) return;
    if (!inFrame(p) && pointers.size === 0) return;
    pointers.set(ev.pointerId, p);
    svg.setPointerCapture(ev.pointerId);
    if (pointers.size === 1) {{
      var now = Date.now();               // double-tap reset (touch)
      if (now - lastTap < 350) {{ reset(); lastTap = 0; return; }}
      lastTap = now;
      dragStart = p;
      band = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      band.setAttribute("fill", "currentColor");
      band.setAttribute("fill-opacity", "0.08");
      band.setAttribute("stroke", "currentColor");
      band.setAttribute("stroke-dasharray", "4 3");
      svg.appendChild(band);
    }} else if (pointers.size === 2) {{   // pinch begins: drop the band
      if (band) {{ band.remove(); band = null; }}
      dragStart = null;
      var ps = Array.from(pointers.values());
      pinchStart = {{d: Math.hypot(ps[0].x - ps[1].x, ps[0].y - ps[1].y),
                    cx: (ps[0].x + ps[1].x) / 2,
                    cy: (ps[0].y + ps[1].y) / 2, t0: t}};
    }}
    ev.preventDefault();
  }});
  svg.addEventListener("pointermove", function (ev) {{
    if (!pointers.has(ev.pointerId)) return;
    var p = toUser(ev);
    pointers.set(ev.pointerId, p);
    if (pointers.size === 1 && dragStart && band) {{
      band.setAttribute("x", Math.min(dragStart.x, p.x));
      band.setAttribute("y", Math.min(dragStart.y, p.y));
      band.setAttribute("width", Math.abs(p.x - dragStart.x));
      band.setAttribute("height", Math.abs(p.y - dragStart.y));
    }} else if (pointers.size === 2 && pinchStart) {{
      var ps = Array.from(pointers.values());
      var d = Math.hypot(ps[0].x - ps[1].x, ps[0].y - ps[1].y);
      var z = Math.min(Math.max(d / Math.max(pinchStart.d, 1), 0.05), 50);
      var t0 = pinchStart.t0;
      var nk = Math.min(Math.max(t0.kx * z, 1), 400);
      z = nk / t0.kx;
      t = {{kx: t0.kx * z, ky: t0.ky * z,
           x: pinchStart.cx - z * (pinchStart.cx - t0.x),
           y: pinchStart.cy - z * (pinchStart.cy - t0.y)}};
      if (t.kx === 1) {{ t = {{x: 0, y: 0, kx: 1, ky: 1}}; }}
      apply();
    }}
  }});
  function endPointer(ev) {{
    if (!pointers.has(ev.pointerId)) return;
    var p = toUser(ev);
    pointers.delete(ev.pointerId);
    if (pinchStart && pointers.size < 2) {{ pinchStart = null; }}
    if (dragStart && pointers.size === 0) {{
      var x0 = Math.min(dragStart.x, p.x), x1 = Math.max(dragStart.x, p.x);
      var y0 = Math.min(dragStart.y, p.y), y1 = Math.max(dragStart.y, p.y);
      if (band) {{ band.remove(); band = null; }}
      dragStart = null;
      if (x1 - x0 >= 8 && y1 - y0 >= 8) {{
        var zx = pw / (x1 - x0), zy = ph / (y1 - y0);
        t = {{kx: t.kx * zx, ky: t.ky * zy,
             x: ml - zx * (x0 - t.x), y: mt - zy * (y0 - t.y)}};
        apply();
      }}
    }}
    if (band && pointers.size === 0) {{ band.remove(); band = null; }}
  }}
  svg.addEventListener("pointerup", endPointer);
  svg.addEventListener("pointercancel", endPointer);
  svg.addEventListener("wheel", function (ev) {{
    var p = toUser(ev);
    if (!inFrame(p)) return;
    ev.preventDefault();
    var z = ev.deltaY < 0 ? 1.25 : 0.8;
    var nk = Math.min(Math.max(t.kx * z, 1), 400);
    z = nk / t.kx;
    t = {{kx: t.kx * z, ky: t.ky * z,
         x: p.x - z * (p.x - t.x), y: p.y - z * (p.y - t.y)}};
    if (t.kx === 1) {{ t = {{x: 0, y: 0, kx: 1, ky: 1}}; }}
    apply();
  }}, {{passive: false}});
  svg.addEventListener("dblclick", reset);
}});
</script>
</body></html>
"""

COMMON_SUB = ("fineweb10B (gpt2, vocab 50304), 64K tokens/step, seq 2048, "
              "all-bf16 (AdamW m/v bf16, fp32 math), cosine LR peak 3e-4, "
              "1000 steps. Reference = independent pytorch nn.Module; "
              "engine = dataflowd service at a device budget. "
              "Byte-identical init + one deterministic data stream.")


def ema_curve(xs: list[float], alpha: float = 0.98) -> list[float]:
    out, m = [], xs[0]
    for x in xs:
        m = alpha * m + (1 - alpha) * x
        out.append(m)
    return out


def gpt2_pair_section(R: dict) -> str | None:
    """gpt2 reference-vs-engine agreement: the 1000-step certified pair
    (reference + engine on one box) plus the long-run engine's first
    1000 steps from the second box (cross-box reproduction)."""
    import json as json_mod

    pair_dir = RESULTS / "gpt2_pair"
    ref_p = pair_dir / "reference_1k.json"
    eng_p = pair_dir / "engine_tubingen_1k.json"
    long_run = R.get("gpt2_124m_engine_t512k_adamw_10k")
    if not (ref_p.exists() and eng_p.exists()):
        return None
    ref = json_mod.loads(ref_p.read_text())["losses"]
    eng = json_mod.loads(eng_p.read_text())["losses"]

    series = [Series(label="reference (pytorch, 3090)",
                     x=list(range(len(ref))), y=ref,
                     color=PALETTE[0], dashed=True),
              Series(label="engine @16 GiB (3090)",
                     x=list(range(len(eng))), y=eng, color=PALETTE[1])]
    rep = parity.compare(ref, eng, a_label="reference", b_label="engine")
    prows = [["engine vs pytorch reference (same box)",
              f"{rep.step0_abs:.4f}", f"{rep.max_abs:.4f}",
              f"{rep.final_abs:.4f}", f"{rep.ema_abs:.4f}",
              "ALIGNED" if rep.passed else "DIVERGED"]]
    if long_run is not None:
        chi = long_run.losses[:len(eng)]
        series.append(Series(label="engine @16 GiB (5090, long-run leg)",
                             x=list(range(len(chi))), y=chi,
                             color=PALETTE[2]))
        rep2 = parity.compare(eng, chi, a_label="engine-3090",
                              b_label="engine-5090")
        prows.append(["engine 5090 vs engine 3090 (cross-box, "
                      "sm120 vs sm86)",
                      f"{rep2.step0_abs:.4f}", f"{rep2.max_abs:.4f}",
                      f"{rep2.final_abs:.4f}", f"{rep2.ema_abs:.4f}",
                      "ALIGNED" if rep2.passed else "DIVERGED"])
    svg = svg_line_chart(series,
                         title="gpt2_124m — cross-entropy vs step "
                               "(first 1000 steps)",
                         xlabel="optimizer step", ylabel="mean CE loss")
    tbl = table_html(prows, ["comparison", "step0 Δ", "max Δ",
                             "final Δ", "ema Δ", "verdict"])
    note = ("<p>Same recipe, seed, and doc-aware token feed on every "
            "curve; byte-identical init through the weight bridge. The "
            "engine legs run different round geometries on different "
            "GPU generations — the loss trajectory is invariant to "
            "both (budget/geometry invariance), so the curves overlay "
            "within the cross-process numeric band.</p>")
    return (f"<section><h2>Reference agreement "
            f"(pytorch twin, two boxes)</h2>{note}"
            f"<div class='chart'>{svg}</div>{tbl}</section>")


def gpt2_optimizer_section(R: dict) -> str | None:
    """The 10k-step optimizer study: adamw baseline, shared-schedule
    muon twin, and the practice-lr muon leg, on identical init/data."""
    import json as json_mod

    legs = [("gpt2_124m_engine_t512k_adamw_10k", "adamw", PALETTE[0]),
            ("gpt2_124m_engine_t512k_muon_10k",
             "muon (shared schedule)", PALETTE[1]),
            ("gpt2_124m_engine_t512k_muon_hot_10k",
             "muon (practice lr 1.8e-3)", PALETTE[2])]
    have = [(k, R[k], lbl, c) for k, lbl, c in legs if k in R]
    if not have:
        return None
    evals_p = RESULTS / "gpt2_124m_evals.json"
    evals = (json_mod.loads(evals_p.read_text())
             if evals_p.exists() else {})

    series, rows = [], []
    for key, run, lbl, color in have:
        smooth = ema_curve(run.losses)
        series.append(Series(label=lbl,
                             x=list(range(len(smooth))), y=smooth,
                             color=color))
        ev = evals.get(key, {})
        val = ev.get("val_10000")
        rows.append([lbl, f"{smooth[-1]:.4f}", f"{min(run.losses):.4f}",
                     f"{val:.4f}" if val else "—",
                     f"{math.exp(val):.2f}" if val else "—",
                     f"{ev.get('val_9500'):.4f}"
                     if ev.get("val_9500") else "—"])
    svg = svg_line_chart(series,
                         title="gpt2_124m 10k × 512K tokens/step — "
                               "train CE, EMA(0.98)",
                         xlabel="optimizer step", ylabel="mean CE loss")
    tbl = table_html(rows, ["optimizer", "train EMA final", "train min",
                            "fineweb-val @10000", "val ppl",
                            "val @9500"])
    note = ("<p>Identical init (seed 11), token order, llm.c recipe "
            "(peak 6e-4, warmup 1000, cosine to 6e-5, betas 0.9/0.95, "
            "wd 0.1, no clip) on every leg; the optimizer is the only "
            "variable. The practice-lr leg gives muon matrices their "
            "own peak lr of 1.8e-3 riding the same schedule shape — "
            "the exact update-RMS of the first Muon speedrun record "
            "(their lr 3.6e-4 under a √max(r,c) scale ≡ 1.8e-3 under "
            "this repo's 0.2·√max(r,c) scale, any matrix shape); "
            "non-matrix params (embeddings, head, norms, biases) stay "
            "on adamw at the shared lr in both muon legs. Validation "
            "is 10.5M held-out fineweb-val tokens "
            "(tools/train/train_solo.py eval). Context, not a comparison "
            "claim: llm.c's 124M reaches ~3.29 val at 10B tokens — "
            "twice this study's 5.24B token budget.</p>")
    return (f"<section><h2>Optimizer study: adamw vs muon at 512K "
            f"tokens/step</h2>{note}"
            f"<div class='chart'>{svg}</div>{tbl}</section>")


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
        "gpt2", "gpt2 124M pretraining: reference agreement + "
        "optimizer study",
        "llm.c GPT-2 124M (LayerNorm+bias, learned positions, "
        "GELU-tanh, vocab 50304) — the nanogpt-speedrun baseline "
        "family. 512K tokens/step, doc-aware fineweb feed.",
        [gpt2_pair_section(R), gpt2_optimizer_section(R),
         throughput_section(R, ("gpt2_124m_engine",))])

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
        "l3_1b_distperf",
        "llama3-1B distributed training: the performance story",
        "RTX 5090 + RTX 3090 over direct 25 GbE, data-parallel with "
        "weighted rounds (3:1), global-denominator loss. Every number "
        "measured on this pair; correctness pinned by the 1000-step "
        "ALIGNED parity runs.",
        distperf_sections(R))

    write_report(
        "index", "Pretraining studies",
        "One report per study; combined throughput below.",
        ["<section><h2>Studies</h2><ul>"
         "<li><a href='llama3.html'>llama3 — parity, distributed DP, "
         "scaling ladder</a></li>"
         "<li><a href='qwen35.html'>qwen3.5 — parity</a></li>"
         "<li><a href='dsv3_2b.html'>dsv3 (MoE, 1.89B) — parity + "
         "load balancing</a></li>"
         "<li><a href='l3_1b_distperf.html'>llama3-1B distributed "
         "performance — ladder, sweep, calibration</a></li>"
         "</ul></section>",
         throughput_section(R)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
