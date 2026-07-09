"""Minimal dependency-free SVG line charts for the pretraining report.

Produces self-contained ``<svg>`` strings (no matplotlib, no external
assets) suitable for embedding in an HTML report / Artifact. Axis text uses
``currentColor`` so it adapts to a light/dark theme; series use an explicit
palette that reads on both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]


@dataclass
class Series:
    label: str
    x: list[float]
    y: list[float]
    color: str = ""
    dashed: bool = False
    markers: bool = False


def _downsample(xs, ys, n=250):
    if len(xs) <= n:
        return list(xs), list(ys)
    step = len(xs) / n
    idx = sorted({min(len(xs) - 1, int(i * step)) for i in range(n)} | {len(xs) - 1})
    return [xs[i] for i in idx], [ys[i] for i in idx]


def _nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1, 2, 2.5, 5, 10):
        if raw / mag <= m:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def svg_line_chart(series: list[Series], *, width=760, height=430,
                   title="", xlabel="", ylabel="", xlog=False,
                   ymin=None, ymax=None, xmin=None, xmax=None) -> str:
    ml, mr, mt, mb = 62, 150, 40, 52          # margins (mr leaves room for legend)
    pw, ph = width - ml - mr, height - mt - mb
    all_x = [x for s in series for x in s.x]
    all_y = [y for s in series for y in s.y]
    if not all_x:
        return f'<svg width="{width}" height="{height}"></svg>'
    tx = (lambda v: math.log10(max(v, 1e-9))) if xlog else (lambda v: v)
    x0 = tx(xmin if xmin is not None else min(all_x))
    x1 = tx(xmax if xmax is not None else max(all_x))
    y0 = ymin if ymin is not None else min(all_y)
    y1 = ymax if ymax is not None else max(all_y)
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    y0 -= (y1 - y0) * 0.04
    y1 += (y1 - y0) * 0.04

    def px(v):
        return ml + (tx(v) - x0) / (x1 - x0) * pw

    def py(v):
        return mt + (1 - (v - y0) / (y1 - y0)) * ph

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="12" '
           f'fill="currentColor">']
    if title:
        out.append(f'<text x="{ml}" y="20" font-size="14" font-weight="600">{title}</text>')
    # frame
    out.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" '
               f'stroke="currentColor" stroke-opacity="0.25"/>')
    # y ticks + gridlines
    for t in _nice_ticks(y0, y1):
        yy = py(t)
        if yy < mt - 1 or yy > mt + ph + 1:
            continue
        out.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" '
                   f'stroke="currentColor" stroke-opacity="0.08"/>')
        out.append(f'<text x="{ml - 8}" y="{yy + 4:.1f}" text-anchor="end" '
                   f'fill-opacity="0.7">{t:g}</text>')
    # x ticks
    if xlog:
        lo, hi = math.floor(x0), math.ceil(x1)
        xticks = [10 ** e for e in range(int(lo), int(hi) + 1)]
    else:
        xticks = _nice_ticks(x0, x1)
    for t in xticks:
        xx = px(t)
        if xx < ml - 1 or xx > ml + pw + 1:
            continue
        out.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt + ph}" '
                   f'stroke="currentColor" stroke-opacity="0.08"/>')
        lab = f"{t:g}"
        out.append(f'<text x="{xx:.1f}" y="{mt + ph + 18}" text-anchor="middle" '
                   f'fill-opacity="0.7">{lab}</text>')
    if xlabel:
        out.append(f'<text x="{ml + pw / 2:.0f}" y="{height - 8}" text-anchor="middle" '
                   f'fill-opacity="0.8">{xlabel}</text>')
    if ylabel:
        out.append(f'<text transform="translate(16,{mt + ph / 2:.0f}) rotate(-90)" '
                   f'text-anchor="middle" fill-opacity="0.8">{ylabel}</text>')
    # series
    for i, s in enumerate(series):
        color = s.color or PALETTE[i % len(PALETTE)]
        xs, ys = _downsample(s.x, s.y)
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        dash = ' stroke-dasharray="6 4"' if s.dashed else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                   f'stroke-width="2"{dash}/>')
        if s.markers:
            for x, y in zip(xs, ys):
                out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>')
        ly = mt + 6 + i * 20
        out.append(f'<line x1="{ml + pw + 12}" y1="{ly}" x2="{ml + pw + 34}" y2="{ly}" '
                   f'stroke="{color}" stroke-width="2"{dash}/>')
        out.append(f'<text x="{ml + pw + 40}" y="{ly + 4}" fill-opacity="0.85">{s.label}</text>')
    out.append("</svg>")
    return "".join(out)
