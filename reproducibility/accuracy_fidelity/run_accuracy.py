#!/usr/bin/env python
"""Loss-curve fidelity across device budgets, for any model family.

    python reproducibility/accuracy_fidelity/run_accuracy.py --preset gpt2_124m \
        --batch 64 --ga-rounds 8 --steps 10000 --peak-lr 6e-4 --auto-budgets

The question this answers: does training the SAME model, recipe, and data
stream through the engine at different device budgets — save-everything,
recompute-boundary, deep-offload — produce the SAME loss curve, and does
that curve match the family's independent pure-torch twin
(``reference_models``, run through ``tools/train train.py reference``)?

Invariance laws the driver enforces so legs are mathematically comparable
(each was a real footgun when this was driven by hand):

- ONE schedule horizon: every leg gets the same ``--steps``; warmup and the
  cosine tail derive from it (steps/10, peak/10), so a shorter comparison
  run must still be launched at the full horizon.
- ONE data geometry: ``--batch`` and ``--ga-rounds`` are passed explicitly
  to every leg, reference included — preset defaults are NOT trusted,
  because tokens/step = batch x seq x ga is part of the math.
- ONE seed, ONE data spec: byte-identical init and one deterministic
  stream (doc-aware fineweb by default).

Stages (each resumable; ``--stages`` to run a subset):

    plan       choose the regimes (see stage table in the README)
    engine     one training leg per (budget, t_round), FIRST — the
               budget-invariance evidence banks before the long
               reference anchor runs
    reference  the pure-torch twin anchor
    report     verdicts + the val ladder

    (docstring stage detail below predates the ordering change; the
    README is authoritative)

    plan       roofline-plan the budget ladder (CPU, seconds): predicted
               s/step, recompute share, peak fast bytes, offload traffic
               per rung — and, with --auto-budgets, CHOOSE the three
               interesting rungs (comfortable / boundary / deep) so the
               training legs stress genuinely different plan behavior
    reference  the pure-torch twin at the same recipe -> the anchor curve
    engine     one training leg per budget (sequential; checkpoints land
               under results/pretrain/checkpoints — symlink that to cold
               storage when disk is tight)
    report     per-leg verdicts (engine-vs-reference and engine-vs-engine
               pairwise) using the same drift metrics the parity gates
               use: step0 / max / final / EMA deltas -> REPORT.md

Every leg is a ``tools/train/train.py`` subprocess (the shipped tool, so a
leg here is byte-identical to a hand-launched run); the driver tails each
leg's log to narrate progress and refine an ETA from the measured step
rate. Curves land under ``<results>/curves/`` as train.py's own run-curve
JSON; a leg whose curve already holds >= steps losses is skipped, and an
interrupted leg with checkpoints resumes via ``--resume auto``.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAIN = REPO / "tools" / "train" / "train.py"


@dataclass
class Config:
    python: str = sys.executable
    preset: str = "gpt2_124m"
    opt: str = "adamw"
    steps: int = 10000
    peak_lr: float = 6e-4
    seed: int = 11
    batch: int | None = None            # manual override only
    ga_rounds: int | None = None        # manual override only
    t_step: int = 65536
    max_seq_len: int | None = None
    data: str | None = None
    packing_policy: str | None = None
    allow_round_split: bool = False
    budgets: tuple[float, ...] = ()
    auto_budgets: bool = False
    backing_gib: float = 24.0
    ckpt_every: int = 1000
    grad_checkpoint: bool = False
    stages: tuple[str, ...] = ("plan", "engine", "reference", "report")
    resume: bool = False
    results: Path = field(default_factory=lambda: HERE / "results_accuracy")

    @property
    def curves(self) -> Path:
        return self.results / "curves"

    @property
    def logs(self) -> Path:
        return self.results / "logs"

    @property
    def metrics(self) -> Path:
        return self.results / "metrics"

    def leg_name(self, kind: str, budget: float | None = None) -> str:
        base = f"{self.preset}_{self.opt}"
        if kind == "reference":
            return f"{base}_reference"
        return f"{base}_b{budget:g}"


ALL_STAGES = ("plan", "engine", "reference", "report")


METRICS_SCHEMA = {
    "description": "one JSON object per parsed training step line, "
                   "appended live while the leg runs (and on resume)",
    "fields": {
        "ts": "unix seconds at parse time",
        "step": "optimizer step index",
        "loss": "mean CE loss the trainer logged for this step",
        "lr": "learning rate at this step",
        "tok_s": "trainer-reported tokens/second",
        "step_s": "trainer-reported seconds/step",
        "gpu_mem_mib": "nvidia-smi device memory at the poll (whole GPU)",
        "peak_gpu_mem_mib": "running max of gpu_mem_mib over this leg",
    },
}


def update_manifest(cfg: Config, **patch) -> None:
    """results_accuracy/manifest.json — THE structured index: campaign
    config, chosen (budget, t_round) pairs, per-leg status and files,
    stage timestamps. Rewritten atomically at every stage boundary so a
    reader (or a resume) never depends on parsing logs."""
    path = cfg.results / "manifest.json"
    m = {}
    if path.exists():
        try:
            m = json.load(open(path))
        except ValueError:
            m = {}
    m.setdefault("campaign", {
        "preset": cfg.preset, "opt": cfg.opt, "steps": cfg.steps,
        "t_step": cfg.t_step, "max_seq_len": cfg.max_seq_len,
        "peak_lr": cfg.peak_lr, "seed": cfg.seed,
        "data": cfg.data or "pipeline default (fixed windows)",
        "ckpt_every": cfg.ckpt_every,
    })
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(m.get(k), dict):
            m[k].update(v)
        else:
            m[k] = v
    m["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2) + "\n")
    tmp.replace(path)
    (cfg.metrics / "schema.json").parent.mkdir(parents=True, exist_ok=True)
    (cfg.metrics / "schema.json").write_text(
        json.dumps(METRICS_SCHEMA, indent=2) + "\n")


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 \
        else f"{s // 60}:{s % 60:02d}"


def seq_of(cfg: Config) -> int:
    from dataflow_training.run.presets import resolve_preset

    base = resolve_preset(cfg.preset)
    return cfg.max_seq_len or getattr(base, "seq_len", None) \
        or getattr(base, "max_seq_len")


def shaped(cfg: Config, t_round: int):
    """Model config at (seq, t_round, t_step): batch = t_round/seq,
    ga = t_step/t_round. Step composition is t_step-determined under the
    fixed-window feed, so t_round is free to differ per leg — it only
    regroups rounds, and grad accumulation is certified invariant to
    the grouping."""
    from dataclasses import replace

    from dataflow_training.run.presets import resolve_preset

    base = resolve_preset(cfg.preset)
    seq = seq_of(cfg)
    assert t_round % seq == 0 and cfg.t_step % t_round == 0, \
        (seq, t_round, cfg.t_step)
    overrides = {"batch": t_round // seq,
                 "grad_accum_rounds": cfg.t_step // t_round}
    if hasattr(base, "seq_len"):
        overrides["seq_len"] = seq
    if hasattr(base, "max_seq_len"):
        overrides["max_seq_len"] = max(getattr(base, "max_seq_len"), seq)
    return replace(base, **overrides)


def t_round_candidates(cfg: Config) -> list[int]:
    seq = seq_of(cfg)
    out = []
    t = seq
    while t <= cfg.t_step:
        if cfg.t_step % t == 0:
            out.append(t)
        t *= 2
    return out


# ------------------------------------------------------------------ plan ---

SHARE_LADDER = (0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.05)


def probe_rungs(cfg: Config, plan_at_budget, measured, profile_cache):
    """A scale-free ladder: plan once effectively unbounded to learn the
    model's own save-everything peak, then probe log-spaced SHARES of it
    down to the feasibility floor. The same rule sizes gpt2-124M and an
    8B the same way — no absolute GiB constants to retune per family."""
    from dataclasses import replace

    UNBOUNDED_GIB = 10_000.0    # a budget no plan can bind against —
    # plan_at_budget's second positional is DEVICE GiB, never steps
    # (cells are one-step programs by construction, num_steps=1)
    cell = replace(shaped(cfg, t_round_candidates(cfg)[-1]), num_steps=1)
    top = plan_at_budget(cell, UNBOUNDED_GIB, recompute=False,
                         measured=measured, profile_cache=profile_cache,
                         backing_gib=cfg.backing_gib)
    peak_gib = top.peak_fast_bytes / 1024 ** 3
    rungs = tuple(round(peak_gib * s, 2) for s in SHARE_LADDER)
    say(f"  save-everything peak {peak_gib:.2f} GiB -> ladder {rungs}")
    return rungs


def stage_plan(cfg: Config) -> tuple[float, ...]:
    """Roofline-plan the ladder; pick (or just describe) the budgets.

    Selection rule with --auto-budgets: DEEP = the smallest feasible rung
    (heaviest offload+recompute the planner will accept), BOUNDARY = the
    largest rung that still recomputes (just under save-everything),
    COMFORTABLE = the smallest rung with zero recompute. Three regimes,
    three genuinely different plans — the invariance claim is only
    interesting if the plans differ."""
    from dataclasses import replace

    plan_path = cfg.results / "plan.json"
    if cfg.resume and cfg.auto_budgets and plan_path.exists():
        chosen = json.load(open(plan_path)).get("chosen")
        if chosen:
            say(f"plan — resuming with recorded (budget, t_round) {chosen}")
            return tuple((c["budget"], c["t_round"]) for c in chosen)
    from dataflow_training.run.driver import plan_at_budget

    try:
        import torch
        measured = torch.cuda.is_available()
    except Exception:
        measured = False
    say(f"plan — frontier t_round per budget, "
        f"({'PROFILED costs on this GPU' if measured else 'roofline '
           'fallback, no GPU here — costs indicative only'})")
    profile_cache: dict = {}
    rungs = cfg.budgets or probe_rungs(cfg, plan_at_budget, measured,
                                       profile_cache)
    candidates = ([cfg.batch * seq_of(cfg)] if cfg.batch
                  else t_round_candidates(cfg))
    rows = []
    for b in rungs:
        best = None
        reason = None
        for tr in candidates:
            cell = replace(shaped(cfg, tr), num_steps=1)
            try:
                planned = plan_at_budget(cell, b, recompute=True,
                                         measured=measured,
                                         profile_cache=profile_cache,
                                         backing_gib=cfg.backing_gib)
            except ValueError as exc:
                reason = str(exc).splitlines()[0][:90]
                continue
            if best is None or planned.makespan_us < best[1].makespan_us:
                best = (tr, planned)
        if best is None:
            rows.append({"budget": b, "infeasible": reason or "no fit"})
            say(f"  b{b:g}: infeasible — {rows[-1]['infeasible']}")
            continue
        tr, planned = best
        n_rc = sum(1 for lv in (planned.recompute_levels or {}).values() if lv)
        stats = planned.transfer_stats or {}
        h2d = stats.get("from_slow", {}).get("bytes", 0) / 1e9
        rows.append({
            "budget": b,
            "t_round": tr,
            "pred_step_s": planned.makespan_us / 1e6,
            "recompute_objects": n_rc,
            "peak_fast_gib": planned.peak_fast_bytes / 1024 ** 3,
            "h2d_gb_per_step": h2d,
        })
        say(f"  b{b:g}: best tr{tr}  pred "
            f"{rows[-1]['pred_step_s']:.2f}s/step  recompute objs {n_rc}  "
            f"peak fast {rows[-1]['peak_fast_gib']:.2f} GiB  "
            f"h2d {h2d:.1f} GB/step")
    cfg.results.mkdir(parents=True, exist_ok=True)
    plan_path = cfg.results / "plan.json"
    by_budget = {r["budget"]: r for r in rows if "t_round" in r}
    if not cfg.auto_budgets:
        chosen = [{"budget": b, "t_round": by_budget[b]["t_round"]}
                  for b in cfg.budgets if b in by_budget]
        plan_path.write_text(json.dumps(
            {"preset": cfg.preset, "t_step": cfg.t_step,
             "seq": seq_of(cfg), "rungs": rows, "chosen": chosen},
            indent=2))
        return tuple((c["budget"], c["t_round"]) for c in chosen)
    feasible = [r for r in rows if "pred_step_s" in r]
    if not feasible:
        raise SystemExit("no feasible rung in the probe ladder")
    saving = [r for r in feasible if r["recompute_objects"]]
    clean = [r for r in feasible if not r["recompute_objects"]]
    picks = [feasible[0]]
    if saving and saving[-1] is not picks[0]:
        picks.append(saving[-1])
    if clean and clean[0] not in picks:
        picks.append(clean[0])
    chosen = [{"budget": r["budget"], "t_round": r["t_round"]}
              for r in picks]
    say(f"  chosen (budget, t_round): "
        f"{[(c['budget'], c['t_round']) for c in chosen]} "
        f"(deep / boundary / comfortable)")
    plan_path.write_text(json.dumps(
        {"preset": cfg.preset, "t_step": cfg.t_step, "seq": seq_of(cfg),
         "rungs": rows, "chosen": chosen}, indent=2))
    update_manifest(cfg, plan={"chosen": chosen, "rungs": rows})
    return tuple((c["budget"], c["t_round"]) for c in chosen)


def outlook(cfg: Config, budgets) -> None:
    """Upfront ETA per aspect, from the plan's own predictions: each
    engine leg = steps x its predicted s/step; the reference twin has no
    plan, so it starts as a labeled ~2.5x-of-comfortable estimate and is
    REFINED from its measured rate as soon as step lines appear."""
    plan_path = cfg.results / "plan.json"
    if not (budgets and plan_path.exists()):
        return
    rows = {r["budget"]: r for r in json.load(open(plan_path))["rungs"]
            if "pred_step_s" in r}
    total = 0.0
    say("campaign outlook (refined live once each leg reports rate):")
    if "engine" in cfg.stages:
        for b, tr in budgets:
            r = rows.get(b)
            if not r:
                continue
            eta_h = cfg.steps * r["pred_step_s"] / 3600
            total += eta_h
            say(f"  engine b{b:g} tr{tr}: ~{eta_h:.1f}h "
                f"({r['pred_step_s']:.2f}s/step predicted)")
    if "reference" in cfg.stages:
        base = rows.get(budgets[-1][0])
        if base:
            ref_h = cfg.steps * base["pred_step_s"] * 2.5 / 3600
            total += ref_h
            say(f"  reference: ~{ref_h:.1f}h (rough 2.5x of the "
                f"comfortable engine leg; measured rate will correct)")
    say(f"  campaign total: ~{total:.1f}h GPU")


# ------------------------------------------------------------- training ---

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)\s+lr\s+([0-9.e+-]+)"
    r"\s+(\d+)\s+tok/s(?:.*?\((\d+\.\d+)s\))?")


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def curve_complete(path: Path, steps: int) -> bool:
    if not path.exists():
        return False
    try:
        return len(json.load(open(path)).get("losses", [])) >= steps
    except (ValueError, OSError):
        return False


def run_leg(cfg: Config, name: str, cmd: list[str]) -> None:
    """One training subprocess, narrated and instrumented.

    The leg's stdout streams to logs/<name>.log; every step line is also
    parsed into metrics/<name>.jsonl as it appears — {ts, step, loss, lr,
    tok_s, step_s} plus a device-memory sample per poll and its running
    peak — so tok/s, memory, and losses are on disk and greppable while
    the leg runs, and a resumed leg APPENDS to the same files. The ETA
    narration derives from the measured step rate. A failed leg halts
    the campaign: it is a fault, never a fidelity result."""
    cfg.logs.mkdir(parents=True, exist_ok=True)
    cfg.metrics.mkdir(parents=True, exist_ok=True)
    log = cfg.logs / f"{name}.log"
    met = cfg.metrics / f"{name}.jsonl"
    say(f"  $ {shlex.join(cmd)}")
    update_manifest(cfg, legs={name: {
        "status": "running", "started": time.strftime("%H:%M:%S"),
        "log": str(log)}})
    t0 = time.time()
    peak_mib = 0
    offset = log.stat().st_size if log.exists() else 0
    with open(log, "a") as fh, open(met, "a") as mh:
        fh.write(f"\n$ {shlex.join(cmd)}\n")
        fh.flush()
        child = subprocess.Popen(cmd, cwd=REPO, stdout=fh,
                                 stderr=subprocess.STDOUT)
        last_report = 0.0
        last_line = None
        while True:
            done = child.poll() is not None
            with open(log) as rd:
                rd.seek(offset)
                fresh = rd.read()
                offset = rd.tell()
            mem = gpu_mem_used_mib()
            if mem:
                peak_mib = max(peak_mib, mem)
            for m in STEP_RE.finditer(fresh):
                step, total, loss, lr, tok_s, step_s = m.groups()
                if step_s is None:
                    # reference lines carry no wall suffix — derive it
                    step_s = cfg.t_step / max(int(tok_s), 1)
                last_line = (int(step), int(total), float(step_s))
                mh.write(json.dumps({
                    "ts": round(time.time(), 1), "step": int(step),
                    "loss": float(loss), "lr": float(lr),
                    "tok_s": int(tok_s), "step_s": float(step_s),
                    "gpu_mem_mib": mem, "peak_gpu_mem_mib": peak_mib,
                }) + "\n")
            mh.flush()
            if done:
                break
            first = last_report == 0.0 and last_line
            if (first or time.time() - last_report >= 120) and last_line:
                last_report = time.time()
                step, total, rate = last_line
                remain = (total - step) * rate
                say(f"  [{name}] step {step}/{total} "
                    f"({100 * step / total:.0f}%)  {rate:.2f}s/step  "
                    f"peak mem {peak_mib} MiB  eta {remain / 3600:.1f}h")
            time.sleep(10)
    if child.returncode != 0:
        update_manifest(cfg, legs={name: {"status": "FAILED",
                                          "log": str(log)}})
        raise SystemExit(
            f"leg {name} FAILED (exit {child.returncode}) — see {log}; "
            f"a failed leg is a fault, not a fidelity result")
    say(f"  [{name}] done in {elapsed(t0)}  peak mem {peak_mib} MiB")
    update_manifest(cfg, legs={name: {
        "status": "complete", "wall_s": int(time.time() - t0),
        "peak_gpu_mem_mib": peak_mib, "log": str(log),
        "metrics": str(met)}})


def data_flags(cfg: Config) -> list[str]:
    """Empty by default: the normal train.py pipeline (varlen packing,
    default data source) IS the configuration under test. Flags appear
    only when a campaign overrides them deliberately."""
    flags = []
    if cfg.data:
        flags += ["--data", cfg.data]
    if cfg.packing_policy:
        flags += ["--packing-policy", cfg.packing_policy]
    if cfg.allow_round_split:
        flags.append("--allow-round-split")
    return flags


def geometry_flags(cfg: Config, t_round: int) -> list[str]:
    seq = seq_of(cfg)
    return ["--batch", str(t_round // seq),
            "--ga-rounds", str(cfg.t_step // t_round),
            "--max-seq-len", str(seq)]


def stage_reference(cfg: Config, budgets) -> None:
    say(f"reference ({cfg.preset}) — the pure-torch twin anchor")
    # the reference has no budget; it runs the COMFORTABLE rung's
    # frontier geometry (last chosen pair) — mathematically irrelevant
    # under the fixed-window feed, realistic for wall-clock
    t_round = budgets[-1][1] if budgets else cfg.t_step // 4
    out = cfg.curves / f"{cfg.leg_name('reference')}.json"
    if not cfg.resume and out.exists():
        out.unlink()
    if curve_complete(out, cfg.steps):
        say(f"  complete curve already at {out.name} — skipping")
        return
    cfg.curves.mkdir(parents=True, exist_ok=True)
    cmd = [cfg.python, "-u", str(TRAIN), "reference",
           "--preset", cfg.preset, "--steps", str(cfg.steps),
           "--peak-lr", f"{cfg.peak_lr:g}", "--opt", cfg.opt,
           "--checkpoint-every", str(cfg.ckpt_every),
           "--out", str(out)] + geometry_flags(cfg, t_round) + data_flags(cfg)
    if cfg.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    ck = REPO / "results" / "pretrain" / "checkpoints" / out.stem
    if cfg.resume and ck.exists() and any(ck.iterdir()):
        cmd.append("--resume")
    run_leg(cfg, cfg.leg_name("reference"), cmd)


def stage_engine(cfg: Config, budgets) -> None:
    for b, t_round in budgets:
        say(f"engine ({cfg.preset} @ {b:g} GiB, tr{t_round})")
        out = cfg.curves / f"{cfg.leg_name('engine', b)}.json"
        if not cfg.resume and out.exists():
            out.unlink()
        if curve_complete(out, cfg.steps):
            say(f"  complete curve already at {out.name} — skipping")
            continue
        cfg.curves.mkdir(parents=True, exist_ok=True)
        cmd = [cfg.python, "-u", str(TRAIN), "train",
               "--preset", cfg.preset, "--steps", str(cfg.steps),
               "--peak-lr", f"{cfg.peak_lr:g}", "--opt", cfg.opt,
               "--seed", str(cfg.seed),
               "--fast-budget", f"{b:g}",
               "--backing-budget", f"{cfg.backing_gib:g}",
               "--checkpoint-every", str(cfg.ckpt_every),
               "--out", str(out)] + geometry_flags(cfg, t_round) \
            + data_flags(cfg)
        ck = REPO / "results" / "pretrain" / "checkpoints" / out.stem
        if cfg.resume and ck.exists() and any(ck.iterdir()):
            cmd += ["--resume", "auto"]
        run_leg(cfg, cfg.leg_name("engine", b), cmd)


# ---------------------------------------------------------------- report ---

def stage_report(cfg: Config, budgets) -> None:
    say("report — drift verdicts")
    from dataflow_training.run import parity

    def load(name):
        p = cfg.curves / f"{name}.json"
        return json.load(open(p))["losses"] if p.exists() else None

    ref = load(cfg.leg_name("reference"))
    legs = {b: load(cfg.leg_name("engine", b)) for b, _tr in budgets}
    lines = [f"# accuracy fidelity — {cfg.preset} ({cfg.opt})", ""]
    lines += [f"steps {cfg.steps} · tokens/step {cfg.t_step} · seq "
              f"{seq_of(cfg)} · per-leg frontier t_round "
              f"{[(b, tr) for b, tr in budgets]} · peak lr "
              f"{cfg.peak_lr:g} · seed {cfg.seed} · data "
              f"`{cfg.data or 'pipeline default (fixed windows)'}`", ""]
    lines += ["| comparison | step0 Δ | max Δ | final Δ | EMA Δ | verdict |",
              "|---|---|---|---|---|---|"]

    def row(label, a, b):
        n = min(len(a), len(b))
        rep = parity.compare(a[:n], b[:n], a_label=label.split(" vs ")[0],
                             b_label=label.split(" vs ")[1])
        m = rep.metrics if hasattr(rep, "metrics") else {}
        step0 = abs(a[0] - b[0])
        mx = max(abs(x - y) for x, y in zip(a[:n], b[:n]))
        fin = abs(a[n - 1] - b[n - 1])
        k = 0.98
        ea = eb = 0.0
        for x, y in zip(a[:n], b[:n]):
            ea = k * ea + (1 - k) * x
            eb = k * eb + (1 - k) * y
        verdict = "ALIGNED" if rep.passed else "DIVERGED"
        lines.append(f"| {label} (n={n}) | {step0:.4f} | {mx:.4f} | "
                     f"{fin:.4f} | {abs(ea - eb):.4f} | {verdict} |")
        say(f"  {label}: max Δ {mx:.4f} final Δ {fin:.4f} -> {verdict}")

    for b, curve in legs.items():
        if ref and curve:
            row(f"reference vs engine@{b:g}GiB", ref, curve)
    have = [(b, c) for b, c in legs.items() if c]
    for i in range(len(have)):
        for j in range(i + 1, len(have)):
            row(f"engine@{have[i][0]:g}GiB vs engine@{have[j][0]:g}GiB",
                have[i][1], have[j][1])
    val = val_ladder(cfg, budgets)
    if val:
        lines += ["", "## held-out val loss (every retained checkpoint)",
                  ""]
        steps = sorted({s for rows in val.values() for s in rows})
        lines.append("| leg | " + " | ".join(str(s) for s in steps) + " |")
        lines.append("|---|" + "---|" * len(steps))
        for leg, rows in val.items():
            cells = [f"{rows[s]:.4f}" if s in rows else "-" for s in steps]
            lines.append(f"| {leg} | " + " | ".join(cells) + " |")
    lines += ["", f"curves: `{cfg.curves}/`", ""]
    out = cfg.results / "REPORT.md"
    out.write_text("\n".join(lines))
    update_manifest(cfg, report={"path": str(out), "val": val})
    say(f"  wrote {out}")


def val_ladder(cfg: Config, budgets) -> dict:
    """Held-out val loss at EVERY retained checkpoint of every leg
    (~30-60s each: boot + one forward-only val pass), via the shipped
    eval_checkpoint tool. Results land in metrics/val_<leg>.jsonl as
    they are produced, so an interrupted ladder resumes where it
    stopped."""
    eval_tool = REPO / "tools" / "train" / "eval_checkpoint.py"
    ck_root = REPO / "results" / "pretrain" / "checkpoints"
    out: dict = {}
    names = [cfg.leg_name("reference")] + \
        [cfg.leg_name("engine", b) for b, _tr in budgets]
    for name in names:
        ck = ck_root / name
        if not ck.exists():
            continue
        done_path = cfg.metrics / f"val_{name}.jsonl"
        done = {}
        if done_path.exists():
            for line in open(done_path):
                try:
                    r = json.loads(line)
                    done[r["step"]] = r["val_loss"]
                except (ValueError, KeyError):
                    continue
        rows = dict(done)
        steps = sorted(int(d.name.split("_")[1]) for d in ck.iterdir()
                       if d.name.startswith("step_"))
        for s in steps:
            if s in rows:
                continue
            cmd = [cfg.python, str(eval_tool), name,
                   "--preset", cfg.preset, "--step", str(s)]
            if cfg.max_seq_len:
                cmd += ["--max-seq-len", str(cfg.max_seq_len)]
            r = subprocess.run(cmd, cwd=REPO, capture_output=True,
                               text=True)
            m = re.search(r"val[ _-]?loss[ =:]+([0-9.]+)",
                          r.stdout + r.stderr, re.I)
            if not m:
                say(f"  [val {name} step {s}] UNPARSED — see eval output")
                continue
            v = float(m.group(1))
            rows[s] = v
            with open(done_path, "a") as fh:
                fh.write(json.dumps({"step": s, "val_loss": v,
                                     "ts": round(time.time(), 1)}) + "\n")
            say(f"  [val {name}] step {s}: {v:.4f}")
        if rows:
            out[name] = rows
    return out


# ------------------------------------------------------------------ main ---

def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--preset", default="gpt2_124m")
    p.add_argument("--opt", default="adamw", choices=["adamw", "muon"])
    p.add_argument("--steps", type=int, default=10000,
                   help="THE schedule horizon — every leg trains this many "
                        "steps (warmup/cosine derive from it)")
    p.add_argument("--peak-lr", type=float, default=6e-4)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--batch", type=int, default=None,
                   help="rows per round — pass it: tokens/step is part of "
                        "the math being compared")
    p.add_argument("--ga-rounds", type=int, default=None)
    p.add_argument("--t-step", type=int, default=65536,
                   help="tokens per optimizer step — THE campaign "
                        "constant that fixes step composition")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="sequence length override (family-mapped)")
    p.add_argument("--data", default=None,
                   help="data spec override; default = the train "
                        "pipeline's own default (varlen packing)")
    p.add_argument("--packing-policy", default=None)
    p.add_argument("--budgets", default=None,
                   help="device budgets GiB, comma-separated; omit with "
                        "--auto-budgets to let the plan stage choose")
    p.add_argument("--auto-budgets", action="store_true",
                   help="plan stage picks deep/boundary/comfortable rungs")
    p.add_argument("--backing-gib", type=float, default=24.0)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="reference twin recomputes activations in "
                        "backward — only for models whose twin cannot "
                        "otherwise fit the device (slows the leg)")
    p.add_argument("--ckpt-every", type=int, default=1000,
                   help="sparse checkpoints for leg resume; they land under "
                        "results/pretrain/checkpoints (symlink to cold "
                        "storage when disk is tight); 0 disables")
    p.add_argument("--stages", default=",".join(ALL_STAGES))
    p.add_argument("--resume", action="store_true",
                   help="keep finished curves, resume interrupted legs "
                        "from their checkpoints")
    p.add_argument("--results", default=None)
    a = p.parse_args(argv)
    stages = tuple(s.strip() for s in a.stages.split(","))
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        p.error(f"unknown stage(s) {unknown}; choose from {ALL_STAGES}")
    if not a.budgets and not a.auto_budgets and \
            ("engine" in stages or "report" in stages):
        p.error("pass --budgets a,b,c or --auto-budgets")
    return Config(
        python=a.python, preset=a.preset, opt=a.opt, steps=a.steps,
        peak_lr=a.peak_lr, seed=a.seed, batch=a.batch,
        ga_rounds=a.ga_rounds, t_step=a.t_step,
        max_seq_len=a.max_seq_len, data=a.data,
        packing_policy=a.packing_policy,
        budgets=tuple(float(x) for x in a.budgets.split(",")) if a.budgets
        else (),
        auto_budgets=a.auto_budgets, backing_gib=a.backing_gib,
        ckpt_every=a.ckpt_every, grad_checkpoint=a.grad_checkpoint,
        stages=stages, resume=a.resume,
        **({"results": Path(a.results).resolve()} if a.results else {}))


def main(argv=None) -> int:
    cfg = parse_args(argv)
    cfg.results.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "src"))
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short",
                           "HEAD"], capture_output=True,
                          text=True).stdout.strip() or "?"
    say(f"host {__import__('socket').gethostname()}  repo {head}  "
        f"preset {cfg.preset}  stages {','.join(cfg.stages)}")
    t_run = time.time()
    budgets = cfg.budgets
    if "plan" in cfg.stages:
        t0 = time.time()
        budgets = stage_plan(cfg) or budgets
        say(f"  plan took {elapsed(t0)}")
        outlook(cfg, budgets)
    else:
        plan_path = cfg.results / "plan.json"
        if plan_path.exists():
            chosen = json.load(open(plan_path)).get("chosen") or []
            budgets = tuple((c["budget"], c["t_round"]) for c in chosen)
            say(f"using recorded plan: {[(b, tr) for b, tr in budgets]}")
        elif cfg.auto_budgets:
            raise SystemExit(
                "--auto-budgets without the plan stage needs an existing "
                "plan.json (run the plan stage once)")
    if "engine" in cfg.stages:
        t0 = time.time()
        stage_engine(cfg, budgets)
        say(f"  engine took {elapsed(t0)}")
    if "reference" in cfg.stages:
        t0 = time.time()
        stage_reference(cfg, budgets)
        say(f"  reference took {elapsed(t0)}")
    if "report" in cfg.stages:
        t0 = time.time()
        stage_report(cfg, budgets)
        say(f"  report took {elapsed(t0)}")
    say(f"done in {elapsed(t_run)} — everything under {cfg.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
