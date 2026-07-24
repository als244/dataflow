#!/usr/bin/env python
"""Run the throughput / sim-fidelity sweep, end to end, on whatever GPU is here.

    python reproducibility/throughput_fidelity/run_experiment.py
    python reproducibility/throughput_fidelity/run_experiment.py --help

Every knob is a flag with a default derived from the machine, so `--help` is the
configuration reference and nothing has to be discovered by reading source. The
stages run in order and each consumes the previous one's output:

    probe     what can this box hold?          -> env.json
    predict   the whole grid, and its edges    -> data/predict_measured_{opt}.jsonl
    select    which cells deserve GPU time     -> cells.json
    measure   what the engine actually does    -> data/measure_{opt}.jsonl
    shipped   do the documented commands work? -> logs/shipped_bench.log
    report    tables and figures               -> figs/

Stages are separate processes on purpose. Prediction profiles one geometry at a
time and a failure in one sequence length should cost that chunk, not the run,
so each is spawned, waited on, and reported independently.

Use --stages to resume or repeat part of a run, e.g. after adding budgets:

    ... run_experiment.py --stages predict,select,measure --budgets 4,8,16,32
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ALL_STAGES = ("probe", "predict", "select", "measure", "shipped", "report")


@dataclass
class Config:
    """Everything the sweep needs. Fields left None are derived by the probe
    from this machine, which is why the defaults are not written down here."""

    python: str = sys.executable
    preset: str | None = None          # None -> largest model the host can hold
    opts: tuple[str, ...] = ("adamw", "muon")
    seqs: tuple[int, ...] | None = None
    t_rounds: tuple[int, ...] | None = None
    t_steps: tuple[int, ...] | None = None
    budgets: tuple[float, ...] | None = None
    budget_step: float | None = None   # ladder ratio; None -> sqrt(2)
    host_share: float | None = None    # fraction of host RAM; None -> 0.8
    backing_gib: float | None = None   # explicit allowance, overrides the share
    target_cells: int = 18
    steps: int = 6
    stages: tuple[str, ...] = ALL_STAGES
    data: Path = field(default_factory=lambda: HERE / "data")
    logs: Path = field(default_factory=lambda: HERE / "logs")

    @property
    def env_json(self) -> Path:
        return HERE / "env.json"

    @property
    def cells_json(self) -> Path:
        return HERE / "cells.json"


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(cmd: list[str], *, log: Path | None = None, cwd: Path = REPO) -> bool:
    """One stage process. Output is teed to `log` so a failure can be read
    afterwards without re-running the stage that produced it."""
    if log is None:
        return subprocess.run(cmd, cwd=cwd).returncode == 0
    with open(log, "a") as fh:
        fh.write(f"\n$ {shlex.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=fh,
                              stderr=subprocess.STDOUT).returncode == 0


def csv(values) -> str:
    return ",".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values)


# ----------------------------------------------------------------- stages ---

def stage_probe(cfg: Config) -> dict:
    """Read this machine's real limits and choose what can be swept.

    Device memory comes from the driver; the host limit from the scheduler's
    grant if there is one, else a cgroup cap, else physical RAM — a compute node
    reports its full RAM even when the job owns a slice, and sizing a pinned
    slab from the node total gets the job killed."""
    say("probe — what can this box hold?")
    cmd = [cfg.python, str(HERE / "env_probe.py")]
    for flag, value in (("--preset", cfg.preset), ("--seqs", cfg.seqs),
                        ("--t-rounds", cfg.t_rounds), ("--t-steps", cfg.t_steps),
                        ("--budgets", cfg.budgets),
                        ("--budget-step", cfg.budget_step),
                        ("--host-share", cfg.host_share),
                        ("--backing-gib", cfg.backing_gib),
                        ("--steps", cfg.steps)):
        if value is None:
            continue
        cmd += [flag, csv(value) if isinstance(value, tuple) else str(value)]
    if not run(cmd, log=cfg.logs / "probe.log"):
        raise SystemExit("probe failed — see logs/probe.log")
    env = json.loads(cfg.env_json.read_text())
    say(f"  {env['device']}  ·  {env['preset']}  ·  host {env['host_limit_gib']} GiB "
        f"({env['host_limit_source']})")
    say(f"  budgets {env['budgets']}  allowance {env['backing_gib']} GiB")
    return env


def stage_predict(cfg: Config, env: dict) -> None:
    """Plan every cell on costs profiled on this GPU, one sequence length per
    process. Cells the planner cannot fit are recorded with the planner's
    reason rather than skipped, so the feasibility boundary is data."""
    for opt in cfg.opts:
        say(f"predict ({opt}) — every cost profiled on this GPU")
        out = cfg.data / f"predict_measured_{opt}.jsonl"
        out.write_text("")
        log = cfg.logs / f"predict_{opt}.log"
        for seq in env["seqs"]:
            ok = run([cfg.python, str(HERE / "sweep.py"),
                      "--mode", "predict-measured", "--preset", env["preset"],
                      "--opt", opt, "--seq", str(seq),
                      "--t-round", csv(env["t_rounds"]),
                      "--t-step", csv(env["t_steps"]),
                      "--budget", csv(env["budgets"]),
                      "--backing-gib", str(env["backing_gib"]),
                      "--out", str(out)], log=log)
            rows = sum(1 for _ in out.open()) if out.exists() else 0
            say(f"  seq {seq}: {'ok' if ok else 'FAILED'} ({rows} rows)")


def stage_select(cfg: Config) -> None:
    """Choose which cells are worth real GPU time: reduce over tokens-per-round
    to the frontier, cluster what remains by plan behaviour, keep a budget
    spine, and add dominated controls that test the reduction."""
    say("select — which cells deserve real GPU time?")
    run([cfg.python, str(HERE / "select_cells.py"),
         "--opts", ",".join(cfg.opts), "--target", str(cfg.target_cells)],
        log=cfg.logs / "select.log")
    if cfg.cells_json.exists():
        cells = json.loads(cfg.cells_json.read_text())
        say(f"  {len(cells)} cells x {len(cfg.opts)} optimizers")


def stage_measure(cfg: Config, env: dict) -> None:
    """Run the selected cells on the real engine and report each measurement
    beside the simulator's prediction for that same plan."""
    for opt in cfg.opts:
        say(f"measure ({opt})")
        out = cfg.data / f"measure_{opt}.jsonl"
        out.write_text("")
        ok = run([cfg.python, str(HERE / "sweep.py"), "--mode", "measure",
                  "--preset", env["preset"], "--opt", opt,
                  "--steps", str(cfg.steps), "--cells", str(cfg.cells_json),
                  "--backing-gib", str(env["backing_gib"]), "--out", str(out)],
                 log=cfg.logs / f"measure_{opt}.log")
        rows = sum(1 for _ in out.open()) if out.exists() else 0
        say(f"  {opt}: {'ok' if ok else 'FAILED'} ({rows} rows)")


def stage_shipped(cfg: Config, env: dict) -> None:
    """Run the repository's own bench commands at the spine geometry. The rest
    of this directory drives the library directly; this checks that what a
    reader would actually type still works at this scale."""
    say("shipped — do the documented commands still work?")
    if not cfg.cells_json.exists():
        say("  no cells.json; skipped")
        return
    cells = json.loads(cfg.cells_json.read_text())
    spine = [c for c in cells if "budget_spine" in c["spines"]] or cells
    budgets = sorted({c["budget"] for c in spine})[:2]
    c, log = spine[0], cfg.logs / "shipped_bench.log"
    common = ["--preset", env["preset"], "--t-round", str(c["t_round"]),
              "--tokens-step", str(c["t_step"]), "--budget", csv(budgets),
              "--seq-len", str(c["seq"])]
    ok = run([cfg.python, "tools/bench/predict_step.py", "--measured", *common,
              "--backing", str(env["backing_gib"])], log=log)
    ok &= run([cfg.python, "tools/bench/measure_step.py", "--measured-plan",
               *common, "--backing-gib", str(env["backing_gib"]),
               "--steps", str(cfg.steps)], log=log)
    say(f"  {'ok' if ok else 'had errors — see logs/shipped_bench.log'}")


def stage_report(cfg: Config) -> None:
    """Tables to stdout, figures to figs/."""
    say("report")
    run([cfg.python, str(HERE / "analyze.py")])
    for opt in cfg.opts:
        run([cfg.python, str(HERE / "make_plots.py"), opt],
            log=cfg.logs / "plots.log")


# ------------------------------------------------------------------- main ---

def numbers(text: str, cast):
    return tuple(cast(x) for x in text.split(","))


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Defaults marked (probed) are read from this machine, not "
               "hard-coded; run with no flags for a full sweep.")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter for the stage processes (default: this one)")
    p.add_argument("--preset", default=None,
                   help="model, e.g. l3_1b (probed: largest whose parameters, "
                        "optimizer state and gradients fit the host)")
    p.add_argument("--opts", default="adamw,muon",
                   help="optimizers to sweep; one of them roughly halves the run")
    p.add_argument("--seqs", default=None,
                   help="sequence lengths (probed: 1024,2048,4096,8192 capped "
                        "by the preset)")
    p.add_argument("--t-rounds", default=None,
                   help="tokens per round — grad-accumulation granularity "
                        "(probed: 8192,16384,32768,65536). Must divide by the "
                        "sequence length and into tokens per step")
    p.add_argument("--t-steps", default=None,
                   help="tokens per optimizer step (probed: scaled to the device)")
    p.add_argument("--budgets", default=None,
                   help="GPU memory budgets in GiB, explicitly (probed: a "
                        "ladder from a one-task floor to 0.85 x device)")
    p.add_argument("--budget-step", type=float, default=None,
                   help="ratio between budget rungs instead of replacing the "
                        "ladder: 2 for octaves, 1.2 for fine (default 1.414)")
    p.add_argument("--host-share", type=float, default=None,
                   help="fraction of host memory offered as the allowance "
                        "(default 0.8)")
    p.add_argument("--backing-gib", type=float, default=None,
                   help="host allowance outright, ignoring --host-share")
    p.add_argument("--target-cells", type=int, default=18,
                   help="how many cells get real GPU runs (default: 18)")
    p.add_argument("--steps", type=int, default=6,
                   help="steps per measured cell; first 3 are warmup, keep >= 4")
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help=f"stages to run, in order, from {','.join(ALL_STAGES)}")
    a = p.parse_args(argv)

    stages = tuple(s.strip() for s in a.stages.split(","))
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        p.error(f"unknown stage(s) {unknown}; choose from {ALL_STAGES}")
    if a.steps < 4:
        p.error("--steps must be at least 4 (the first 3 are warmup)")
    return Config(
        python=a.python, preset=a.preset,
        opts=tuple(o.strip() for o in a.opts.split(",")),
        seqs=numbers(a.seqs, int) if a.seqs else None,
        t_rounds=numbers(a.t_rounds, int) if a.t_rounds else None,
        t_steps=numbers(a.t_steps, int) if a.t_steps else None,
        budgets=numbers(a.budgets, float) if a.budgets else None,
        budget_step=a.budget_step, host_share=a.host_share,
        backing_gib=a.backing_gib, target_cells=a.target_cells,
        steps=a.steps, stages=stages)


def main(argv=None) -> int:
    cfg = parse_args(argv)
    cfg.data.mkdir(parents=True, exist_ok=True)
    cfg.logs.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "?"
    say(f"host {__import__('socket').gethostname()}  repo {head}  "
        f"stages {','.join(cfg.stages)}")

    env = json.loads(cfg.env_json.read_text()) if cfg.env_json.exists() else {}
    if "probe" in cfg.stages:
        env = stage_probe(cfg)
    elif not env:
        raise SystemExit("no env.json — run the probe stage first")

    if "predict" in cfg.stages:
        stage_predict(cfg, env)
    if "select" in cfg.stages:
        stage_select(cfg)
    if "measure" in cfg.stages:
        stage_measure(cfg, env)
    if "shipped" in cfg.stages:
        stage_shipped(cfg, env)
    if "report" in cfg.stages:
        stage_report(cfg)
    say(f"done — data in {cfg.data}, logs in {cfg.logs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
