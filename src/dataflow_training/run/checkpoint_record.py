"""The checkpoint record (format 2): one format at every world
size.

``<run>/step_NNNNNN/checkpoint_record.json`` — written LAST by the conductor as
the completeness marker — describes the whole checkpoint:

    {
      "format": 2,
      "step", "seed", "world",
      "data_cursor": {...},        # ONE global cursor
      "losses": [...],             # global step means
      "save_plan": {object_id: [{rank, lo, hi, role}]},
      "artifacts": ["rank0", ...], # relative per-rank snapshot dirs
      "launch": {argv, resolved, data, git, env, ranks, programs}
    }

Each rank's artifact is an ordinary engine snapshot written by THAT
rank per its responsibility (ranged params + its own whole objects —
the slice-snapshot primitive); restore replays every artifact whose
entries a rank needs, and the engine's native range compose
reassembles complete objects. Per-rank planned programs land beside
the artifacts as programs/rankN.json: a checkpoint captures plan-time
decisions, not just weights.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

FORMAT = 2


def git_identity(repo: Path) -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                             cwd=repo, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=repo, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:
        return "unknown"


def env_identity() -> dict:
    out = {"python": sys.version.split()[0]}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda or "none"
    except Exception:
        pass
    return out


def launch_record(*, argv, resolved: dict, data: dict, ranks: list,
                  repo: Path, programs: list) -> dict:
    """The block that makes a run re-invocable from its checkpoint:
    the literal argv, the resolved settings, the data identity, code
    + env identity, per-rank host/device, and the relative paths of
    the saved per-rank planned programs."""
    return {
        "argv": list(argv) if argv else [],
        "resolved": resolved,
        "data": data,
        "git": git_identity(repo),
        "env": env_identity(),
        "ranks": ranks,
        "programs": programs,
    }


def write_record(step_dir: Path, *, step: int, seed: int, world: int,
                   data_cursor, losses, save_plan: dict,
                   artifacts: list, launch: dict) -> Path:
    """checkpoint_record.json, written atomically and LAST (the completeness
    marker — a crash mid-snapshot leaves no marker)."""
    record = {
        "format": FORMAT,
        "step": step, "seed": seed, "world": world,
        "created_t": time.time(),
        "data_cursor": data_cursor,
        "losses": list(losses),
        "save_plan": save_plan,
        "artifacts": list(artifacts),
        "launch": launch,
    }
    tmp = step_dir / "checkpoint_record.json.tmp"
    tmp.write_text(json.dumps(record, indent=1))
    out = step_dir / "checkpoint_record.json"
    tmp.rename(out)
    return out


def read_record(step_dir: Path) -> dict:
    mf = Path(step_dir) / "checkpoint_record.json"
    if not mf.is_file():
        raise RuntimeError(f"no checkpoint_record.json at {step_dir}")
    record = json.loads(mf.read_text())
    if record.get("format") != FORMAT:
        raise RuntimeError(
            f"checkpoint at {step_dir} has format "
            f"{record.get('format')!r}; this build reads format "
            f"{FORMAT} only (older checkpoints: use the retired tools "
            f"in tools/train/internal/)")
    return record


def save_programs(step_dir: Path, prog_dicts: list) -> list:
    """programs/rankN.json beside the artifacts; returns the relative
    paths for the launch record."""
    pdir = Path(step_dir) / "programs"
    pdir.mkdir(parents=True, exist_ok=True)
    rel = []
    for i, pd in enumerate(prog_dicts):
        path = pdir / f"rank{i}.json"
        path.write_text(json.dumps(pd))
        rel.append(f"programs/rank{i}.json")
    return rel


def artifacts_for_restore(record: dict, rank: int) -> list:
    """Which artifacts THIS rank must restore, in order: every
    artifact holding a range of an object the rank needs (all of
    them, for parameter reassembly) — its OWN artifact last so its
    whole-object entries (O shards) win any overlap."""
    world = record["world"]
    order = [record["artifacts"][r] for r in range(world) if r != rank]
    order.append(record["artifacts"][rank])
    return order
