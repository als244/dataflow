"""Launch provenance for checkpoint records: the block that makes a
run re-invocable from its checkpoint — the literal argv, resolved
settings, data identity, code and environment identity, per-rank
host/device, and the saved per-rank planned programs. The record
itself (checkpoint_record.json — schema, slices, validation,
completeness) is ``dataflow.checkpoint``; this module supplies the
opaque ``launch`` payload the training layer stores in it.
"""
import json
import subprocess
import sys
from pathlib import Path


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


def save_programs(step_dir: Path, prog_dicts: list) -> list:
    """programs/rankN.json beside the snapshots; returns the relative
    paths for the launch record."""
    pdir = Path(step_dir) / "programs"
    pdir.mkdir(parents=True, exist_ok=True)
    rel = []
    for i, pd in enumerate(prog_dicts):
        path = pdir / f"rank{i}.json"
        path.write_text(json.dumps(pd))
        rel.append(f"programs/rank{i}.json")
    return rel
