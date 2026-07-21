"""Fleet checkpointing: manifest-v2 write/resume orchestration —
responsibility-driven ranged saves, artifact distribution, ordered
restore. Split from the conductor (fleet.py) at phase close; fleet
re-exports these names."""
from pathlib import Path

from ..distributed.hosts import repo_path, run_on

def resolve_resume(run_dir: Path, resume: str, log) -> dict:
    """Locate the resume fleet manifest. ``resume`` is a step
    directory path or "auto" (newest COMPLETE checkpoint wins —
    fleet.json is written LAST by the conductor, so its presence is
    the completeness marker; a crash mid-snapshot leaves no marker
    and auto skips that step)."""
    import json

    from .manifest import read_manifest

    if resume != "auto":
        manifest = read_manifest(Path(resume))
        manifest["_step_dir"] = str(resume)
        return manifest
    candidates = sorted(run_dir.glob("step_*/fleet.json"))
    if not candidates:
        raise RuntimeError(f"resume=auto found no complete checkpoint "
                           f"under {run_dir}")
    mf = candidates[-1]
    log(f"[fleet] resume=auto -> {mf.parent}")
    manifest = read_manifest(mf.parent)
    manifest["_step_dir"] = str(mf.parent)
    return manifest


def push_dir(host, src_dir: str, dest_dir: str) -> None:
    """Ship a checkpoint artifact directory to a remote host (scp -r;
    local hosts are a no-op — the artifact is already there).
    ``dest_dir`` may be repo-relative; it lands under the host's
    repo, mirroring how the daemon resolves it at restore."""
    import subprocess

    if host.is_local():
        return
    dest = repo_path(host, dest_dir)
    run_on(host, f"mkdir -p {dest}")
    subprocess.run(["scp", "-q", "-r", src_dir,
                    f"{host.ssh}:{dest}/"], check=True)


def distribute_artifacts(fleet_manifest: dict, hosts, log) -> None:
    """Make EVERY rank artifact locally available on every resuming
    host (each rank replays all artifacts — parameter ranges compose
    across them). Same path layout on every host; hosts that already
    hold an artifact (its writer, or a same-box peer) skip the push."""
    import subprocess

    step_dir = Path(fleet_manifest["_step_dir"])
    by_name = {h.name: h for h in hosts}
    writers = [r["host"] for r in fleet_manifest["launch"]["ranks"]]
    for i, art in enumerate(fleet_manifest["artifacts"]):
        src = step_dir / art
        if not src.is_dir():
            # written on a REMOTE rank's box — pull it to the
            # conductor first (the manifest records each writer host)
            writer = by_name.get(writers[i])
            if writer is None or writer.is_local():
                raise RuntimeError(
                    f"checkpoint artifact missing at {src} and its "
                    f"writer {writers[i]!r} is not reachable")
            subprocess.run(
                ["scp", "-q", "-r",
                 f"{writer.ssh}:{repo_path(writer, str(src))}",
                 str(step_dir)], check=True)
            log(f"[fleet] artifact {art} pulled from {writer.name}")
            if not src.is_dir():
                raise RuntimeError(
                    f"artifact {art} unavailable after pull from "
                    f"{writer.name}")
        for host in hosts:
            if host.is_local():
                continue
            probe = run_on(host, f"test -d {repo_path(host, str(src))} "
                                 f"&& echo yes || echo no").strip()
            if probe != "yes":
                push_dir(host, str(src), str(step_dir))
                log(f"[fleet] artifact {art} -> {host.name}")


def checkpoint_fleet(ranks, ck: dict, step_next: int, meta: dict,
                     losses_so_far: list, log) -> None:
    """Conductor-orchestrated checkpoint at a step boundary, manifest
    v2: each rank snapshots exactly what it is RESPONSIBLE for (its
    param byte ranges + its own whole objects — rank_save_args over
    the responsibility map), the conductor saves every rank's planned
    program beside the artifacts, and fleet.json (format 2) is
    written LAST as the completeness marker."""
    import os

    from .manifest import launch_record, save_programs, write_manifest
    from ..distributed.responsibility import rank_save_args

    step_dir = ck["dir"] / f"step_{step_next:06d}"
    os.makedirs(step_dir, exist_ok=True)   # conductor side (fleet.json)
    plan = ck["responsibility"]
    snaps = []
    for i, rank in enumerate(ranks):
        persist = set(rank.persist_ids)
        own = sorted(oid for oid in persist if oid.startswith("O_"))
        ids, ranges = rank_save_args(plan, i, own_objects=own)
        ids = [oid for oid in ids if oid in persist]
        ranges = {oid: rng for oid, rng in ranges.items()
                  if oid in persist}
        dest = str(step_dir / f"rank{i}")
        out = rank.client.snapshot(
            "all", dest, ids=ids, ranges=ranges,
            client_meta={"step": step_next, "rank": i, **meta})
        snaps.append((rank, out["snap_id"]))
    for rank, snap_id in snaps:
        s = rank.client.wait_snapshot(snap_id, timeout=600.0)
        if s["state"] != "done":
            raise RuntimeError(f"{rank.name} snapshot failed: {s}")
    progs = save_programs(step_dir,
                          [rank.prog_dict for rank in ranks])
    launch = launch_record(
        argv=ck.get("argv"),
        resolved=dict(ck.get("resolved") or {},
                      world=len(ranks),
                      rank_rounds=meta.get("rank_rounds"),
                      backend=meta.get("backend"),
                      hosts=meta.get("hosts")),
        data=ck.get("data_meta") or {},
        ranks=[{"host": r.name,
                "device": ck["hosts_by_name"][r.name].device}
               for r in ranks],
        repo=Path.cwd(), programs=progs)
    write_manifest(step_dir, step=step_next, seed=meta["seed"],
                   world=len(ranks), data_cursor=meta.get("data_cursor"),
                   losses=losses_so_far, save_plan=plan,
                   artifacts=[f"rank{i}" for i in range(len(ranks))],
                   launch=launch)
    log(f"[fleet] checkpoint @ step {step_next} -> {step_dir} "
        f"(v2, {len(ranks)} artifact(s))")
    keep = ck.get("keep_last", 0)
    if keep > 0:
        import shutil

        complete = sorted(ck["dir"].glob("step_*/fleet.json"))
        for mf in complete[:-keep]:
            old_dir = mf.parent
            shutil.rmtree(old_dir, ignore_errors=True)
            for rank in ranks:
                host = ck["hosts_by_name"][rank.name]
                if not host.is_local():
                    run_on(host, f"rm -rf {repo_path(host, str(old_dir))}")
            log(f"[fleet] pruned checkpoint {old_dir.name}")


