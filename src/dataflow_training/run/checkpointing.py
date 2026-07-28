"""Fleet checkpointing: the training layer's orchestration over the
checkpoint record — WHEN to save, policy compilation, snapshot
distribution, resume location — with all byte- and record-level work
delegated to ``dataflow.checkpoint``.

Saving compiles the configured source policy (``simple`` by default:
every writer saves everything it holds, self-sufficient per-rank
snapshots; ``dedup`` opt-in: one copy of each replicated object as
disjoint responsibility slices) and hands the writers to the
composer, which fans out snapshots, runs the replication-drift
certificate over the streamed hashes, and lands
``checkpoint_record.json`` atomically LAST — the completeness marker.

Loading resolves targets against the record: a rank's own view for
resume, the logical view for evaluation and inspection. Weights-only
loads simply target the parameter objects — optimizer bytes never
enter the store, there is nothing to release afterwards.
"""
from pathlib import Path

from ..distributed.hosts import repo_path, run_on


def resolve_resume(run_dir: Path, resume: str, log) -> dict:
    """Locate the resume checkpoint record. ``resume`` is a step
    directory path or "auto" (newest COMPLETE checkpoint wins —
    checkpoint_record.json is written LAST by the composer, so its
    presence is the completeness marker; a crash mid-snapshot leaves
    no marker and auto skips that step)."""
    from dataflow.checkpoint import read_record

    if resume != "auto":
        record = read_record(Path(resume))
        record["_step_dir"] = str(resume)
        return record
    candidates = sorted(run_dir.glob("step_*/checkpoint_record.json"))
    if not candidates:
        raise RuntimeError(f"resume=auto found no complete checkpoint "
                           f"under {run_dir}")
    mf = candidates[-1]
    log(f"[fleet] resume=auto -> {mf.parent}")
    record = read_record(mf.parent)
    record["_step_dir"] = str(mf.parent)
    return record


def push_dir(host, src_dir: str, dest_dir: str) -> None:
    """Ship a snapshot directory to a remote host (scp -r; local
    hosts are a no-op). ``dest_dir`` may be repo-relative; it lands
    under the host's repo, mirroring how the daemon resolves it at
    restore."""
    import subprocess

    if host.is_local():
        return
    dest = repo_path(host, dest_dir)
    run_on(host, f"mkdir -p {dest}")
    subprocess.run(["scp", "-q", "-r", src_dir,
                    f"{host.ssh}:{dest}/"], check=True)


def distribute_artifacts(record: dict, hosts, log) -> None:
    """Make every writer snapshot locally available on every resuming
    host. Under the simple policy a same-mapping resume never needs
    this — each rank's snapshot is already on its box — so hosts that
    hold a snapshot skip the push; the function earns its keep on
    REMAPPED resumes and logical loads of another box's shards."""
    import subprocess

    step_dir = Path(record["_step_dir"])
    by_name = {h.name: h for h in hosts}
    writer_hosts = [r["host"] for r in record["launch"]["ranks"]]
    for i, snap in enumerate(record["snapshots"]):
        src = step_dir / snap["path"]
        if not src.is_dir():
            # written on a REMOTE rank's box — pull it to the
            # conductor first (the launch names each writer host)
            writer = by_name.get(writer_hosts[i])
            if writer is None or writer.is_local():
                raise RuntimeError(
                    f"snapshot {snap['path']} missing at {src} and "
                    f"its writer {writer_hosts[i]!r} is not reachable")
            subprocess.run(
                ["scp", "-q", "-r",
                 f"{writer.ssh}:{repo_path(writer, str(src))}",
                 str(step_dir)], check=True)
            log(f"[fleet] snapshot {snap['path']} pulled from "
                f"{writer.name}")
            if not src.is_dir():
                raise RuntimeError(
                    f"snapshot {snap['path']} unavailable after pull "
                    f"from {writer.name}")
        for host in hosts:
            if host.is_local():
                continue
            probe = run_on(host, f"test -d {repo_path(host, str(src))} "
                                 f"&& echo yes || echo no").strip()
            if probe != "yes":
                push_dir(host, str(src), str(step_dir))
                log(f"[fleet] snapshot {snap['path']} -> {host.name}")


def persisted_writer_specs(ranks) -> dict:
    """{writer_index: [(id, size_bytes), ...]} — each rank's
    persistent objects at its resident sizes, read off the marker in
    its serialized program. An object may appear under several
    locations (placement pre-copies); it is ONE object, listed
    once."""
    specs = {}
    for i, rank in enumerate(ranks):
        by_id = {}
        for o in rank.prog_dict["initial_objects"]:
            if o.get("persistent"):
                by_id[o["id"]] = int(o["size_bytes"])
        specs[i] = sorted(by_id.items())
    return specs


def save_checkpoint(ranks, ck: dict, step_next: int, meta: dict,
                    losses_so_far: list, log) -> None:
    """Conductor-driven checkpoint at a step boundary: compile the
    source policy, hand the writers to the composer, prune old
    steps. The record lands LAST or not at all."""
    import os

    from dataflow.checkpoint import save_checkpoint as compose_checkpoint
    from ..distributed.source_policy import compile_source_policy
    from .checkpoint_record import launch_record, save_programs

    step_dir = ck["dir"] / f"step_{step_next:06d}"
    os.makedirs(step_dir, exist_ok=True)
    policy = ck.get("source_policy") or "simple"
    logical, per_writer = compile_source_policy(
        policy=policy, world=len(ranks),
        writer_specs=persisted_writer_specs(ranks),
        plan=ck["responsibility"], opt_slices=ck.get("opt_slices"))
    writers = {}
    for i, rank in enumerate(ranks):
        writers[i] = {"client": rank.client, "path": f"rank{i}",
                      "slices": per_writer[i]["slices"],
                      "record": per_writer[i]["record"],
                      "objects": per_writer[i]["objects"],
                      "client_meta": {"step": step_next, "rank": i,
                                      **meta}}
    progs = save_programs(step_dir, [r.prog_dict for r in ranks])
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
    compose_checkpoint(
        writers, step_dir, step=step_next, seed=meta["seed"],
        logical_objects=logical,
        scheme={"world": len(ranks),
                "responsibility": ck["responsibility"],
                "rank_rounds": meta.get("rank_rounds"),
                "source_policy": policy},
        client_payload={"losses": list(losses_so_far),
                        "data_cursor": meta.get("data_cursor"),
                        "seed": meta["seed"]},
        launch=launch)
    log(f"[fleet] checkpoint @ step {step_next} -> {step_dir} "
        f"({policy}, {len(ranks)} snapshot(s))")
    keep = ck.get("keep_last", 0)
    if keep > 0:
        import shutil

        complete = sorted(ck["dir"].glob("step_*/checkpoint_record.json"))
        for mf in complete[:-keep]:
            old_dir = mf.parent
            shutil.rmtree(old_dir, ignore_errors=True)
            for rank in ranks:
                host = ck["hosts_by_name"][rank.name]
                if not host.is_local():
                    run_on(host, f"rm -rf {repo_path(host, str(old_dir))}")
            log(f"[fleet] pruned checkpoint {old_dir.name}")


def load_checkpoint(step_dir, *, targets, client=None,
                    backing_gib=None):
    """Restore a checkpoint's TARGETS into an engine and return
    ``(record, client)``.

    ``targets`` is anything the record resolver takes: ``"all"`` for
    the logical view (complete objects reassembled from every
    writer's slices), a list of ids for a subset — weights-only
    evaluation simply targets the parameter objects, so optimizer
    bytes never enter the store — or ``{writer_key: ids-or-Program}``
    for a rank's own view. ``client=None`` boots a scratch in-process
    fake engine sized from the resolved plan itself — the targeted
    objects' bytes plus slack — so any checkpoint the record
    describes loads without a capacity guess (``backing_gib``
    overrides)."""
    from dataflow.checkpoint import read_record, resolve_targets

    step_dir = Path(step_dir)
    record = read_record(step_dir)
    plan = resolve_targets(record, targets)
    sizes = {}
    for step in plan:
        for windows in step["remap"].values():
            for w in windows:
                sizes[w["id"]] = int(w["bytes"])
    needed = sum(sizes.values())
    if client is not None:
        # capability, never placement: the engine must have room
        capacity = client.query_backing().get("capacity_bytes", 0)
        if capacity < needed:
            from dataflow.checkpoint import CheckpointError

            raise CheckpointError(
                f"engine backing {capacity / 1024 ** 3:.2f} GiB "
                f"cannot hold the {needed / 1024 ** 3:.2f} GiB the "
                f"targets restore")
    if client is None:
        import tempfile
        import threading
        import time

        from dataflow.service import EngineClient, EngineConfig, Server

        if backing_gib is None:
            backing_gib = max(0.25, 1.25 * needed / 1024 ** 3)
        sock = str(Path(tempfile.mkdtemp()) / "ckload.sock")
        server = Server(EngineConfig(socket_path=sock, fake=True,
                                     slab_backing_gib=backing_gib))
        threading.Thread(target=server.serve_forever,
                         daemon=True).start()
        for _ in range(600):
            try:
                EngineClient(sock, client_name="probe").close()
                break
            except OSError:
                time.sleep(0.01)
        client = EngineClient(sock, client_name="ckload")
    for step in plan:
        client.restore_snapshot(str(step_dir / step["path"]),
                                remap=step["remap"], overwrite=True)
    return record, client
