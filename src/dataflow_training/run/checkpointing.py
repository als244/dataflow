"""Fleet checkpointing: the training layer's orchestration over the
checkpoint record — WHEN to save, policy compilation, snapshot
distribution, resume location — with all byte- and record-level work
delegated to ``dataflow.checkpoint``.

Saving compiles the configured source policy (``simple`` by default:
every source saves everything it holds, self-sufficient per-rank
snapshots; ``dedup`` opt-in: one copy of each replicated object as
disjoint responsibility slices) and hands the sources to the
composer, which fans out snapshots, runs the replication-drift
certificate over the streamed hashes, and lands
``checkpoint_record.json`` atomically LAST — the completeness marker.

Loading resolves targets against the record: a rank's own view for
resume, the logical view for evaluation and inspection. Weights-only
loads simply target the parameter objects — optimizer bytes never
enter the store, there is nothing to release afterwards.
"""
from pathlib import Path

from ..data.feed import cursor_to_json
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
    """Make every source snapshot locally available on every resuming
    host. Under the simple policy a same-mapping resume never needs
    this — each rank's snapshot is already on its box — so hosts that
    hold a snapshot skip the push; the function earns its keep on
    REMAPPED resumes and logical loads of another box's shards."""
    import subprocess

    step_dir = Path(record["_step_dir"])
    by_name = {h.name: h for h in hosts}
    source_hosts = [r["host"] for r in record["launch"]["ranks"]]
    for i, snap in enumerate(record["snapshots"]):
        src = step_dir / snap["path"]
        if not src.is_dir():
            # written on a REMOTE rank's box — pull it to the
            # conductor first (the launch names each source host)
            source = by_name.get(source_hosts[i])
            if source is None or source.is_local():
                raise RuntimeError(
                    f"snapshot {snap['path']} missing at {src} and "
                    f"its source {source_hosts[i]!r} is not reachable")
            subprocess.run(
                ["scp", "-q", "-r",
                 f"{source.ssh}:{repo_path(source, str(src))}",
                 str(step_dir)], check=True)
            log(f"[fleet] snapshot {snap['path']} pulled from "
                f"{source.name}")
            if not src.is_dir():
                raise RuntimeError(
                    f"snapshot {snap['path']} unavailable after pull "
                    f"from {source.name}")
        for host in hosts:
            if host.is_local():
                continue
            probe = run_on(host, f"test -d {repo_path(host, str(src))} "
                                 f"&& echo yes || echo no").strip()
            if probe != "yes":
                push_dir(host, str(src), str(step_dir))
                log(f"[fleet] snapshot {snap['path']} -> {host.name}")


def persisted_source_specs(ranks) -> dict:
    """{source_key: [(id, size_bytes), ...]} — each rank's
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
    source policy, hand the sources to the composer, prune old
    steps. The record lands LAST or not at all."""
    import os

    from dataflow.checkpoint import save_checkpoint as compose_checkpoint
    from ..distributed.source_policy import compile_source_policy
    from .checkpoint_record import launch_record, save_programs

    step_dir = ck["dir"] / f"step_{step_next:06d}"
    os.makedirs(step_dir, exist_ok=True)
    policy = ck.get("source_policy") or "simple"
    logical, per_source = compile_source_policy(
        policy=policy, world=len(ranks),
        source_specs=persisted_source_specs(ranks),
        plan=ck["responsibility"], opt_slices=ck.get("opt_slices"))
    # run state (step/losses/cursor) lives in the record's
    # client_payload alone — snapshots carry no client_meta from
    # training (snapshot client_meta remains the bare-engine
    # round-trip feature)
    sources = {}
    for i, rank in enumerate(ranks):
        sources[i] = {"client": rank.client, "path": f"rank{i}",
                      "slices": per_source[i]["slices"],
                      "record": per_source[i]["record"],
                      "objects": per_source[i]["objects"]}
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
        sources, step_dir, step=step_next, seed=meta["seed"],
        logical_objects=logical,
        scheme={"world": len(ranks),
                "responsibility": ck["responsibility"],
                "rank_rounds": meta.get("rank_rounds"),
                "source_policy": policy},
        client_payload={"losses": list(losses_so_far),
                        "data_cursor": cursor_to_json(meta.get("data_cursor")),
                        "seed": meta["seed"]},
        summary=({"last_loss": round(float(losses_so_far[-1]), 4),
                  "steps_recorded": len(losses_so_far)}
                 if losses_so_far else {}),
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


class SoloSource:
    """The world-1 source as the conductor-shaped save path sees it:
    one local client and its program."""

    def __init__(self, client, prog_dict):
        self.name = "local"
        self.client = client
        self.prog_dict = prog_dict


def save_solo_checkpoint(client, prog_dict, ckpt_dir, step_next, *,
                         cfg, seed, argv, losses, data_cursor,
                         data_meta, keep_last, log) -> None:
    """A world-1 run's checkpoint through the SAME policy-compiled
    save path the fleet uses: one source, the world-1 responsibility
    plan (every root whole), a v1 record landing last."""
    from ..distributed.responsibility import responsibility_map
    from ..distributed.topology import HostSpec, repo_root

    host = HostSpec(name="local", peer_listen="127.0.0.1:0",
                    ssh=None, repo=str(repo_root()))
    ck = {"dir": Path(ckpt_dir),
          "responsibility": responsibility_map(cfg, 1),
          "opt_slices": None, "source_policy": "simple",
          "keep_last": int(keep_last), "argv": list(argv),
          "resolved": {"preset": getattr(cfg, "preset", None),
                       "seed": seed, "opt_shard": None,
                       "tp_mlp": False},
          "data_meta": data_meta or {},
          "hosts_by_name": {"local": host}}
    meta = {"seed": seed, "rank_rounds": [cfg.grad_accum_rounds],
            "backend": None, "hosts": ["local"],
            "data_cursor": data_cursor}
    save_checkpoint([SoloSource(client, prog_dict)], ck, step_next,
                    meta, losses, log)


def load_checkpoint(step_dir, *, targets=None, client=None,
                    backing_gib=None, engines=None):
    """Restore a checkpoint into engines and return the record with
    what was booted.

    TARGETS form (``engines=None``): returns ``(record, client)``.
    ``targets`` is anything the record resolver takes: ``"all"`` for
    the logical view (complete objects reassembled from every
    source's slices), a list of ids for a subset — weights-only
    evaluation simply targets the parameter objects, so optimizer
    bytes never enter the store — or ``{source_key: ids-or-Program}``
    for a rank's own view. ``client=None`` boots a scratch in-process
    fake engine sized from the resolved plan itself — the targeted
    objects' bytes plus slack — so any checkpoint the record
    describes loads without a capacity guess (``backing_gib``
    overrides).

    ENGINES form: returns ``(record, {source_key: client})`` with
    every source's FULL rank view restored — the checkpoint stood
    back up as a fleet, no conductor involved. ``engines`` is either
    a caller-supplied ``{source_key: client}`` mapping (each engine
    capability-checked before any restore) or ``"replicate"``, which
    launches one local child daemon per source shaped by the
    record's engine spec (device, fake) and sized from the source's
    resident bytes; the caller owns shutdown of the returned
    clients."""
    from dataflow.checkpoint import read_record, resolve_targets

    step_dir = Path(step_dir)
    record = read_record(step_dir)
    if engines is not None:
        return record, restore_fleet(step_dir, record, engines)
    if targets is None:
        raise ValueError("load_checkpoint needs targets= or engines=")
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


def restore_fleet(step_dir, record, engines) -> dict:
    """Every source's FULL rank view into one engine per source.
    ``engines="replicate"`` boots local child daemons shaped by the
    record's engine spec; a ``{source_key: client}`` mapping uses
    the caller's engines, capability-checked first."""
    from dataflow.checkpoint import (CheckpointError, resolve_targets,
                                     source_resident_bytes)

    sources = sorted({s["source"] for s in record["snapshots"]})
    if engines == "replicate":
        clients = boot_replicas(record, sources)
    else:
        clients = dict(engines)
        missing = [k for k in sources if k not in clients]
        if missing:
            raise CheckpointError(
                f"engine mapping lacks sources {missing}")
        for key in sources:
            needed = source_resident_bytes(record, key)
            capacity = clients[key].query_backing().get(
                "capacity_bytes", 0)
            if capacity < needed:
                raise CheckpointError(
                    f"source {key}: engine backing "
                    f"{capacity / 1024 ** 3:.2f} GiB cannot hold its "
                    f"{needed / 1024 ** 3:.2f} GiB rank state")
    for key in sources:
        ids = sorted({oid for s in record["snapshots"]
                      if s["source"] == key
                      for oid in (s.get("objects") or {})})
        plan = resolve_targets(record, {key: ids})
        for step in plan:
            clients[key].restore_snapshot(
                str(Path(step_dir) / step["path"]),
                remap=step["remap"], overwrite=True)
    return clients


def boot_replicas(record, sources) -> dict:
    """One local child daemon per source: device and fake mode from
    the record's engine spec, backing sized from the source's
    resident bytes. Ports walk up from a fixed base; shutting the
    returned clients down stops the daemons."""
    import time

    from dataflow.checkpoint import source_resident_bytes
    from dataflow.service import EngineClient
    from ..distributed import daemons
    from ..distributed.topology import HostSpec, repo_root

    clients = {}
    for i, key in enumerate(sources):
        spec = (record.get("engine_spec") or {}).get(key) or {}
        needed = source_resident_bytes(record, key)
        backing = max(0.25, 1.25 * needed / 1024 ** 3)
        host = HostSpec(name=f"ckload{key}",
                        peer_listen=f"127.0.0.1:{29901 + i}",
                        ssh=None, repo=str(repo_root()),
                        backing_gib=backing, budget_gib=1.0,
                        device=int(spec.get("device") or 0))
        flags = "--fake" if spec.get("fake") else ""
        p = daemons.launch(host, lane="ckload", backing_gib=backing,
                           extra_flags=flags)
        deadline = time.time() + 60.0
        while True:
            try:
                clients[key] = EngineClient(p["sock"],
                                            client_name="ckload")
                break
            except (ConnectionError, FileNotFoundError, OSError):
                if time.time() > deadline:
                    raise
                time.sleep(0.05)
    return clients
