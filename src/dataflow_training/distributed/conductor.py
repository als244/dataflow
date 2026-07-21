"""The conductor: launch (or attach to) every member daemon, guard
versions and run identity, register per-rank programs, and hand off to
the loop. Zero-config world-1 is one local child daemon through the
same machinery. Split from fleet.py at phase close."""
from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

from dataflow.core.jsonio import program_to_dict
from dataflow_training.lowering.planning import plan_program

from ..run.driver import RunResult, init_model
from ..run.presets import cfg_dict, tokens_per_step
from ..run.recipe import Recipe
from .checkpointing import checkpoint_fleet, distribute_artifacts, resolve_resume
from .grouped_lowering import GroupedBuildVariant, lower_with_group
from .hostops import (
    daemon_paths,
    kill_daemon,
    launch_daemon,
    nsys_command,
    repo_path,
    run_on,
    uds_forward,
    wait_daemon_exit,
)
from .ranks import HostRig, RankState, wait_client
from .hostops import fetch_file
from .sharding import (
    ParallelConfig,
    layer_fields_by_root,
    tp_mlp_shards,
    zero1_halves,
    zero1rs_block_params,
)
from .topology import load_topology
from .loop import fleet_loop

def check_fleet_versions(hosts, log) -> None:
    """Refuse a MIXED-VERSION fleet: every member's repo must sit on
    the same commit as the conductor's. Version skew produces the
    worst failure class there is — structurally mismatched collectives
    that hang (nccl) or silently reduce garbage (hostmem seq-pairing),
    with nothing pointing at the cause (the fp32-partials incident:
    one uncommitted block-side change made a two-box run NaN from
    step 0)."""
    import subprocess

    from .hostops import run_on
    from .topology import repo_root

    local_rev = subprocess.run(
        ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo_root()), "status", "--porcelain",
         "--untracked-files=no"],
        capture_output=True, text=True, check=True).stdout.strip()
    import torch

    local_env = (f"torch {torch.__version__} cuda {torch.version.cuda} "
                 f"cudnn {torch.backends.cudnn.version()}")
    env_probe = ('import torch; print("torch", torch.__version__, '
                 '"cuda", torch.version.cuda, '
                 '"cudnn", torch.backends.cudnn.version())')
    for host in hosts:
        if host.is_local():
            continue
        rev = run_on(host, f"git -C {host.repo} rev-parse HEAD").strip()
        if rev != local_rev:
            raise RuntimeError(
                f"fleet version skew: {host.name} is at {rev[:10]} but "
                f"the conductor is at {local_rev[:10]} — push/pull "
                f"before launching (mixed-version collectives hang or "
                f"corrupt silently)")
        if dirty:
            # local uncommitted edits CANNOT be on the remote — the
            # exact skew that caused the incident
            raise RuntimeError(
                f"fleet version skew: the conductor repo has "
                f"uncommitted tracked changes that {host.name} cannot "
                f"have:\n{dirty}\ncommit+push+pull (or stash) before "
                f"a fleet run")
        env = run_on(host, f"cd {host.repo} && {host.python} -c "
                           f"'{env_probe}'").strip()
        if env != local_env:
            # replicated compute must actually replicate: a torch/cuda
            # version difference changes kernel algorithms — data
            # parallelism is structurally immune (the grad allreduce
            # shares one sum) but tensor parallelism's replicated
            # fields silently train a chimera (the 1B tp incident,
            # round two)
            raise RuntimeError(
                f"fleet ENV skew: {host.name} runs [{env}] but the "
                f"conductor runs [{local_env}] — align the "
                f"environments before a fleet run")
    log(f"[fleet] version handshake ok ({local_rev[:10]}, {local_env})")




def run_fleet_dp(global_cfg, recipe: Recipe, pipeline, steps: int, *,
                 rank_rounds=(6, 2), budgets=None, slabs=None,
                 topology=None, group: str = "dp", attach=None,
                 seed: int = 11, log=print, log_every: int = 10,
                 profile: dict | None = None,
                 backend: str | None = None, opt_shard: str | None = None,
                 tp_mlp: bool = False,
                 execute_padding: bool = False,
                 launch_argv=None,
                 checkpoint_every: int | None = None,
                 checkpoint_dir: str = "results/pretrain/checkpoints",
                 checkpoint_redundancy: int = 1,
                 checkpoint_keep_last: int = 0,
                 run_name: str = "run",
                 resume: str | None = None,
                 prof_dir: str = "results/pretrain/logs") -> RunResult:
    """Train ``global_cfg``'s step batch across the group's hosts;
    returns the conductor's RunResult (losses = GLOBAL step means).

    ``attach`` maps host names to pre-launched daemon sockets (their
    lifecycle stays with the caller); every other member is launched
    here — profiled runs wrap each launched daemon in the canonical
    nsys command and fetch remote reports into ``prof_dir``."""
    topo = topology if topology is not None else load_topology()
    gspec = topo.group(group)
    if backend is not None:
        from .topology import GroupSpec

        gspec = GroupSpec(name=gspec.name, members=gspec.members,
                          backend=backend)
    hosts = topo.group_hosts(group)
    world = len(hosts)
    # DP DEFAULT (responsibility model): zero1rs at world > 1 — each
    # rank steps params/n. opt_shard="co" is the co-responsible
    # DIAGNOSTIC lane (replicated stepping; certified bitwise
    # equality across ranks as a comm-corruption tripwire).
    if opt_shard is None and world > 1 and not tp_mlp:
        opt_shard = "zero1rs"
    if opt_shard == "co":
        opt_shard = None
    if world < 1:
        raise ValueError(f"group {group!r} has no members")
    if world == 1 and opt_shard is not None:
        raise ValueError("opt_shard is meaningless at world 1")
    if world == 1 and tp_mlp:
        raise ValueError("tp_mlp needs at least two ranks")
    if world > 2 and gspec.backend == "hostmem":
        raise ValueError(
            "the hostmem lane is pairwise (world-2 CI); groups with "
            f"{world} members need backend 'nccl' or 'auto'")
    if len(rank_rounds) != world:
        raise ValueError(f"rank_rounds {rank_rounds} vs {world} members")
    r_global = global_cfg.grad_accum_rounds
    if tp_mlp and opt_shard is not None:
        raise ValueError("tp_mlp and opt_shard are separate "
                         "parallelism configurations — pick one")
    if tp_mlp:
        # tensor parallelism splits COMPUTE, not data: every rank runs
        # the full step batch (rank_rounds does not apply)
        cfgs = [global_cfg for _ in range(world)]
        round_map = [tuple(range(r_global)) for _ in range(world)]
    else:
        if sum(rank_rounds) != r_global:
            raise ValueError(f"rank_rounds {rank_rounds} must sum to "
                             f"the global grad_accum_rounds {r_global}")
        cfgs = [replace(global_cfg, grad_accum_rounds=k)
                for k in rank_rounds]
        round_map = []
        start = 0
        for k in rank_rounds:
            round_map.append(tuple(range(start, start + k)))
            start += k
    budgets = tuple(budgets) if budgets else tuple(h.budget_gib
                                                   for h in hosts)
    slabs = tuple(slabs) if slabs else tuple(h.slab_gib for h in hosts)

    parallels = [None] * world
    if tp_mlp:
        plan = tp_mlp_shards(layer_fields_by_root(global_cfg),
                             gspec.name, world)
        plan.validate(getattr(global_cfg, "opt_policy", None))
        plan.consumable("tp")
        parallels = [ParallelConfig(group=gspec.name, rank=i, world=world,
                                    plan=plan) for i in range(world)]
    elif opt_shard == "zero1rs":
        pass    # byte-equal rs/ag: derived at lowering (no ShardPlan)
    elif opt_shard is not None:
        if opt_shard != "zero1":
            raise ValueError(f"opt_shard {opt_shard!r}: 'zero1' "
                             f"(field-snapped) or 'zero1rs' "
                             f"(byte-equal rs/ag)")
        plan = zero1_halves(layer_fields_by_root(global_cfg),
                            gspec.name, world)
        plan.validate(getattr(global_cfg, "opt_policy", None))
        plan.v1_consumable()
        parallels = [ParallelConfig(group=gspec.name, rank=i, world=world,
                                    plan=plan) for i in range(world)]

    ck = None
    if checkpoint_every:
        from dataflow_training.model_families.families import resolve_family

        from .responsibility import responsibility_map

        if world == 1:
            resp = responsibility_map(global_cfg, 1)
        elif opt_shard == "zero1rs":
            fam0 = resolve_family(global_cfg)
            dims0, _ = fam0.family_layouts(global_cfg)
            resp = responsibility_map(
                global_cfg, world, mode="zero1rs",
                shard_params=zero1rs_block_params(
                    layer_fields_by_root(global_cfg), dims0, world))
        else:
            resp = responsibility_map(global_cfg, world, mode="co")
        ck = {"every": int(checkpoint_every),
              "dir": Path(checkpoint_dir) / run_name, "run": run_name,
              "responsibility": resp,
              "argv": launch_argv,
              "resolved": {"preset": getattr(global_cfg, "preset", None),
                           "seed": seed,
                           "opt_shard": opt_shard, "tp_mlp": tp_mlp},
              "data_meta": (pipeline.describe()
                            if hasattr(pipeline, "describe") else {}),
              "redundancy": int(checkpoint_redundancy),
              "keep_last": int(checkpoint_keep_last),
              "hosts_by_name": {h.name: h for h in hosts}}
    fleet_manifest = None
    if resume is not None:
        fleet_manifest = resolve_resume(
            Path(checkpoint_dir) / run_name, resume, log)
        distribute_artifacts(fleet_manifest, hosts, log)
        resolved = fleet_manifest["launch"]["resolved"]
        expect = {"world": (fleet_manifest["world"], world),
                  "seed": (fleet_manifest["seed"], seed),
                  "rank_rounds": (resolved.get("rank_rounds"),
                                  [len(m) for m in round_map]),
                  "backend": (resolved.get("backend"), gspec.backend),
                  "hosts": (resolved.get("hosts"),
                            [h.name for h in hosts])}
        for key, (got, want) in expect.items():
            if got != want:
                raise RuntimeError(
                    f"resume manifest mismatch: {key} was {got!r} at "
                    f"checkpoint time but the run asks {want!r}")

    run_lock = None
    if ck is not None:
        # per-run-name exclusive lock: a second launch of the same run
        # REFUSES instead of silently sharing GPUs and colliding on
        # the checkpoint directory (the double-run incident's other
        # half). Held for the run's duration; the OS drops it on any
        # exit, crash included.
        import fcntl

        ck["dir"].mkdir(parents=True, exist_ok=True)
        lock_path = ck["dir"] / ".run_lock"
        run_lock = open(lock_path, "w")
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            run_lock.close()
            raise RuntimeError(
                f"run {run_name!r} is already active (lock at "
                f"{lock_path}) — a second same-name launch would "
                f"share GPUs and collide on checkpoints; stop the "
                f"other run or pick a different --run-name")
        import os as _os

        run_lock.write(str(_os.getpid()))
        run_lock.flush()

    attach = dict(attach or {})
    rigs = [HostRig(h, slabs[i], budgets[i])
            for i, h in enumerate(hosts)]
    check_fleet_versions(hosts, log)
    try:
        for rig in rigs:
            host = rig.host
            if host.name in attach:
                rig.sock = attach[host.name]
            else:
                wrap = ""
                if profile is not None:
                    if host.is_local():
                        rig.prof_out = str(Path(prof_dir).resolve()
                                           / f"dp_prof_{host.name}")
                    else:
                        rig.prof_out = f"/tmp/dp_prof_{host.name}"
                    wrap = nsys_command(host, rig.prof_out)
                kill_daemon(host)
                rdma_flag = (f"--peer-rdma-device {host.ib_dev}"
                             if host.ib_dev else "")
                paths = launch_daemon(host, slab_gib=rig.slab_gib,
                                      wrap=wrap, extra_flags=rdma_flag)
                rig.launched = True
                if host.is_local():
                    rig.sock = paths["sock"]
                else:
                    local_sock = (f"/tmp/dataflow-fleet-{host.name}-"
                                  f"{os.getpid()}.sock")
                    rig.forward = uds_forward(host, paths["sock"],
                                              local_sock)
                    rig.sock = local_sock
            log_path = daemon_paths(host)["log"]
            rig.client = wait_client(
                rig.sock, name=f"fleet-{host.name}", timeout_s=180,
                fail_hint=(f"{host.name} daemon unreachable; see "
                           f"{log_path} on that host"))
            log(f"[fleet] {host.name} up (slab {rig.slab_gib} GiB)")

        ranks = [RankState(rig.host.name, rig.client, cfgs[i],
                           round_map[i]) for i, rig in enumerate(rigs)]
        coordinator = ranks[0].client
        for other in rigs[1:]:
            coordinator.peer_connect(other.host.name,
                                     other.host.peer_listen)
            deadline = time.time() + 5.0
            peak = {}
            while time.time() < deadline:
                status = coordinator._call(
                    "peer_status", {"peer_id": other.host.name})
                peak = status.get("peak_gbps", {})
                if "rdma" in peak or other.host.ib_dev is None:
                    break
                time.sleep(0.25)
            log(f"[fleet] link {rigs[0].host.name}<->{other.host.name}"
                f" peak Gbit/s: {peak or 'unmeasured'}")
        return fleet_loop(ranks, gspec, recipe, pipeline, steps,
                          budgets=budgets, seed=seed, log=log,
                          log_every=log_every,
                          tokens_step=tokens_per_step(global_cfg),
                          r_global=r_global, profile=profile,
                          parallels=parallels,
                          tp_mode=tp_mlp, checkpoint=ck,
                          fleet_manifest=fleet_manifest,
                          zero1rs_world=(world if opt_shard == "zero1rs"
                                         else None),
                          execute_padding=execute_padding,
                          tp_mlp=tp_mlp)
    finally:
        if run_lock is not None:
            run_lock.close()
        for rig in rigs:
            if rig.client is None:
                continue
            try:
                if rig.launched:
                    rig.client.shutdown()   # daemon exits; a profiler
                else:                       # wrapper then finalizes
                    rig.client.close()
            except Exception:
                pass
        for rig in rigs:
            if not rig.launched:
                continue
            try:
                wait_daemon_exit(rig.host, timeout_s=180.0)
                if rig.prof_out is not None and not rig.host.is_local():
                    dest = str(Path(prof_dir)
                               / f"dp_prof_{rig.host.name}.nsys-rep")
                    if fetch_file(rig.host, rig.prof_out + ".nsys-rep",
                                  dest):
                        log(f"[fleet] fetched {dest}")
                    else:
                        log(f"[fleet] WARNING: no report fetched from "
                            f"{rig.host.name} ({rig.prof_out}.nsys-rep)")
                kill_daemon(rig.host)
            except Exception as e:
                log(f"[fleet] teardown {rig.host.name}: {e}")
            if rig.forward is not None:
                rig.forward.terminate()


