"""Fleet DP driver (conductor v1): one daemon per topology-group host
training ONE model data-parallel with weighted round distribution and
the global-denominator convention.

The conductor: boots (or attaches to) every member daemon — remote
control planes ride ssh unix-socket forwards — connects the peer link
over the topology's data-plane addresses, registers PER-RANK programs
(same model, per-rank grad_accum_rounds = the weighted round split of
the ORIGINAL global config; the dp group baked into optimizer tasks),
performs the WARM-UP + RE-SEED + RE-PUT dance (kernel loads must
precede any parked collective; the init program refills token buffers
too — findings), creates the group, then drives lockstep steps: each
rank gets ITS SLICE of the original stream's rounds, all runs fire
concurrently, per-round losses (each Sum(nll)/GLOBAL_valid) sum across
ranks into the global step mean — directly comparable to the
single-box curves in results/pretrain/.

All machine facts (hosts, addresses, devices, sizes) come from
topology.toml — see topology.example.toml.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from pathlib import Path

from dataflow.core.jsonio import program_to_dict
from dataflow.service import EngineClient
from dataflow_training.model_families.llama3 import family_layouts
from dataflow_training.lowering.planning import plan_program
from dataflow_training.lowering.shaped_program import (
    ShapedHardware,
    build_shaped_program,
    roofline_block_kind_spec,
)

from ..run.driver import RunResult, init_model
from .hostops import (
    daemon_paths,
    fetch_file,
    kill_daemon,
    launch_daemon,
    nsys_command,
    repo_path,
    run_on,
    uds_forward,
    wait_daemon_exit,
)
from ..run.presets import cfg_dict, tokens_per_step
from ..run.recipe import Recipe
from .sharding import (
    ALL_RANKS,
    ParallelConfig,
    layer_fields_by_root,
    shard_block_params,
    tp_mlp_shards,
    tp_opt_block_params,
    tp_view,
    update_regions,
    zero1_halves,
    zero1rs_block_params,
)
from .topology import load_topology


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
    from .responsibility import rank_save_args

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


def lower_with_group(cfg, dp_group: str, recompute_levels=None,
                     parallel=None,
                     zero1rs_world: int | None = None):
    """``parallel`` (sharding.ParallelConfig with a plan) makes this a
    PER-RANK lowering. An optimizer-consumable plan (zero1): optimizer
    tasks gain shard block_params and the rank's O objects shrink to
    owned slots. A resident-narrowed plan (tensor parallelism):
    fwd/recompute/bwd tasks additionally gain tp block_params, W/dW/
    A/O objects take their sizes from the per-rank sliced layouts,
    and the optimizer runs in replica-grads mode (no reduce; local
    update; owner broadcast)."""
    hw = ShapedHardware()
    shard_params = None
    tp_params = None
    opt_regions = None
    rank_view = None
    opt_slices = None
    if zero1rs_world is not None:
        dims0, fl0 = family_layouts(cfg)
        shard_params = zero1rs_block_params(
            layer_fields_by_root(cfg), dims0, zero1rs_world)
        if not shard_params:
            raise ValueError("zero1rs: no root is byte-equal eligible "
                             "(needs uniform adamw + uniform dtypes "
                             "with param==grad)")
        opt_slices = {root: {"n_slice": sh["n_slice"],
                             "n_tail": sh["n_tail"],
                             "opt_dtype": sh["opt_dtype"]}
                      for root, sh in shard_params.items()}
    elif parallel is not None and parallel.plan is not None:
        plan = parallel.plan
        narrowed = any(a.resident != ALL_RANKS for a in plan.assignments)
        if narrowed:
            plan.consumable("tp")
            rank_view = tp_view(plan, parallel.rank)
            tp_params = {
                root: {name: list(sl) for name, sl in slices.items()}
                for root, slices in rank_view.items()}
            shard_params = tp_opt_block_params(plan, parallel.rank)
            opt_regions = {root: dict(sh["update"])
                           for root, sh in shard_params.items()}
        else:
            shard_params = shard_block_params(plan, parallel.rank)
            opt_regions = update_regions(plan, parallel.rank)
    # Three composable passes (family lowering stays parallelism-blind):
    # fam.lower -> annotate_groups -> exact sizes with the rank view.
    # The equivalence gates (test_group_annotation, digest-pinned) prove
    # this path identical to the retired in-builder grouped lowering.
    from dataflow_training.model_families.families import resolve_family

    from .group_annotation import annotate_groups

    fam = resolve_family(cfg)
    program = fam.lower(cfg, recompute_levels=recompute_levels)
    if dp_group is None:
        # world-1: the solo program IS the rank program — no group
        # handles, no shard/tp (validated upstream)
        if shard_params or tp_params:
            raise ValueError("shard/tp params need a group")
        return program
    program = annotate_groups(program, group=dp_group,
                              shard_params=shard_params,
                              tp_params=tp_params)
    if opt_regions is None and opt_slices is None and rank_view is None:
        return program          # plain DP: solo sizes are the rank sizes
    # sharded/narrowed ranks re-size with the rank view. The layout
    # pieces are llama3's until the family object-sizer hook lands
    # with the responsibility map (plan S4); zero1rs/tp reach fleet
    # only via llama3 today (equivalence-certified).
    from dataflow_training.lowering.emit import (
        apply_exact_sizes,
        narrow_layouts,
        object_size_factory,
    )

    dims, fl = fam.family_layouts(cfg)
    if rank_view:
        fl = narrow_layouts(fl, rank_view)
    return apply_exact_sizes(
        program, f"{fam.name}-exact",
        object_size=object_size_factory(dims, fl, opt_update_regions=opt_regions,
                                opt_slice_by_root=opt_slices))


class GroupedBuildVariant:
    """plan_program's recompute rebuilder for dp_group lowerings."""

    def __init__(self, cfg, dp_group: str,
                 parallel=None, zero1rs_world=None):
        self.cfg = cfg
        self.dp_group = dp_group
        self.parallel = parallel
        self.zero1rs_world = zero1rs_world

    def __call__(self, levels):
        return lower_with_group(self.cfg, self.dp_group,
                                recompute_levels=levels,
                                parallel=self.parallel,
                                zero1rs_world=self.zero1rs_world)


class RankState:
    def __init__(self, name: str, client, cfg, rounds: tuple,
                 prog_id: str | None = None):
        self.name = name
        self.client = client
        self.cfg = cfg
        self.rounds = rounds           # ORIGINAL-stream round indices
        self.prog_id = prog_id
        self.losses: list = []
        self.error: str | None = None
        self.persist_ids: list = []    # W_*/O_* ids from the rank's
                                       # registered program (SavePlan)


class StepRun:
    def __init__(self, rank: RankState, step: int, valid: int,
                 seq_lens: dict | None = None):
        self.rank = rank
        self.step = step
        self.valid = valid
        self.seq_lens = seq_lens
        self.fetched: dict | None = None
        self.out: dict | None = None
        self.error = None

    def __call__(self):
        try:
            fetch = [f"loss_0_{r}"
                     for r in range(self.rank.cfg.grad_accum_rounds)]
            args = {"step": self.step, "valid_rows": self.valid}
            if self.seq_lens is not None:
                args["seq_lens"] = self.seq_lens
            out = self.rank.client.run(
                self.rank.prog_id, args=args, fetch=fetch)
            if out.get("state") != "done":
                raise RuntimeError(f"{self.rank.name}: {out}")
            self.out = out
            self.fetched = out["fetched"]
        except Exception as e:
            self.error = e


def put_rank_rounds(rank: RankState, packed, tokens_per_round: int, *,
                    execute_padding: bool = False,
                    require_full: bool = False) -> dict:
    """Ship the rank ITS slice of the step's packed rounds. Returns
    the rank's wire seq_lens bounds — CONTENT bounds by default (tasks
    compute only content rows); under ``execute_padding`` an under-full
    round's tail rides as one masked segment and executes.
    ``require_full``: raise on an under-full round unless padding
    executes — the TP lane's collectives run planner-sized buffers, so
    content-width rounds cannot cross them yet."""
    lens = {}
    for local_r, orig_r in enumerate(rank.rounds):
        rnd = packed.rounds[orig_r]
        if (require_full and not execute_padding
                and rnd.content < tokens_per_round):
            raise RuntimeError(
                f"round {orig_r} is under-full ({rnd.content}/"
                f"{tokens_per_round}) and this configuration requires "
                f"full rounds (tp_mlp collectives are planner-sized) — "
                f"pass --execute-padding or use exact-fill packing "
                f"(--allow-round-split)")
        bounds = rnd.bounds()
        if execute_padding and rnd.content < tokens_per_round:
            bounds = bounds + [tokens_per_round]
        lens[str(local_r)] = bounds
        rank.client.put_object(f"tokens_0_{local_r}",
                               rnd.tokens.tobytes())
        rank.client.put_object(f"targets_0_{local_r}",
                               rnd.targets.tobytes())
    return lens


def wait_client(sock: str, *, name: str, timeout_s: float,
                fail_hint: str) -> EngineClient:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            probe = EngineClient(sock, client_name="probe")
            probe.health()
            probe.close()
            return EngineClient(sock, client_name=name)
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(fail_hint)


class HostRig:
    """One group member's runtime state under the conductor."""

    def __init__(self, host, slab_gib: float, budget_gib: float):
        self.host = host
        self.slab_gib = slab_gib
        self.budget_gib = budget_gib
        self.launched = False
        self.forward = None
        self.sock: str | None = None
        self.client: EngineClient | None = None
        self.prof_out: str | None = None


def local_topology(*, budget_gib: float = 8.0, slab_gib: float = 8.0,
                   device: int = 0, peer_port: int = 29711) -> "Topology":
    """Zero-config world-1: one localhost member, one group ("local").
    The conductor launches the daemon as a LOCAL CHILD process (the
    HostSpec.ssh=None lane) — the child-daemon pattern at world 1."""
    import os

    from .topology import GroupSpec, HostSpec, Topology

    host = HostSpec(name="local", peer_listen=f"127.0.0.1:{peer_port}",
                    ssh=None, repo=os.getcwd(),
                    slab_gib=slab_gib, budget_gib=budget_gib,
                    device=device)
    return Topology(conductor="local", hosts={"local": host},
                    groups={"local": GroupSpec(name="local",
                                               members=("local",),
                                               backend="hostmem")},
                    source="<local world-1>")


def local_pair_topology(*, budget_gib: float = 4.0,
                        slab_gib: float = 4.0, device: int = 0,
                        ports=(29721, 29722)) -> "Topology":
    """Two localhost members sharing one GPU over the hostmem backend
    — the same-box world-2 pattern the drills and CI ride."""
    import os

    from .topology import GroupSpec, HostSpec, Topology

    hosts = {}
    for i, port in enumerate(ports):
        name = f"local{i}"
        hosts[name] = HostSpec(name=name,
                               peer_listen=f"127.0.0.1:{port}",
                               ssh=None, repo=os.getcwd(),
                               slab_gib=slab_gib, budget_gib=budget_gib,
                               device=device)
    return Topology(conductor="local0", hosts=hosts,
                    groups={"pair": GroupSpec(name="pair",
                                              members=("local0",
                                                       "local1"),
                                              backend="hostmem")},
                    source="<local world-2 pair>")


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


def fleet_loop(ranks, gspec, recipe, pipeline, steps, *, budgets, seed,
               log, log_every, tokens_step, r_global,
               profile: dict | None = None,
               parallels=None,
               tp_mode: bool = False, checkpoint: dict | None = None,
               fleet_manifest: dict | None = None,
               zero1rs_world: int | None = None,
               execute_padding: bool = False,
               tp_mlp: bool = False) -> RunResult:
    world = len(ranks)
    start_step = int(fleet_manifest["step"]) if fleet_manifest else 0

    cursor = fleet_manifest.get("data_cursor") if fleet_manifest else None
    if cursor is not None:
        stepper = pipeline(cursor)
    else:
        stepper = pipeline(None)
        if start_step:
            from dataflow_training.data.pipeline import fast_forward

            log(f"[fleet] checkpoint has no data cursor — fast-"
                f"forwarding the pipeline {start_step} steps (CPU)")
            fast_forward(stepper, start_step)
    tokens_per_round = ranks[0].cfg.max_tokens
    step_packed = stepper.next_step()      # the START step's rounds
    step_lens_by_rank: dict[int, dict] = {}
    last_cursor = step_packed.cursor_after

    # ---- per-rank register + warm-up ----------------------------------
    for i, rank in enumerate(ranks):
        par = parallels[i] if parallels else None
        rank_group = gspec.name if world > 1 else None
        planned = plan_program(
            lower_with_group(rank.cfg, rank_group,
                             parallel=par, zero1rs_world=zero1rs_world),
            fast_memory_capacity=int(budgets[i] * 1024 ** 3),
            recompute=True,
            build_variant=GroupedBuildVariant(rank.cfg, rank_group,
                                              parallel=par,
                                              zero1rs_world=zero1rs_world))
        prog_dict = program_to_dict(planned.program)
        rank.prog_dict = prog_dict          # manifest v2: saved beside
                                            # the checkpoint artifacts
        from dataflow_training.run.presets import resolver_family

        fam_name = resolver_family(rank.cfg)
        resolver = {"kind": "model_family", "family": fam_name,
                    "cfg": cfg_dict(rank.cfg),
                    "hyper": recipe.hyper_spec()}
        init_kwargs = {"seed": seed}
        if zero1rs_world is not None:
            init_kwargs["object_sizes"] = {
                s.id: s.size_bytes
                for s in planned.program.initial_objects
                if s.id.startswith("O_")}
            o_bytes = sum(init_kwargs["object_sizes"].values())
            log(f"[fleet] {rank.name}: byte-equal sharded optimizer "
                f"state {o_bytes / 1024 ** 3:.2f} GiB")
        if par is not None and par.plan is not None:
            narrowed = any(a.resident != ALL_RANKS
                           for a in par.plan.assignments)
            prefixes = ("O_", "W_") if narrowed else ("O_",)
            if narrowed:
                view = tp_view(par.plan, par.rank)
                init_kwargs["tp_view"] = {
                    root: {f: list(sl) for f, sl in per.items()}
                    for root, per in view.items()}
            # the daemon must allocate THIS RANK's shrunken objects —
            # send the registered program's own sizes
            init_kwargs["object_sizes"] = {
                s.id: s.size_bytes
                for s in planned.program.initial_objects
                if s.id.startswith(prefixes)}
            o_bytes = sum(b for oid, b in init_kwargs["object_sizes"].items()
                          if oid.startswith("O_"))
            log(f"[fleet] {rank.name}: sharded optimizer state "
                f"{o_bytes / 1024 ** 3:.2f} GiB"
                + (" (tp: sharded weights too)" if narrowed else ""))
        init_model(rank.client, fam_name, cfg_dict(rank.cfg), **init_kwargs)
        step_lens_by_rank[i] = put_rank_rounds(rank, step_packed,
                                               tokens_per_round,
                                               execute_padding=execute_padding,
                                               require_full=tp_mlp)
        reg = rank.client.register_program(prog_dict, resolver=resolver)
        missing = reg["bindings"]["missing_inputs"]
        if missing:
            raise RuntimeError(f"{rank.name}: unbound {missing}")
        rank.prog_id = reg["prog_id"]
        rank.persist_ids = sorted(
            s.id for s in planned.program.initial_objects
            if s.id.startswith(("W_", "O_")))
        log(f"[fleet] {rank.name}: registered {rank.prog_id} "
            f"(rounds {rank.rounds}, budget {budgets[i]} GiB)")
        # WARM-UP (group absent => comm skips): compiles + loads every
        # kernel; a first launch during a parked collective wedges the
        # device. Then RE-SEED and RE-PUT (init refills token buffers).
        warm = rank.client.run(rank.prog_id,
                               args={"step": 0, "valid_rows": tokens_step})
        if warm.get("state") != "done":
            raise RuntimeError(f"{rank.name} warm-up: {warm}")
        if fleet_manifest is not None:
            # RESUME: restore over the warm-up's mutated state (this
            # ordering makes the kernel warm-up harmless), then feed
            # the START step's rounds. Restores the shared artifact
            # (rank-0-deduped state, distributed to this host by the
            # conductor) plus this rank's own artifact, if any.
            from .manifest import artifacts_for_restore

            restored_step = None
            step_dir = Path(fleet_manifest["_step_dir"])
            for art in artifacts_for_restore(fleet_manifest, i):
                res = rank.client.restore_snapshot(
                    str(step_dir / art), overwrite=True)
                restored_step = res["client_meta"]["step"]
            if restored_step != start_step:
                raise RuntimeError(
                    f"{rank.name}: restored step {restored_step} != "
                    f"resume step {start_step}")
            step_lens_by_rank[i] = put_rank_rounds(rank, step_packed,
                                                   tokens_per_round,
                                                   execute_padding=execute_padding,
                                                   require_full=tp_mlp)
            log(f"[fleet] {rank.name}: restored checkpoint @ step "
                f"{start_step}")
        else:
            init_model(rank.client, fam_name, cfg_dict(rank.cfg),
                       **init_kwargs)
            step_lens_by_rank[i] = put_rank_rounds(rank, step_packed,
                                                   tokens_per_round,
                                                   execute_padding=execute_padding,
                                                   require_full=tp_mlp)
            log(f"[fleet] {rank.name}: warm-up done, re-seeded")

    if world > 1:
        ranks[0].client._call("create_peer_group",
                              {"name": gspec.name,
                               "members": list(gspec.members),
                               "backend": gspec.backend})
        log(f"[fleet] {gspec.name} group up ({gspec.backend}, "
            f"world {world})")
    else:
        log("[fleet] world 1 — no peer group; solo program")

    res = RunResult(backend="fleet-tp" if tp_mode else "fleet-dp",
                    budget_gib=budgets[0],
                    meta={"seed": seed, "world": world,
                          "hosts": [r.name for r in ranks],
                          "rank_rounds": [len(r.rounds) for r in ranks],
                          "prog_ids": [r.prog_id for r in ranks],
                          "budgets_gib": list(budgets),
                          "tokens_per_step": tokens_step})

    if fleet_manifest is not None:
        # continuous curve across the resume: pre-checkpoint losses
        # ride the manifest
        res.losses.extend(fleet_manifest.get("losses", []))
        res.meta["resumed_from_step"] = start_step

    # ---- lockstep step loop -------------------------------------------
    prof_start = profile.get("start") if profile else None
    prof_stop = profile.get("stop") if profile else None
    peaks = {r.name: {} for r in ranks}
    for step in range(start_step, steps):
        if prof_start is not None and step == prof_start:
            for rank in ranks:
                rank.client.profiler_control("start")
            log(f"[fleet] profiler capture STARTED before step {step}")
        t0 = time.perf_counter()
        if step > start_step:
            step_packed = stepper.next_step()
            last_cursor = step_packed.cursor_after
            for k, rank in enumerate(ranks):
                step_lens_by_rank[k] = put_rank_rounds(
                    rank, step_packed, tokens_per_round,
                    execute_padding=execute_padding,
                    require_full=tp_mlp)
        # GLOBAL-DENOMINATOR: every rank normalizes by the step's total
        # (tp replicas share ONE batch — the packed step IS that batch)
        valid = step_packed.valid_rows
        jobs = [StepRun(rank, step, valid, step_lens_by_rank.get(k))
                for k, rank in enumerate(ranks)]
        threads = [threading.Thread(target=j) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for j in jobs:
            if j.error is not None:
                raise RuntimeError(f"step {step} {j.rank.name}: {j.error}")
        if tp_mode:
            # replicas compute the SAME loss (allreduced activations);
            # take rank 0's and demand the others agree — divergence
            # here means the tp group plane broke. NaN checked
            # explicitly: abs(nan - nan) > tol is False, so a NaN run
            # sails through a plain tolerance test (incident lesson)
            per_rank = [sum(j.fetched.values()) for j in jobs]
            step_loss = per_rank[0]
            for k, other in enumerate(per_rank):
                if other != other:
                    raise RuntimeError(
                        f"step {step}: rank {k} loss is NaN")
                if abs(other - step_loss) > 1e-3:
                    raise RuntimeError(
                        f"step {step}: tp replicas disagree — rank 0 "
                        f"{step_loss} vs rank {k} {other}")
        else:
            step_loss = sum(sum(j.fetched.values()) for j in jobs)
        for j in jobs:
            for key in ("peak_fast_bytes", "torch_reserved_peak",
                        "torch_allocated_peak", "placement_extent_bytes"):
                val = j.out.get(key) if j.out else None
                if val is not None:
                    prev = peaks[j.rank.name].get(key, 0)
                    peaks[j.rank.name][key] = max(prev, int(val))
        dt = time.perf_counter() - t0
        res.losses.append(step_loss)
        res.step_wall_s.append(dt)
        res.tok_per_s.append(tokens_step / dt)
        if step % log_every == 0 or step == steps - 1:
            log(f"[fleet] step {step:4d}/{steps}  loss {step_loss:.4f}  "
                f"lr {recipe.lr(step):.2e}  {tokens_step / dt:.0f} tok/s"
                f"  ({dt:.2f}s)")
        if prof_stop is not None and step == prof_stop:
            for rank in ranks:
                rank.client.profiler_control("stop")
            log(f"[fleet] profiler capture STOPPED after step {step} "
                f"(training continues)")
        if (checkpoint is not None
                and (step + 1) % checkpoint["every"] == 0
                and step + 1 < steps):
            checkpoint_fleet(
                ranks, checkpoint, step + 1,
                {"run": checkpoint["run"], "world": world,
                 "hosts": [r.name for r in ranks],
                 "rank_rounds": [len(r.rounds) for r in ranks],
                 "backend": gspec.backend, "seed": seed,
                 "data_cursor": last_cursor},
                res.losses, log)
    # peak DEVICE + HOST accounting, recorded BY DEFAULT per rank
    for rank in ranks:
        try:
            status = rank.client.engine_status()
            pools = status.get("pools_backing") or status.get("store")
            if pools:
                peaks[rank.name]["host_backing"] = pools
        except Exception:
            pass
    res.meta["peaks"] = peaks
    gib = 1024 ** 3
    for name, row in peaks.items():
        fast = row.get("peak_fast_bytes", 0) / gib
        torch_peak = row.get("torch_reserved_peak", 0) / gib
        log(f"[fleet] {name}: peak fast {fast:.2f} GiB, torch reserved "
            f"{torch_peak:.2f} GiB, host {row.get('host_backing')}")
    return res
