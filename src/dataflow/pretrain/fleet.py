"""Fleet DP driver (conductor v1): one daemon per topology-group host
training ONE model data-parallel with weighted round distribution and
the global-denominator convention.

The conductor: boots (or attaches to) every member daemon — remote
control planes ride ssh unix-socket forwards — connects the peer link
over the topology's data-plane addresses, registers PER-RANK programs
(same model, per-rank grad_accum_rounds = the weighted round split of
the ORIGINAL global config; the dp group baked into optimizer tasks),
performs the WARM-UP + RE-SEED + RE-PUT dance (kernel loads must
precede any parked collective; family_init_all refills token buffers
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
from dataflow.training.models.llama3 import family_layouts
from dataflow.training.planning import plan_program
from dataflow.training.shaped_program import (
    ShapedHardware,
    build_shaped_program,
    roofline_block_kind_spec,
)

from .driver import RunResult
from .hostops import (
    daemon_paths,
    fetch_file,
    kill_daemon,
    launch_daemon,
    nsys_command,
    run_on,
    uds_forward,
    wait_daemon_exit,
)
from .presets import cfg_dict, tokens_per_step
from .recipe import Recipe
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

    if resume != "auto":
        mf = Path(resume) / "fleet.json"
        if not mf.is_file():
            raise RuntimeError(f"no fleet.json at {resume}")
        return json.loads(mf.read_text())
    candidates = sorted(run_dir.glob("step_*/fleet.json"))
    if not candidates:
        raise RuntimeError(f"resume=auto found no complete checkpoint "
                           f"under {run_dir}")
    mf = candidates[-1]
    log(f"[fleet] resume=auto -> {mf.parent}")
    return json.loads(mf.read_text())


def save_plan_for(parallels, world: int) -> dict:
    """The SavePlan (Shein's design): who saves what, derived from
    the ownership algebra. Data objects (tokens/targets/losses) are
    never saved — resume re-derives them from the stream. Returns
    {"shared": [id prefixes identical on every rank; rank 0 writes
    them once], "per_rank": [prefixes unique per rank; every rank
    writes its own]}.

      plain DP: shared = W_* + O_* (fully replicated) — only rank 0
        writes anything;
      zero1:    shared = W_* (replicated weights); per-rank = O_*
        (owned shards);
      tp:       per-rank = W_* + O_* (physically sharded objects).
    """
    par = parallels[0] if parallels else None
    if par is None or par.plan is None:
        return {"shared": ["W_", "O_"], "per_rank": []}
    narrowed = any(a.resident != ALL_RANKS for a in par.plan.assignments)
    if narrowed:
        return {"shared": [], "per_rank": ["W_", "O_"]}
    return {"shared": ["W_"], "per_rank": ["O_"]}


def push_dir(host, src_dir: str, dest_dir: str) -> None:
    """Ship a checkpoint artifact directory to a remote host (scp -r;
    local hosts are a no-op — the artifact is already there)."""
    import subprocess

    if host.is_local():
        return
    run_on(host, f"mkdir -p {dest_dir}")
    subprocess.run(["scp", "-q", "-r", src_dir,
                    f"{host.ssh}:{dest_dir}/"], check=True)


def distribute_shared(fleet_manifest: dict, hosts, log) -> None:
    """Make the shared (rank-0-deduped) artifact locally available on
    every resuming host; records per-rank local paths in the
    manifest. Redundant copies made at save time shortcut this."""
    shared = fleet_manifest.get("shared_path")
    if not shared:
        fleet_manifest["local_shared"] = [None] * len(hosts)
        return
    local = []
    for host in hosts:
        if host.is_local():
            local.append(shared)
            continue
        dest_parent = str(Path(shared).parent)
        push_dir(host, shared, dest_parent)
        local.append(shared)   # same path layout on every host
        log(f"[fleet] shared checkpoint artifact -> {host.name}")
    fleet_manifest["local_shared"] = local


def checkpoint_fleet(ranks, ck: dict, step_next: int, meta: dict,
                     losses_so_far: list, log) -> None:
    """Conductor-orchestrated fleet checkpoint at a step boundary:
    every rank snapshots its WHOLE store to a host-LOCAL path (per-
    rank stores already hold exactly their shards, so this is
    correct for plain/zero1/tp alike; the SavePlan dedup layer comes
    separately), then the conductor writes fleet.json LAST as the
    completeness marker."""
    import json
    import os

    step_dir = ck["dir"] / f"step_{step_next:06d}"
    os.makedirs(step_dir, exist_ok=True)   # conductor side (fleet.json)
    plan = ck["save_plan"]
    snaps = []
    rank_paths = []
    shared_path = None
    for i, rank in enumerate(ranks):
        # dest is DAEMON-LOCAL: the snapshot writer mkdirs on its host
        own = [oid for oid in rank.persist_ids
               if any(oid.startswith(p) for p in plan["per_rank"])]
        if i == 0 and plan["shared"]:
            shared = [oid for oid in rank.persist_ids
                      if any(oid.startswith(p) for p in plan["shared"])]
            shared_path = str(step_dir / "shared")
            out = rank.client.snapshot(
                "all", shared_path, ids=shared,
                client_meta={"step": step_next, "shared": True, **meta})
            snaps.append((rank, out["snap_id"]))
        dest = None
        if own:
            dest = str(step_dir / f"rank{i}")
            out = rank.client.snapshot(
                "all", dest, ids=own,
                client_meta={"step": step_next, "rank": i, **meta})
            snaps.append((rank, out["snap_id"]))
        rank_paths.append(dest)
    for rank, snap_id in snaps:
        s = rank.client.wait_snapshot(snap_id, timeout=600.0)
        if s["state"] != "done":
            raise RuntimeError(f"{rank.name} snapshot failed: {s}")
    manifest = {"step": step_next, **meta,
                "save_plan": plan,
                "shared_path": shared_path,
                "rank_paths": rank_paths,
                "losses": list(losses_so_far)}
    with open(step_dir / "fleet.json", "w") as f:
        json.dump(manifest, f)
    log(f"[fleet] checkpoint @ step {step_next} -> {step_dir} "
        f"(shared: {bool(shared_path)}, per-rank: "
        f"{sum(1 for d in rank_paths if d)})")


def lower_with_group(cfg, dp_group: str, recompute_levels=None,
                     dp_overlap: bool = False, parallel=None):
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
    if parallel is not None and parallel.plan is not None:
        plan = parallel.plan
        narrowed = any(a.resident != ALL_RANKS for a in plan.assignments)
        if narrowed:
            plan.consumable("tp")
            rank_view = tp_view(plan, parallel.rank)
            tp_params = {
                root: {"group": parallel.group,
                       "slices": {name: list(sl)
                                  for name, sl in slices.items()}}
                for root, slices in rank_view.items()}
            shard_params = tp_opt_block_params(plan, parallel.rank)
            opt_regions = {root: dict(sh["update"])
                           for root, sh in shard_params.items()}
        else:
            shard_params = shard_block_params(plan, parallel.rank)
            opt_regions = update_regions(plan, parallel.rank)
    shaped = build_shaped_program(
        cfg, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(cfg, hw)},
        dp_group=dp_group, recompute_levels=recompute_levels,
        dp_overlap=dp_overlap, shard_params=shard_params,
        tp_params=tp_params)
    from dataflow.training.lowering import apply_exact_sizes, size_of_factory

    dims, fl = family_layouts(cfg, tp_view=rank_view)
    return apply_exact_sizes(
        shaped, "llama3-exact",
        size_of=size_of_factory(dims, fl, opt_update_regions=opt_regions))


class GroupedBuildVariant:
    """plan_program's recompute rebuilder for dp_group lowerings."""

    def __init__(self, cfg, dp_group: str, dp_overlap: bool = False,
                 parallel=None):
        self.cfg = cfg
        self.dp_group = dp_group
        self.dp_overlap = dp_overlap
        self.parallel = parallel

    def __call__(self, levels):
        return lower_with_group(self.cfg, self.dp_group,
                                recompute_levels=levels,
                                dp_overlap=self.dp_overlap,
                                parallel=self.parallel)


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
    def __init__(self, rank: RankState, step: int, valid: int):
        self.rank = rank
        self.step = step
        self.valid = valid
        self.fetched: dict | None = None
        self.out: dict | None = None
        self.error = None

    def __call__(self):
        try:
            fetch = [f"loss_0_{r}"
                     for r in range(self.rank.cfg.grad_accum_rounds)]
            out = self.rank.client.run(
                self.rank.prog_id,
                args={"step": self.step, "valid_rows": self.valid},
                fetch=fetch)
            if out.get("state") != "done":
                raise RuntimeError(f"{self.rank.name}: {out}")
            self.out = out
            self.fetched = out["fetched"]
        except Exception as e:
            self.error = e


def put_rank_rounds(rank: RankState, stream, step: int,
                    r_global_count: int) -> int:
    """Feed the rank ITS slice of the original stream's rounds for one
    step. Returns valid tokens contributed."""
    valid = 0
    for local_r, orig_r in enumerate(rank.rounds):
        tok, tgt = stream(step * r_global_count + orig_r)
        valid += int((tgt >= 0).sum())
        rank.client.put_object(f"tokens_0_{local_r}",
                               tok.numpy().tobytes())
        rank.client.put_object(f"targets_0_{local_r}",
                               tgt.numpy().tobytes())
    return valid


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


def run_fleet_dp(global_cfg, recipe: Recipe, stream, steps: int, *,
                 rank_rounds=(6, 2), budgets=None, slabs=None,
                 topology=None, group: str = "dp", attach=None,
                 seed: int = 11, log=print, log_every: int = 10,
                 profile: dict | None = None, dp_overlap: bool = False,
                 backend: str | None = None, opt_shard: str | None = None,
                 tp_mlp: bool = False,
                 checkpoint_every: int | None = None,
                 checkpoint_dir: str = "results/pretrain/checkpoints",
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
    if world != 2:
        raise ValueError(f"the hostmem fleet driver is world-2 for "
                         f"now; group {group!r} has {world} members")
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
    elif opt_shard is not None:
        if opt_shard != "zero1":
            raise ValueError(f"opt_shard {opt_shard!r}: only 'zero1' "
                             f"(field-snapped equal halves) exists")
        plan = zero1_halves(layer_fields_by_root(global_cfg),
                            gspec.name, world)
        plan.validate(getattr(global_cfg, "opt_policy", None))
        plan.v1_consumable()
        parallels = [ParallelConfig(group=gspec.name, rank=i, world=world,
                                    plan=plan) for i in range(world)]

    ck = None
    if checkpoint_every:
        ck = {"every": int(checkpoint_every),
              "dir": Path(checkpoint_dir) / run_name, "run": run_name,
              "save_plan": save_plan_for(parallels, world)}
    fleet_manifest = None
    if resume is not None:
        fleet_manifest = resolve_resume(
            Path(checkpoint_dir) / run_name, resume, log)
        distribute_shared(fleet_manifest, hosts, log)
        expect = {"world": world, "rank_rounds": list(rank_rounds),
                  "backend": gspec.backend, "seed": seed,
                  "hosts": [h.name for h in hosts]}
        for key, want in expect.items():
            got = fleet_manifest[key]
            if got != want:
                raise RuntimeError(
                    f"resume manifest mismatch: {key} was {got!r} at "
                    f"checkpoint time but the run asks {want!r}")

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
        return fleet_loop(ranks, gspec, recipe, stream, steps,
                          budgets=budgets, seed=seed, log=log,
                          log_every=log_every,
                          tokens_step=tokens_per_step(global_cfg),
                          r_global=r_global, profile=profile,
                          dp_overlap=dp_overlap, parallels=parallels,
                          tp_mode=tp_mlp, checkpoint=ck,
                          fleet_manifest=fleet_manifest)
    finally:
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


def fleet_loop(ranks, gspec, recipe, stream, steps, *, budgets, seed,
               log, log_every, tokens_step, r_global,
               profile: dict | None = None,
               dp_overlap: bool = False, parallels=None,
               tp_mode: bool = False, checkpoint: dict | None = None,
               fleet_manifest: dict | None = None) -> RunResult:
    world = len(ranks)
    start_step = int(fleet_manifest["step"]) if fleet_manifest else 0

    # ---- per-rank register + warm-up ----------------------------------
    for i, rank in enumerate(ranks):
        par = parallels[i] if parallels else None
        planned = plan_program(
            lower_with_group(rank.cfg, gspec.name, dp_overlap=dp_overlap,
                             parallel=par),
            fast_memory_capacity=int(budgets[i] * 1024 ** 3),
            recompute=True,
            build_variant=GroupedBuildVariant(rank.cfg, gspec.name,
                                              dp_overlap=dp_overlap,
                                              parallel=par))
        prog_dict = program_to_dict(planned.program)
        resolver = {"family": "llama3", "cfg": cfg_dict(rank.cfg),
                    "hyper": recipe.hyper_spec()}
        fill = {"kind": "family_init_all", "family": "llama3",
                "cfg": cfg_dict(rank.cfg), "seed": seed}
        if par is not None and par.plan is not None:
            narrowed = any(a.resident != ALL_RANKS
                           for a in par.plan.assignments)
            prefixes = ("O_", "W_") if narrowed else ("O_",)
            if narrowed:
                view = tp_view(par.plan, par.rank)
                fill["tp_view"] = {
                    root: {f: list(sl) for f, sl in per.items()}
                    for root, per in view.items()}
            # the daemon must allocate THIS RANK's shrunken objects —
            # send the registered program's own sizes
            fill["object_sizes"] = {
                s.id: s.size_bytes
                for s in planned.program.initial_objects
                if s.id.startswith(prefixes)}
            o_bytes = sum(b for oid, b in fill["object_sizes"].items()
                          if oid.startswith("O_"))
            log(f"[fleet] {rank.name}: sharded optimizer state "
                f"{o_bytes / 1024 ** 3:.2f} GiB"
                + (" (tp: sharded weights too)" if narrowed else ""))
        rank.client.materialize_group(fill)
        put_rank_rounds(rank, stream, 0, r_global)
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
            restored_step = None
            shared = fleet_manifest.get("shared_path")
            if shared:
                res = rank.client.restore_snapshot(
                    fleet_manifest["local_shared"][i], overwrite=True)
                restored_step = res["client_meta"]["step"]
            own = fleet_manifest["rank_paths"][i]
            if own:
                res = rank.client.restore_snapshot(own, overwrite=True)
                meta = res["client_meta"]
                if meta["rank"] != i:
                    raise RuntimeError(
                        f"{rank.name}: rank artifact meta {meta} is "
                        f"not rank {i}")
                restored_step = meta["step"]
            if restored_step != start_step:
                raise RuntimeError(
                    f"{rank.name}: restored step {restored_step} != "
                    f"resume step {start_step}")
            put_rank_rounds(rank, stream, start_step, r_global)
            log(f"[fleet] {rank.name}: restored checkpoint @ step "
                f"{start_step}")
        else:
            rank.client.materialize_group(fill)
            put_rank_rounds(rank, stream, 0, r_global)
            log(f"[fleet] {rank.name}: warm-up done, re-seeded")

    ranks[0].client._call("create_peer_group",
                          {"name": gspec.name,
                           "members": list(gspec.members),
                           "backend": gspec.backend})
    log(f"[fleet] {gspec.name} group up ({gspec.backend}, "
        f"world {world})")

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
            valid = 0
            for k, rank in enumerate(ranks):
                v = put_rank_rounds(rank, stream, step, r_global)
                if k == 0 or not tp_mode:
                    valid += v   # tp replicas share ONE batch: count once
        else:
            valid = tokens_step        # start-step rounds already resident
        jobs = [StepRun(rank, step, valid) for rank in ranks]
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
                f"lr {recipe.lr_at(step):.2e}  {tokens_step / dt:.0f} tok/s"
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
                 "backend": gspec.backend, "seed": seed},
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
