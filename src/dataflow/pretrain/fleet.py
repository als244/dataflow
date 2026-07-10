"""Fleet DP driver (conductor v1): TWO daemons — chicago (this box) +
tubingen over ssh — training ONE model data-parallel with weighted
round distribution and the global-denominator convention.

The conductor: boots/attaches both daemons (tubingen's S1 socket rides
an ssh UNIX-socket forward — client-over-TCP is a follow-up), connects
the peer link over the direct 25 GbE addresses, registers PER-RANK
programs (same model, per-rank grad_accum_rounds = the weighted round
split of the ORIGINAL global config; dp_group baked into optimizer
tasks), performs the WARM-UP + RE-SEED + RE-PUT dance (kernel loads
must precede any parked collective; family_init_all refills token
buffers too — findings, P4a), creates the group, then drives lockstep
steps: each rank gets ITS SLICE of the original stream's rounds, both
runs fire concurrently, per-round losses (each Sum(nll)/GLOBAL_valid)
sum across ranks into the global step mean — directly comparable to
the single-box curves in results/pretrain/.
"""
from __future__ import annotations

import json
import os
import subprocess
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

from .driver import RunResult, daemon_client
from .presets import cfg_dict, tokens_per_step
from .recipe import Recipe

TUB = "tubingen_local"     # LAN alias (192.168.50.31); the bare
                           # "tubingen" alias resolves to the WAN
                           # public IP and hairpins through the router
TUB_PY = "/home/shein/miniconda3/envs/dataflow/bin/python"
TUB_REPO = "/home/shein/Documents/dataflow"
TUB_SOCK = "/tmp/dataflowd-fleet.sock"
TUB_UNIT = "dataflow-fleet"        # transient systemd --user unit
TUB_LOG = "/tmp/dataflowd-fleet.log"
TUB_PROF_OUT = "/tmp/dp_prof_tubingen"
TUB_PEER_ADDR = "192.168.50.32:29700"
CHI_PEER_ADDR = "192.168.50.23:29700"

# Canonical nsys wrapper. capture-range=cudaProfilerApi arms nsys but
# records ONLY the window bracketed by the profiler_control verb
# (SwitchableAnnotator start/stop_capture -> cudaProfilerStart/Stop).
# NOTE nsys 2025.5.2 rejects a 'nccl' trace value — NCCL activity is
# captured through cuda kernels + its NVTX ranges instead.
NSYS_TRACE = "cuda,nvtx,osrt,cublas,cudnn"
CHI_IB_DEV = "mlx5_1"      # enp114s0f1np1 = 192.168.50.23 (25G link)
TUB_IB_DEV = "mlx5_0"      # enp4s0f0np0  = 192.168.50.32 (25G link)


def nsys_command(out_path: str, ib_dev: str, nsys: str = "nsys") -> str:
    return (f"{nsys} profile --trace={NSYS_TRACE} "
            f"--capture-range=cudaProfilerApi --capture-range-end=stop "
            f"--gpu-metrics-devices=0 --ib-net-info-devices={ib_dev} "
            f"-o {out_path} --force-overwrite true")


def ssh(cmd: str, *, timeout: float = 240.0) -> str:
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", TUB, cmd],
                         capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"ssh rc={out.returncode}: {out.stderr[-400:]}")
    return out.stdout


def launch_remote_daemon(slab_gib: float, profile: bool) -> None:
    """Start tubingen's daemon as a TRANSIENT systemd --user unit.

    Never fire-and-forget a raw "cmd &" over ssh here: when cmd is an
    nsys wrapper, nsys's helper daemons keep the ssh session's pipes
    open and the remote bash never releases the session — the client
    ssh then blocks for minutes (the sporadic "ssh hang", findings).
    systemd-run detaches the daemon from the session entirely and
    gives teardown a handle that needs no /proc-scanning pattern kill.
    Requires loginctl enable-linger on the remote user (set up once).
    """
    wrap = nsys_command(TUB_PROF_OUT, TUB_IB_DEV,
                        nsys="/usr/local/bin/nsys") if profile else ""
    wrap = os.environ.get("FLEET_TUB_LAUNCH_PREFIX", wrap)
    ssh(f"rm -f {TUB_LOG}; "
        f"systemd-run --user --collect --unit {TUB_UNIT} "
        f"-p WorkingDirectory={TUB_REPO} "
        f"-p StandardOutput=append:{TUB_LOG} "
        f"-p StandardError=append:{TUB_LOG} "
        f"{wrap} {TUB_PY} -u tools/dataflowd.py start "
        f"--socket {TUB_SOCK} --slab-gib {slab_gib} "
        f"--peer-name tubingen --peer-listen {TUB_PEER_ADDR}",
        timeout=60.0)


def kill_remote_daemon() -> None:
    # systemctl stop = clean SIGTERM to the unit's cgroup; nsys (when
    # profiling) finalizes its report before exiting. NO pkill/fuser:
    # /proc-scanning pattern kills stall for minutes on this box, and
    # the daemon may not match a stable pattern under a wrapper anyway.
    ssh(f"systemctl --user stop {TUB_UNIT} 2>/dev/null; "
        f"systemctl --user reset-failed {TUB_UNIT} 2>/dev/null; "
        f"rm -f {TUB_SOCK}; true", timeout=180.0)


def lower_with_group(cfg, dp_group: str, recompute_levels=None):
    hw = ShapedHardware()
    shaped = build_shaped_program(
        cfg, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(cfg, hw)},
        dp_group=dp_group, recompute_levels=recompute_levels)
    from dataflow.training.lowering import apply_exact_sizes, size_of_factory

    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "llama3-exact",
                             size_of=size_of_factory(dims, fl))


class GroupedBuildVariant:
    """plan_program's recompute rebuilder for dp_group lowerings."""

    def __init__(self, cfg, dp_group: str):
        self.cfg = cfg
        self.dp_group = dp_group

    def __call__(self, levels):
        return lower_with_group(self.cfg, self.dp_group,
                                recompute_levels=levels)


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


def run_fleet_dp(global_cfg, recipe: Recipe, stream, steps: int, *,
                 rank_rounds=(6, 2), budgets=(14.0, 12.0),
                 slabs=(60.0, 30.0), seed: int = 11, log=print,
                 log_every: int = 10, profile: dict | None = None
                 ) -> RunResult:
    """Train ``global_cfg``'s step batch across the pair; returns the
    conductor's RunResult (losses = GLOBAL step means)."""
    r_global = global_cfg.grad_accum_rounds
    assert sum(rank_rounds) == r_global, (rank_rounds, r_global)
    cfgs = [replace(global_cfg, grad_accum_rounds=k) for k in rank_rounds]
    round_map = (tuple(range(rank_rounds[0])),
                 tuple(range(rank_rounds[0], r_global)))

    # ---- ATTACH-ONLY mode: daemons pre-launched externally; the
    # conductor performs ZERO ssh/process management (profiling rigs,
    # and the workaround for the ssh-in-conductor stall — findings)
    attach_tub = os.environ.get("FLEET_ATTACH_TUB_SOCK")
    attach_chi = os.environ.get("FLEET_ATTACH_CHI_SOCK")
    if attach_tub and attach_chi:
        tub_client = EngineClient(attach_tub, client_name="fleet-tub")
        chi_client = EngineClient(attach_chi, client_name="fleet-chi")
        try:
            return fleet_loop(chi_client, tub_client, cfgs, round_map,
                              recipe, stream, steps, budgets=budgets,
                              seed=seed, log=log, log_every=log_every,
                              tokens_step=tokens_per_step(global_cfg),
                              r_global=r_global, profile=profile)
        finally:
            for c in (tub_client, chi_client):
                try:
                    c.close()
                except Exception:
                    pass

    # ---- tubingen daemon + ssh socket forward -------------------------
    kill_remote_daemon()
    launch_remote_daemon(slabs[1], profile=profile is not None)
    local_fwd = f"/tmp/dataflow-fleet-tub-{os.getpid()}.sock"
    fwd = subprocess.Popen(
        ["ssh", "-N", "-o", "BatchMode=yes",
         "-o", "StreamLocalBindUnlink=yes",
         "-L", f"{local_fwd}:{TUB_SOCK}", TUB],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    tub_client = None
    while time.time() < deadline:
        try:
            probe = EngineClient(local_fwd, client_name="probe")
            probe.health()
            probe.close()
            tub_client = EngineClient(local_fwd, client_name="fleet-tub")
            break
        except Exception:
            time.sleep(1.0)
    if tub_client is None:
        fwd.terminate()
        raise RuntimeError("tubingen daemon unreachable; see "
                           "tubingen:/tmp/dataflowd-fleet.log")
    log(f"[fleet] tubingen up (slab {slabs[1]} GiB, forward {local_fwd})")

    with daemon_client(slab_gib=slabs[0], log=log,
                       peer_name="chicago",
                       peer_listen=CHI_PEER_ADDR) as chi_client:
        try:
            return fleet_loop(chi_client, tub_client, cfgs, round_map,
                              recipe, stream, steps, budgets=budgets,
                              seed=seed, log=log, log_every=log_every,
                              tokens_step=tokens_per_step(global_cfg),
                              r_global=r_global, profile=profile)
        finally:
            try:
                tub_client.close()
            except Exception:
                pass
            fwd.terminate()
            kill_remote_daemon()


def fleet_loop(chi_client, tub_client, cfgs, round_map, recipe, stream,
               steps, *, budgets, seed, log, log_every, tokens_step,
               r_global, profile: dict | None = None) -> RunResult:
    ranks = [RankState("chicago", chi_client, cfgs[0], round_map[0]),
             RankState("tubingen", tub_client, cfgs[1], round_map[1])]

    # ---- peer link + per-rank register + warm-up ----------------------
    chi_client.peer_connect("tubingen", TUB_PEER_ADDR)
    for i, rank in enumerate(ranks):
        planned = plan_program(
            lower_with_group(rank.cfg, "dp"),
            fast_memory_capacity=int(budgets[i] * 1024 ** 3),
            recompute=True,
            build_variant=GroupedBuildVariant(rank.cfg, "dp"))
        prog_dict = program_to_dict(planned.program)
        resolver = {"family": "llama3", "cfg": cfg_dict(rank.cfg),
                    "hyper": recipe.hyper_spec()}
        rank.client.materialize_group({"kind": "family_init_all",
                                       "family": "llama3",
                                       "cfg": cfg_dict(rank.cfg),
                                       "seed": seed})
        put_rank_rounds(rank, stream, 0, r_global)
        reg = rank.client.register_program(prog_dict, resolver=resolver)
        missing = reg["bindings"]["missing_inputs"]
        if missing:
            raise RuntimeError(f"{rank.name}: unbound {missing}")
        rank.prog_id = reg["prog_id"]
        log(f"[fleet] {rank.name}: registered {rank.prog_id} "
            f"(rounds {rank.rounds}, budget {budgets[i]} GiB)")
        # WARM-UP (group absent => comm skips): compiles + loads every
        # kernel; a first launch during a parked collective wedges the
        # device. Then RE-SEED and RE-PUT (init refills token buffers).
        warm = rank.client.run(rank.prog_id,
                               args={"step": 0, "valid_rows": tokens_step})
        if warm.get("state") != "done":
            raise RuntimeError(f"{rank.name} warm-up: {warm}")
        rank.client.materialize_group({"kind": "family_init_all",
                                       "family": "llama3",
                                       "cfg": cfg_dict(rank.cfg),
                                       "seed": seed})
        put_rank_rounds(rank, stream, 0, r_global)
        log(f"[fleet] {rank.name}: warm-up done, re-seeded")

    chi_client._call("create_peer_group",
                     {"name": "dp", "members": ["chicago", "tubingen"],
                      "backend": "hostmem"})
    log("[fleet] dp group up (hostmem, world 2)")

    res = RunResult(backend="fleet-dp", budget_gib=budgets[0],
                    meta={"seed": seed, "world": 2,
                          "rank_rounds": [len(r.rounds) for r in ranks],
                          "prog_ids": [r.prog_id for r in ranks],
                          "budgets_gib": list(budgets),
                          "tokens_per_step": tokens_step})

    # ---- lockstep step loop -------------------------------------------
    prof_start = profile.get("start") if profile else None
    prof_stop = profile.get("stop") if profile else None
    peaks = {r.name: {} for r in ranks}
    for step in range(steps):
        if prof_start is not None and step == prof_start:
            for rank in ranks:
                rank.client.profiler_control("start")
            log(f"[fleet] profiler capture STARTED before step {step}")
        t0 = time.perf_counter()
        if step > 0:
            valid = 0
            for rank in ranks:
                valid += put_rank_rounds(rank, stream, step, r_global)
        else:
            valid = tokens_step        # step-0 rounds already resident
        jobs = [StepRun(rank, step, valid) for rank in ranks]
        threads = [threading.Thread(target=j) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for j in jobs:
            if j.error is not None:
                raise RuntimeError(f"step {step} {j.rank.name}: {j.error}")
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
