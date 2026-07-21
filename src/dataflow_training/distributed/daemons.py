"""Daemon lifecycle on a host: launch a daemonized dataflowd (with
an optional nsys wrap), probe it, wait it out, kill its whole process
group with verification, and know its runtime paths/env. Module
namespacing does the naming: daemons.launch(host), daemons.kill(host).
Host plumbing (run_on, paths, files) lives in hosts.py."""
from __future__ import annotations

import time

from .hosts import run_on
from .topology import HostSpec

NSYS_TRACE = "cuda,nvtx,osrt,cublas,cudnn"


def nsys_command(host: HostSpec, out_path: str) -> str:
    parts = [f"{host.nsys} profile --trace={NSYS_TRACE}",
             "--capture-range=cudaProfilerApi --capture-range-end=stop",
             "--gpu-metrics-devices=0"]
    if host.ib_dev:
        parts.append(f"--ib-net-info-devices={host.ib_dev}")
    parts.append(f"-o {out_path} --force-overwrite true")
    return " ".join(parts)


def paths(host: HostSpec, lane: str = "fleet") -> dict:
    # keyed by (lane, topology entry): multi-GPU hosts run one daemon
    # per entry, and entry names are unique by construction
    base = f"/tmp/dataflowd-{lane}-{host.name}"
    return {"sock": f"{base}.sock", "log": f"{base}.log",
            "pid": f"{base}.pid"}


# NCCL transport defaults — the N2 bench's winner. NCCL's own
# RoCE/IB path errors on this fabric even with GID_INDEX=3 (WR_FLUSH
# + local access violation; our pyverbs lane works because we control
# every QP knob). Tuned multi-socket transport: 16 Gbit/s on the 25G
# link vs 11.4 untuned.
NCCL_DEFAULT_ENV = {"NCCL_IB_DISABLE": "1",
                    "NCCL_SOCKET_NTHREADS": "4",
                    "NCCL_NSOCKS_PERTHREAD": "4"}


def env(host: HostSpec, extra: dict | None = None) -> str:
    """Env prefix for a daemon launch: NCCL wiring derived from the
    topology (socket iface + HCA) + tuned defaults; ``extra`` (bench
    overrides) wins over everything."""
    merged = dict(NCCL_DEFAULT_ENV)
    if host.iface:
        merged["NCCL_SOCKET_IFNAME"] = host.iface
    if host.ib_dev:
        merged["NCCL_IB_HCA"] = host.ib_dev
    merged.update(extra or {})
    return " ".join(f"{k}={v}" for k, v in merged.items())


def launch(host: HostSpec, *, lane: str = "fleet",
           slab_gib: float, peer_port: int | None = None,
           extra_flags: str = "", wrap: str = "",
           extra_env: dict | None = None) -> dict:
    """Start the host's dataflowd detached (tools/train/daemonize.py). The
    optional ``wrap`` prefix (e.g. nsys_command(...)) runs INSIDE the
    daemonized session, so profiler helpers can never hold the
    launching ssh session open. Returns the daemon's runtime paths."""
    p = paths(host, lane)
    env_str = env(host, extra_env)
    env_prefix = f"env {env_str} " if env_str else ""
    # env(1) rather than shell VAR=... syntax: the daemonizer execs
    # argv directly, so assignments would be taken as the program
    inner = (f"{env_prefix}{wrap} {host.python} -u "
             f"{host.repo}/tools/train/dataflowd.py "
             f"start --socket {p['sock']} --slab-gib {slab_gib} "
             f"--device {host.device} "
             f"--peer-name {host.name} "
             f"--peer-listen {host.peer_addr(peer_port)} "
             f"{extra_flags}").strip()
    run_on(host,
           f"rm -f {p['sock']} {p['log']}; "
           f"{host.python} {host.repo}/tools/train/daemonize.py "
           f"--pidfile {p['pid']} --logfile {p['log']} "
           f"--cwd {host.repo} -- {inner}",
           timeout=60.0)
    return p


def alive(host: HostSpec, *, lane: str = "fleet") -> bool:
    p = paths(host, lane)
    out = run_on(host,
                 f"if [ -f {p['pid']} ]; then "
                 f"read pid pgid < {p['pid']}; "
                 f"kill -0 $pid 2>/dev/null && echo alive; fi; true",
                 timeout=30.0)
    return "alive" in out


def wait_exit(host: HostSpec, *, lane: str = "fleet",
              timeout_s: float = 180.0) -> bool:
    """Wait for the daemonized tree to exit ON ITS OWN (e.g. after a
    shutdown verb) — a profiler wrapper finalizes its report in this
    window and must not be signaled."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not alive(host, lane=lane):
            return True
        time.sleep(2.0)
    return False


def kill(host: HostSpec, *, lane: str = "fleet",
         grace_s: float = 60.0) -> None:
    """SIGTERM the daemon's PROCESS GROUP (reaches wrapper + helpers),
    grace-wait, SIGKILL leftovers, clean the runtime files. Explicit
    pids only — no /proc-scanning pattern kills."""
    p = paths(host, lane)
    steps = max(1, int(grace_s * 2))
    run_on(host,
           f"if [ -f {p['pid']} ]; then "
           f"read pid pgid < {p['pid']}; "
           f"if kill -0 $pid 2>/dev/null; then "
           f"kill -TERM -- -$pgid 2>/dev/null; "
           f"for i in $(seq {steps}); do "
           f"kill -0 $pid 2>/dev/null || break; sleep 0.5; done; "
           f"kill -0 $pid 2>/dev/null && kill -KILL -- -$pgid "
           f"2>/dev/null; fi; fi; "
           f"rm -f {p['pid']} {p['sock']}; true",
           timeout=grace_s + 60.0)
