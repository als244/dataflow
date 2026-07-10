"""Host operations for fleet tooling: run commands on topology hosts
(local shell or one ssh session), launch/tear down daemons portably
(double-fork daemonizer + pidfile process-group kill — no systemd, no
process-name pattern kills), forward remote control sockets, fetch
files back, and build the canonical profiling wrapper.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .topology import HostSpec

# Canonical nsys wrapper. capture-range=cudaProfilerApi arms nsys but
# records ONLY the window bracketed by the profiler_control verb
# (annotator start/stop_capture -> cudaProfilerStart/Stop).
# NOTE nsys 2025.5 rejects a 'nccl' trace value — NCCL activity is
# captured through cuda kernels + its NVTX ranges; re-add 'nccl' when
# the fleet's nsys is upgraded.
NSYS_TRACE = "cuda,nvtx,osrt,cublas,cudnn"


def nsys_command(host: HostSpec, out_path: str) -> str:
    parts = [f"{host.nsys} profile --trace={NSYS_TRACE}",
             "--capture-range=cudaProfilerApi --capture-range-end=stop",
             "--gpu-metrics-devices=0"]
    if host.ib_dev:
        parts.append(f"--ib-net-info-devices={host.ib_dev}")
    parts.append(f"-o {out_path} --force-overwrite true")
    return " ".join(parts)


def run_on(host: HostSpec, cmd: str, *, timeout: float = 120.0) -> str:
    """Run a shell command on the host. Local hosts get a local shell;
    remote hosts one BatchMode ssh session."""
    if host.is_local():
        argv = ["bash", "-c", cmd]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", host.ssh, cmd]
    out = subprocess.run(argv, capture_output=True, text=True,
                         timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"[{host.name}] rc={out.returncode}: "
                           f"{out.stderr[-400:]}")
    return out.stdout


def run_py(host: HostSpec, code: str, *, timeout: float = 120.0) -> str:
    """Run a python snippet on the host with the repo importable."""
    quoted = code.replace("'", "'\"'\"'")
    return run_on(host, f"cd {host.repo} && {host.python} -c '{quoted}'",
                  timeout=timeout)


def daemon_paths(host: HostSpec, lane: str = "fleet") -> dict:
    base = f"/tmp/dataflowd-{lane}"
    return {"sock": f"{base}.sock", "log": f"{base}.log",
            "pid": f"{base}.pid"}


def launch_daemon(host: HostSpec, *, lane: str = "fleet",
                  slab_gib: float, peer_port: int | None = None,
                  extra_flags: str = "", wrap: str = "") -> dict:
    """Start the host's dataflowd detached (tools/daemonize.py). The
    optional ``wrap`` prefix (e.g. nsys_command(...)) runs INSIDE the
    daemonized session, so profiler helpers can never hold the
    launching ssh session open. Returns the daemon's runtime paths."""
    paths = daemon_paths(host, lane)
    inner = (f"{wrap} {host.python} -u {host.repo}/tools/dataflowd.py "
             f"start --socket {paths['sock']} --slab-gib {slab_gib} "
             f"--peer-name {host.name} "
             f"--peer-listen {host.peer_addr(peer_port)} "
             f"{extra_flags}").strip()
    run_on(host,
           f"rm -f {paths['sock']} {paths['log']}; "
           f"{host.python} {host.repo}/tools/daemonize.py "
           f"--pidfile {paths['pid']} --logfile {paths['log']} "
           f"--cwd {host.repo} -- {inner}",
           timeout=60.0)
    return paths


def daemon_alive(host: HostSpec, *, lane: str = "fleet") -> bool:
    paths = daemon_paths(host, lane)
    out = run_on(host,
                 f"if [ -f {paths['pid']} ]; then "
                 f"read pid pgid < {paths['pid']}; "
                 f"kill -0 $pid 2>/dev/null && echo alive; fi; true",
                 timeout=30.0)
    return "alive" in out


def wait_daemon_exit(host: HostSpec, *, lane: str = "fleet",
                     timeout_s: float = 180.0) -> bool:
    """Wait for the daemonized tree to exit ON ITS OWN (e.g. after a
    shutdown verb) — a profiler wrapper finalizes its report in this
    window and must not be signaled."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not daemon_alive(host, lane=lane):
            return True
        time.sleep(2.0)
    return False


def kill_daemon(host: HostSpec, *, lane: str = "fleet",
                grace_s: float = 60.0) -> None:
    """SIGTERM the daemon's PROCESS GROUP (reaches wrapper + helpers),
    grace-wait, SIGKILL leftovers, clean the runtime files. Explicit
    pids only — no /proc-scanning pattern kills."""
    paths = daemon_paths(host, lane)
    steps = max(1, int(grace_s * 2))
    run_on(host,
           f"if [ -f {paths['pid']} ]; then "
           f"read pid pgid < {paths['pid']}; "
           f"if kill -0 $pid 2>/dev/null; then "
           f"kill -TERM -- -$pgid 2>/dev/null; "
           f"for i in $(seq {steps}); do "
           f"kill -0 $pid 2>/dev/null || break; sleep 0.5; done; "
           f"kill -0 $pid 2>/dev/null && kill -KILL -- -$pgid "
           f"2>/dev/null; fi; fi; "
           f"rm -f {paths['pid']} {paths['sock']}; true",
           timeout=grace_s + 60.0)


def uds_forward(host: HostSpec, remote_sock: str, local_sock: str):
    """ssh -N unix-socket forward for a remote daemon's control plane.
    Returns the Popen (terminate() on teardown), or None for local
    hosts (connect to the socket directly)."""
    if host.is_local():
        return None
    return subprocess.Popen(
        ["ssh", "-N", "-o", "BatchMode=yes",
         "-o", "StreamLocalBindUnlink=yes",
         "-L", f"{local_sock}:{remote_sock}", host.ssh],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)


def fetch_file(host: HostSpec, remote_path: str, local_path: str) -> bool:
    """Copy a file from the host to the conductor's filesystem."""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    if host.is_local():
        if Path(remote_path).resolve() == Path(local_path).resolve():
            return True
        out = subprocess.run(["cp", remote_path, local_path],
                             capture_output=True)
        return out.returncode == 0
    out = subprocess.run(["scp", "-q", f"{host.ssh}:{remote_path}",
                          local_path], capture_output=True, text=True,
                         timeout=300)
    return out.returncode == 0
