"""Host plumbing: run commands, map paths, move files, forward
sockets — on ANY host, local or ssh, through one code path keyed by
``HostSpec.ssh`` (None = local). The machine half of the topology
vocabulary; daemon lifecycle lives in daemons.py."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .topology import HostSpec


def repo_path(host: HostSpec, path: str) -> str:
    """Resolve a repo-relative artifact path on a host. Daemons run
    with cwd = the repo root, so relative paths in checkpoint records mean
    "under the repo" — but ssh sessions land in $HOME, so every
    remote shell/scp operation must absolute-ify first. (Learned the
    hard way: scp to a relative dest silently ships checkpoints to
    ~/results/... while the daemon restores from <repo>/results/...)"""
    import os

    if os.path.isabs(path):
        return path
    if host.is_local():
        return path
    if not host.repo:
        raise RuntimeError(f"host {host.name}: repo-relative path "
                           f"{path!r} needs host.repo in the topology")
    return f"{host.repo}/{path}"


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
