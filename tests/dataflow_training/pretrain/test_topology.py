"""Topology loader + portable daemonizer: the pieces that keep every
machine-specific fact out of the codebase and daemon lifecycles out of
ssh sessions. Pure-local (no GPU, no network).

Tests:
- test_loader_roundtrip: load_topology parses hosts and groups, resolves the local host's python/repo/device, lists remotes, and exposes group members and backends including a second local GPU entry.
- test_loader_validation: load_topology raises ValueError for a missing peer_listen, a remote missing python, an unknown conductor, and a ghost group member, and load_topology_or_none returns None for an absent file.
- test_daemonize_detach_and_group_kill: daemonize.py returns immediately with the daemon pid, the daemon is a detached process-group leader with stdio redirected, and one signal to the pgid takes down the whole tree.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dataflow_training.distributed.topology import (
    load_topology,
    load_topology_or_none,
    repo_root,
)

GOOD = """
conductor = "alpha"

[hosts.alpha]
peer_listen = "10.0.0.1:29700"
ib_dev = "ibdev0"
backing_gib = 2.0
budget_gib = 1.0

[hosts.beta]
ssh = "beta-lan"
python = "/opt/py/bin/python"
repo = "/srv/dataflow"
peer_listen = "10.0.0.2:29700"

[hosts.alpha_g1]
peer_listen = "10.0.0.1:29710"
device = 1
backing_gib = 2.0
budget_gib = 1.0

[groups.dp]
members = ["alpha", "beta"]
backend = "hostmem"

[groups.node]
members = ["alpha", "alpha_g1"]
backend = "nccl"
"""


def write_topo(tmp_path, text):
    path = tmp_path / "topology.toml"
    path.write_text(text)
    return str(path)


def test_loader_roundtrip(tmp_path):
    topo = load_topology(write_topo(tmp_path, GOOD))
    assert topo.conductor == "alpha"
    local = topo.local()
    assert local.is_local() and local.python == sys.executable
    assert local.repo == str(repo_root())
    assert local.peer_addr(29610) == "10.0.0.1:29610"
    remote = topo.host("beta")
    assert not remote.is_local()
    assert remote.python == "/opt/py/bin/python"
    assert topo.remotes() == [remote]
    group = topo.group("dp")
    assert group.members == ("alpha", "beta")
    assert [h.name for h in topo.group_hosts("dp")] == ["alpha", "beta"]
    # multi-GPU single machine: a second LOCAL entry with its own CUDA
    # device — the world-N pattern (one daemon per GPU, same host)
    assert local.device == 0
    g1 = topo.host("alpha_g1")
    assert g1.is_local() and g1.device == 1
    assert topo.group("node").backend == "nccl"


def test_loader_validation(tmp_path):
    with pytest.raises(ValueError, match="peer_listen"):
        load_topology(write_topo(
            tmp_path, 'conductor = "a"\n[hosts.a]\nib_dev = "x"\n'))
    with pytest.raises(ValueError, match="python"):
        load_topology(write_topo(
            tmp_path,
            'conductor = "a"\n[hosts.a]\npeer_listen = "1:2"\n'
            '[hosts.b]\nssh = "b"\nrepo = "/r"\npeer_listen = "1:3"\n'))
    with pytest.raises(ValueError, match="conductor"):
        load_topology(write_topo(
            tmp_path,
            'conductor = "b"\n[hosts.a]\npeer_listen = "1:2"\n'))
    with pytest.raises(ValueError, match="member"):
        load_topology(write_topo(
            tmp_path,
            'conductor = "a"\n[hosts.a]\npeer_listen = "1:2"\n'
            '[groups.g]\nmembers = ["a", "ghost"]\n'))
    assert load_topology_or_none(str(tmp_path / "absent.toml")) is None


def read_pidfile(pidfile):
    pid_s, pgid_s = Path(pidfile).read_text().split()
    return int(pid_s), int(pgid_s)


@pytest.mark.skipif(not hasattr(os, "killpg"),
                    reason="POSIX process groups required")
def test_daemonize_detach_and_group_kill(tmp_path):
    """Launch a process tree via tools/train/daemonize.py: the launcher must
    return immediately with the daemon pid, the daemon must be a
    process-group leader in its own session with stdio detached, and
    one signal to the pgid must take down the whole tree."""
    pidfile = str(tmp_path / "d.pid")
    logfile = str(tmp_path / "d.log")
    script = str(repo_root() / "tools" / "train" / "daemonize.py")
    t0 = time.monotonic()
    out = subprocess.run(
        [sys.executable, script, "--pidfile", pidfile,
         "--logfile", logfile, "--cwd", str(tmp_path), "--",
         "bash", "-c", "echo started; sleep 300 & sleep 300"],
        capture_output=True, text=True, timeout=15)
    launch_s = time.monotonic() - t0
    assert out.returncode == 0, out.stderr
    assert launch_s < 5.0, f"launcher blocked {launch_s:.1f}s"
    pid, pgid = read_pidfile(pidfile)
    assert int(out.stdout.strip()) == pid
    assert os.getpgid(pid) == pgid        # pidfile's pgid IS its group
    deadline = time.time() + 10
    while time.time() < deadline:
        if "started" in Path(logfile).read_text():
            break
        time.sleep(0.1)
    assert "started" in Path(logfile).read_text()

    os.killpg(pgid, 15)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.2)
        except ProcessLookupError:
            break
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    # the backgrounded sibling died with the group too
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
