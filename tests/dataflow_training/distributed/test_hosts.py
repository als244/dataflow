"""Distributed host-helper resource ownership.

Tests:
- test_close_uds_forward_reaps_process_and_unlinks_socket: forwarder teardown terminates and reaps its child process and removes the local Unix-socket path.
- test_fleet_surface_status_ignores_unrelated_worktree_edits: version checks
  reject tracked or untracked fleet-code changes without requiring unrelated
  reference-model work to be clean.
"""
import socket
import subprocess
import sys

from dataflow_training.distributed.hosts import close_uds_forward
from dataflow_training.run.conductor import _fleet_surface_status


def test_close_uds_forward_reaps_process_and_unlinks_socket(tmp_path):
    socket_path = tmp_path / "forward.sock"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(socket_path))
    listener.close()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    try:
        close_uds_forward(process, str(socket_path))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.poll() is not None
    assert not socket_path.exists()


def test_fleet_surface_status_ignores_unrelated_worktree_edits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "reference_models").mkdir()
    (repo / "src" / "runtime.py").write_text("VERSION = 1\n")
    (repo / "tests" / "test_runtime.py").write_text("def test_ok(): pass\n")
    (repo / "reference_models" / "llama.py").write_text("MODEL = 1\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'probe'\n")
    (repo / "conftest.py").write_text("")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "baseline"], check=True
    )

    (repo / "reference_models" / "llama.py").write_text("MODEL = 2\n")
    assert _fleet_surface_status(repo) == ""

    (repo / "src" / "runtime.py").write_text("VERSION = 2\n")
    assert "src/runtime.py" in _fleet_surface_status(repo)

    subprocess.run(
        ["git", "-C", str(repo), "restore", "src/runtime.py"], check=True
    )
    (repo / "tests" / "test_new.py").write_text("def test_new(): pass\n")
    assert "tests/test_new.py" in _fleet_surface_status(repo)
