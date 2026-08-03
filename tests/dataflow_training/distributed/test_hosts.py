"""Distributed host-helper resource ownership.

Tests:
- test_close_uds_forward_reaps_process_and_unlinks_socket: forwarder teardown terminates and reaps its child process and removes the local Unix-socket path.
"""
import socket
import subprocess
import sys

from dataflow_training.distributed.hosts import close_uds_forward


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
