"""Out-of-process dataflowd for workload tests.

Spawn a REAL daemon subprocess, wait for its socket, connect a client, and reap
it on exit. This is the workload-test transport that mimics production most
closely: real process isolation and crash containment, not a Server hosted in a
thread of the test process. Fresh daemon per use (max isolation).
"""
import contextlib
import os
import subprocess
import sys
import time
import uuid

from dataflow_training.run.driver import daemon_client

# The subprocess boots the full service stack and serves until killed. It must
# register the program resolvers itself (fresh interpreter), exactly as the
# in-process daemon_client does.
LAUNCH_SOURCE = """
import sys
from dataflow.service import EngineConfig, Server
from dataflow_training.register import register_all

register_all()
Server(EngineConfig(socket_path=sys.argv[1],
                    slab_backing_gib=float(sys.argv[2]),
                    device=int(sys.argv[3]), fake=False)).serve_forever()
"""


@contextlib.contextmanager
def out_of_process_daemon(*, backing_gib: float = 4.0, device: int = 0,
                          boot_timeout: float = 90.0):
    """Yield a client connected to a freshly-spawned out-of-process daemon;
    terminate the daemon and remove its socket on exit."""
    sock = f"/tmp/dataflow-test-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    proc = subprocess.Popen(
        [sys.executable, "-c", LAUNCH_SOURCE, sock, str(backing_gib),
         str(device)])
    try:
        wait_for_socket(proc, sock, boot_timeout)
        with daemon_client(attach=True, socket=sock) as client:
            yield client
    finally:
        reap(proc)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock)


def wait_for_socket(proc, sock, timeout) -> None:
    from dataflow.service import EngineClient

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"daemon exited during boot (code {proc.returncode})")
        try:
            EngineClient(sock, client_name="probe").close()
            return
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.02)
    raise RuntimeError("daemon did not accept connections in time")


def reap(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
