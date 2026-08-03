"""Fleet preflight: say what is wrong with the box before the tests do.

Fleet tests boot real daemons, claim real ports and drive a real fabric, so
they fail for reasons that have nothing to do with the code under test — a
daemon left running by an earlier session still holding the device, a RoCE port
that is down, a topology file that does not describe this machine. Those
failures surface deep inside a fixture as an empty list or a connection refused,
which sends the reader looking for a regression that is not there.

This checks the box once per session and reports what it finds. It does not
clean up: a stray daemon may be someone's running work, and killing it to make
a test pass is not this file's decision to make.
"""
import subprocess
import threading
import time
from pathlib import Path

import pytest


class _ServerThreads:
    """Own every in-process server thread started by one test module."""

    def __init__(self):
        self._servers = []
        self._errors = []

    def start(self, server):
        def run():
            try:
                server.serve_forever()
            except BaseException as exc:  # surface background failures
                self._errors.append((server.config.socket_path, exc))

        thread = threading.Thread(
            target=run,
            name=f"fleet-server:{server.config.socket_path}",
            daemon=True,
        )
        self._servers.append((server, thread))
        thread.start()
        return thread

    def close(self):
        # Client.shutdown() normally requested this already.  Set the same
        # signals here as a backstop when fixture setup or the test body fails
        # before it acquires a client or reaches its own finally block.
        for server, thread in self._servers:
            if not thread.is_alive():
                continue
            server.state.shutdown_requested.set()
            server.dispatcher.stop()

        deadline = time.monotonic() + 30.0
        for _server, thread in self._servers:
            thread.join(max(0.0, deadline - time.monotonic()))

        alive = [thread.name for _server, thread in self._servers
                 if thread.is_alive()]
        if alive:
            pytest.fail("in-process fleet server teardown exceeded 30s: "
                        + ", ".join(alive))
        if self._errors:
            details = "; ".join(f"{path}: {exc!r}"
                                for path, exc in self._errors)
            pytest.fail(f"in-process fleet server thread failed: {details}")


@pytest.fixture(scope="module")
def server_threads():
    """Start and synchronously retire a module's in-process servers."""
    group = _ServerThreads()
    yield group
    group.close()


class _DaemonLanes:
    """Fallback cleanup for daemon lanes explicitly launched by a module."""

    def __init__(self):
        self._lanes = []

    def own(self, host, lane):
        key = (host, lane)
        if key not in self._lanes:
            self._lanes.append(key)

    def close(self):
        from dataflow_training.distributed import daemons

        errors = []
        for host, lane in reversed(self._lanes):
            try:
                daemons.kill(host, lane=lane)
            except Exception as exc:
                errors.append(f"{host.name}:{lane}: {exc!r}")
        if errors:
            pytest.fail("fleet daemon cleanup failed: " + "; ".join(errors))


@pytest.fixture(scope="module")
def daemon_lanes():
    """Retire test-owned daemon lanes even when fixture setup fails."""
    lanes = _DaemonLanes()
    yield lanes
    lanes.close()


class _Forwarders:
    """Terminate and reap SSH UDS-forwarding subprocesses."""

    def __init__(self):
        self._processes = []

    def own(self, process, local_socket):
        self._processes.append((process, Path(local_socket)))
        return process

    def close(self):
        for process, _socket in self._processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 10.0
        alive = []
        for process, socket in self._processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.poll() is None:
                alive.append(str(process.pid))
            socket.unlink(missing_ok=True)
        if alive:
            pytest.fail("SSH forwarder cleanup failed for pid(s): "
                        + ", ".join(alive))


@pytest.fixture(scope="module")
def fleet_forwarders():
    """Own SSH forwarders for both successful and failed setup paths."""
    forwarders = _Forwarders()
    yield forwarders
    forwarders.close()


def stray_daemons() -> list[str]:
    """Dataflow servers already running, which will contend for the device and
    for any fixed port a test claims."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "dataflow.service"],
            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.splitlines()
            if "EngineConfig" in line and "pgrep" not in line]


def roce_ports() -> dict[str, str]:
    """Device -> port state, as the driver reports it."""
    try:
        out = subprocess.run(["ibv_devinfo"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    states, device = {}, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("hca_id:"):
            device = line.split(":", 1)[1].strip()
        elif line.startswith("state:") and device:
            states[device] = line.split(":", 1)[1].strip()
    return states


@pytest.fixture(scope="session", autouse=True)
def fleet_preflight(request):
    """Report the box's condition once, before any fleet test runs."""
    if not request.session.items:
        return
    notes = []
    strays = stray_daemons()
    if strays:
        notes.append(
            f"{len(strays)} dataflow server(s) already running — they hold "
            f"device memory and may hold ports these tests claim; a failure to "
            f"reach RTS or to bind is more likely to be them than a regression")
    states = roce_ports()
    if states:
        notes.append("RoCE ports: " + ", ".join(
            f"{dev} {state}" for dev, state in sorted(states.items())))
        if not any("ACTIVE" in s for s in states.values()):
            notes.append("no ACTIVE port — fabric tests will skip or fail")
    if notes:
        print("\n[fleet preflight] " + "\n[fleet preflight] ".join(notes),
              flush=True)
