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

import pytest


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
