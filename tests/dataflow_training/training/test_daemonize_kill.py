"""daemonize --kill gates: the canonical signal->wait->escalate->
VERIFY path. Exit 0 only when the whole daemonized tree is gone; the
pidfile is consumed; a second --kill on the same pidfile is a clean
no-op. Escalation is exercised with a SIGTERM-ignoring child."""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DAEMONIZE = REPO / "tools" / "train" / "daemonize.py"


def launch(tmp, name, *cmd):
    pidfile = tmp / f"{name}.pid"
    logfile = tmp / f"{name}.log"
    out = subprocess.run(
        [sys.executable, str(DAEMONIZE), "--pidfile", str(pidfile),
         "--logfile", str(logfile), "--cwd", str(tmp), "--", *cmd],
        capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    for _ in range(100):
        if pidfile.is_file() and len(pidfile.read_text().split()) == 2:
            break
        time.sleep(0.05)
    return pidfile


def kill(pidfile):
    return subprocess.run(
        [sys.executable, str(DAEMONIZE), "--pidfile", str(pidfile),
         "--kill"], capture_output=True, text=True, timeout=60)


def group_alive(pidfile_text: str) -> bool:
    import os

    pgid = int(pidfile_text.split()[1])
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def test_kill_terminates_and_verifies(tmp_path):
    pidfile = launch(tmp_path, "sleeper", "sleep", "600")
    txt = pidfile.read_text()
    assert group_alive(txt)
    out = kill(pidfile)
    assert out.returncode == 0, out.stderr
    assert not group_alive(txt)
    assert not pidfile.is_file()          # consumed on success
    again = kill(pidfile)
    assert again.returncode == 0          # idempotent no-op
    assert "nothing to do" in again.stdout


def test_kill_escalates_past_sigterm(tmp_path):
    script = tmp_path / "stubborn.py"
    script.write_text(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(600)\n")
    pidfile = launch(tmp_path, "stubborn", sys.executable, str(script))
    time.sleep(0.3)                        # let the handler install
    txt = pidfile.read_text()
    out = kill(pidfile)
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert "killed" in out.stdout          # escalation path taken
    assert not group_alive(txt)
