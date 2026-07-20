"""Launch-and-detach (POSIX double fork): run a command as a daemon
with no ties to the launching session, plus a pidfile handle for
clean teardown. Pure stdlib — portable to any Linux box (no systemd).

    python tools/train/daemonize.py --pidfile P --logfile L [--cwd DIR] -- CMD ARGS...

Prints the daemon pid on stdout and exits once the daemon is exec'd.
The daemon runs in its own session with stdio on /dev/null + the
logfile and every other inherited fd closed, so nothing in its
process tree can keep the launching (ssh) session open — profiler
wrappers spawn helper daemons that do exactly that through a plain
"cmd &" launch, wedging the client ssh for minutes.

Teardown, portable and with no process-name matching:

    read pid pgid < PIDFILE && kill -TERM -- -$pgid

The pidfile's second field is the daemon's process GROUP id (it is
made a group leader via setsid) — signaling the group reaches the
wrapper and every helper it spawned. Send SIGKILL to the group only
after a grace period: a profiler wrapper needs the grace window to
finalize its report after the daemon exits.
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description="run a command detached from this session")
    ap.add_argument("--pidfile", required=True)
    ap.add_argument("--logfile", required=True)
    ap.add_argument("--cwd", default="/")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- CMD ARGS...")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given (usage: ... -- CMD ARGS...)")

    read_fd, write_fd = os.pipe()
    first = os.fork()
    if first > 0:                       # launcher: report pid, exit
        os.close(write_fd)
        reported = os.read(read_fd, 64).decode().strip()
        os.close(read_fd)
        os.waitpid(first, 0)
        if not reported:
            print("daemonize: daemon died before exec", file=sys.stderr)
            return 1
        print(reported)
        return 0

    os.close(read_fd)
    os.setsid()                         # new session + process group
    if os.fork() > 0:                   # session leader exits at once;
        os._exit(0)                     # the daemon can never reacquire
                                        # a controlling terminal

    os.chdir(args.cwd)
    with open(args.pidfile, "w") as fh:
        fh.write(f"{os.getpid()} {os.getpgid(0)}\n")
    devnull = os.open(os.devnull, os.O_RDWR)
    logfd = os.open(args.logfile,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(devnull, 0)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    os.write(write_fd, str(os.getpid()).encode())
    os.closerange(3, 4096)              # write_fd + everything inherited
    os.execvp(cmd[0], cmd)              # exec failure lands in logfile
    return 1                            # unreachable


if __name__ == "__main__":
    sys.exit(main())
