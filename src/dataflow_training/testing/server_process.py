"""Out-of-process dataflowd for workload tests.

Spawn a REAL server subprocess, wait for its socket, connect a client, and reap
it. This is the workload-test transport that mimics production most closely:
real process isolation and crash containment, not a Server hosted in a thread
of the test process.

Two forms: ``out_of_process_server`` is a context manager (a dedicated server
for one ``with`` block — used where a test needs its own boot config, e.g.
free-poisoning). ``SharedServer`` is the shared server that serves a whole
test run: ``fresh()`` wipes it to a blank store per test, and respawns it if a
test left it dead or unrecoverable (the self-heal). Both build on
``spawn_server`` (explicit spawn + ``ServerProcess.close()`` reap).
"""
import contextlib
import os
import subprocess
import sys
import time
import uuid

# The subprocess boots the full service stack and serves until killed. It must
# register the program resolvers itself (fresh interpreter). ``extra_source``
# (spawn_server) is spliced in after register_all and before serving, so a test
# can register an extra resolver kind (e.g. a crash resolver) without touching
# src.
LAUNCH_PRELUDE = """
import sys
from dataflow.service import EngineConfig, Server
from dataflow_training.register import register_all

register_all()
"""
LAUNCH_SERVE = """
Server(EngineConfig(socket_path=sys.argv[1],
                    slab_backing_gib=float(sys.argv[2]),
                    device=int(sys.argv[3]),
                    poison_on_free=sys.argv[4] == "1",
                    fake=False)).serve_forever()
"""


class ServerProcess:
    """A running out-of-process server plus a connected client. ``close()``
    disconnects the client, terminates the server, and removes its socket. The
    explicit-lifetime form (vs the ``out_of_process_server`` context manager),
    for a caller that manages the server itself — e.g. a SharedServer that
    respawns on demand. ``close()`` tolerates an already-dead server."""

    def __init__(self, proc, sock, client):
        self.proc = proc
        self.sock = sock
        self.client = client

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.client.close()
        reap(self.proc)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.sock)


def spawn_server(*, backing_gib: float = 4.0, device: int = 0,
                 poison_on_free: bool = False, extra_source: str = "",
                 boot_timeout: float = 90.0) -> ServerProcess:
    """Spawn a fresh out-of-process server, wait for its socket, connect a
    client, and return a ServerProcess (call ``.close()`` to reap). On a boot
    failure the server is reaped before the error propagates. ``poison_on_free``
    boots the engine with the 0xFF free-poison debug option on; ``extra_source``
    is spliced into the server's boot after resolvers register and before it
    serves (a test-only resolver hook)."""
    from dataflow.service import EngineClient

    sock = f"/tmp/dataflow-test-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    source = LAUNCH_PRELUDE + extra_source + LAUNCH_SERVE
    proc = subprocess.Popen(
        [sys.executable, "-c", source, sock, str(backing_gib),
         str(device), "1" if poison_on_free else "0"])
    try:
        wait_for_socket(proc, sock, boot_timeout)
        client = EngineClient(sock, client_name="pretrain")
    except BaseException:
        reap(proc)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock)
        raise
    return ServerProcess(proc, sock, client)


@contextlib.contextmanager
def out_of_process_server(*, backing_gib: float = 4.0, device: int = 0,
                          poison_on_free: bool = False,
                          boot_timeout: float = 90.0):
    """Yield a client connected to a freshly-spawned out-of-process server;
    terminate the server and remove its socket on exit. ``poison_on_free``
    boots the engine with the 0xFF free-poison debug option on."""
    handle = spawn_server(backing_gib=backing_gib, device=device,
                          poison_on_free=poison_on_free,
                          boot_timeout=boot_timeout)
    try:
        yield handle.client
    finally:
        handle.close()


class SharedServer:
    """The one shared server for a whole test run.

    ``fresh()`` hands back a CLEAN client for a test: if the server is healthy
    it wipes the store to a blank slate (release every resident object) so the
    test sees a fresh server. If it is UNHEALTHY — health() reports the session
    unrecoverable (a run corrupted the device context), or the process is dead —
    it reaps the old server and spawns a new one, so a corrupting test can't
    cascade into the rest of the suite. (A real illegal-access leaves the
    server ALIVE but dead-for-runs, which a wipe alone would not catch — hence
    the health check.) Spawned lazily on the first ``fresh()``, reaped by
    ``close()``."""

    def __init__(self, *, backing_gib: float = 4.0):
        self.backing_gib = backing_gib
        self.handle: ServerProcess | None = None

    def fresh(self):
        if self.handle is not None:
            try:
                # healthy = the server answers AND no run left its device context
                # corrupted (health.unrecoverable). A dead process makes
                # health() raise; either way we fall through and respawn.
                if not self.handle.client.health().get("unrecoverable", False):
                    client = self.handle.client
                    client.wipe("all")           # release every resident object
                    # ...and drop every registered program so its engine pool is
                    # freed (unregister_program -> close_session). wipe() clears
                    # only the object STORE; the per-program pools are separate,
                    # so without this they accumulate across tests on the shared
                    # server (one placement base each) -> device OOM.
                    for p in client.list_programs():
                        client.unregister_program(p["prog_id"])
                    return client
            except Exception:
                pass
            self.handle.close()          # dead / unrecoverable -> reap the old
        self.handle = spawn_server(backing_gib=self.backing_gib)
        return self.handle.client

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def wait_for_socket(proc, sock, timeout) -> None:
    from dataflow.service import EngineClient

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited during boot (code {proc.returncode})")
        try:
            EngineClient(sock, client_name="probe").close()
            return
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.02)
    raise RuntimeError("server did not accept connections in time")


def reap(proc) -> None:
    """Terminate the server; tolerant of a process that is already dead."""
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
