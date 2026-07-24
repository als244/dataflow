"""Shared-server self-heal — the failure contract, end to end over the server.

A run that corrupts the device context (a real out-of-bounds device access) must
NOT take the process down: the engine catches it, its abort-drain hits the
corrupted context, and it marks the engine unrecoverable — the process stays
ALIVE, and health() reports it. The shared engine-process turns that into a
self-heal: the next fresh() sees the unrecoverable process (a wipe alone would
not — the store op still works) and spawns a clean one, so a corrupting test
can't cascade into the rest of the suite. This drives the whole chain with a
real illegal-access task.

Tests:
- test_self_heal_respawns_after_illegal_access: a task doing a real out-of-bounds device access leaves the server alive but health()-unrecoverable (no crash), and SharedServer.fresh() respawns a new, working server.
"""
import contextlib

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from dataflow.core import OutputSpec, Program, TaskSpec  # noqa: E402
from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow_training.testing.server_process import (  # noqa: E402
    SharedServer,
    spawn_server,
)

pytestmark = pytest.mark.gpu

# Registered into the server (spliced after register_all): a resolver whose
# task does a genuine out-of-bounds device write, corrupting the device context.
CRASH_RESOLVER_SRC = """
from dataflow.service.registry import register_program_resolver


class _IllegalAccess:
    def launch(self, ctx):
        import torch
        # a real out-of-bounds device index -> CUDA fault; the run's abort-drain
        # then hits the corrupted context and marks the session unrecoverable.
        buf = torch.zeros(4, device="cuda")
        idx = torch.tensor([1 << 20], device="cuda")
        buf[idx] = 1.0
        torch.cuda.synchronize()


class _CrashResolver:
    def __call__(self, task):
        return _IllegalAccess()


def _build_crash(spec):
    return _CrashResolver()


register_program_resolver("debug_crash", _build_crash)
"""


def crash_program():
    """One task, resolved to the illegal-access executable."""
    return Program(
        name="crash",
        initial_objects=(),
        tasks=(TaskSpec(id="t0", inputs=(),
                        outputs=(OutputSpec(id="b", size_bytes=1024,
                                            location="fast"),),
                        runtime_us=1.0),),
        fast_memory_capacity=1 << 20)


def test_self_heal_respawns_after_illegal_access():
    server = SharedServer(backing_gib=4.0)
    # point the session server at one that KNOWS the crash resolver
    server.handle = spawn_server(backing_gib=4.0,
                                 extra_source=CRASH_RESOLVER_SRC)
    try:
        client = server.handle.client
        dead_pid = server.handle.proc.pid
        assert client.health().get("unrecoverable") is False    # healthy

        # a real out-of-bounds device access corrupts the device context
        reg = client.register_program(program_to_dict(crash_program()),
                                      resolver={"kind": "debug_crash"})
        with contextlib.suppress(Exception):
            client.run(reg["prog_id"], args={})                 # returns FAILED

        # the server SURVIVED (no crash) but its session is now unrecoverable
        assert server.handle.proc.poll() is None
        assert client.health()["unrecoverable"] is True

        # the self-heal: fresh() sees the unrecoverable server and respawns
        healed = server.fresh()
        assert server.handle.proc.pid != dead_pid               # respawned
        assert healed.health().get("unrecoverable") is False    # clean
        healed.put_object("probe", b"\x00\x01\x02\x03")
        assert bytes(healed.get_object("probe")) == b"\x00\x01\x02\x03"
    finally:
        server.close()
