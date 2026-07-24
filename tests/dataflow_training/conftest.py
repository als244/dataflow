"""One out-of-process server shared by the whole tests/dataflow_training run.

Workload tests drive the engine only through the client (the client-only
contract), and a real server boot is several seconds — so paying it per test,
or even per family, dominates the suite. Instead ONE server serves the whole
subtree: the ``client`` fixture hands each test a freshly-WIPED server (every
resident object released) so the store looks blank, and if a test left the
server dead or unrecoverable the self-heal respawns it so the failure can't
cascade (SharedServer; the self-heal is covered by test_session_daemon.py).

These fixtures live in THIS conftest, so only tests under tests/dataflow_training
can request the server — other subtrees (tests/dataflow, tests/dataflow_sim)
never spawn it. A test that needs a differently-booted server (e.g.
free-poisoning) spawns its own with ``out_of_process_server``.
"""
import pytest

from dataflow_training.testing.server_process import SharedServer

SERVER_BACKING_GIB = 4.0


@pytest.fixture(scope="session")
def shared_server():
    """The one server shared across this subtree — spawned lazily on first use,
    reaped at the end of the run."""
    server = SharedServer(backing_gib=SERVER_BACKING_GIB)
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def client(shared_server):
    """A clean client for one test: the shared server wiped to a blank
    store (or respawned if the prior test left it unrecoverable)."""
    return shared_server.fresh()
