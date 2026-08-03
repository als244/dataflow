"""Synchronous lifecycle helpers for in-process service tests."""
from __future__ import annotations

import threading


def start_server_thread(server) -> threading.Thread:
    """Start ``server`` and return the thread its owner must later join."""
    thread = threading.Thread(
        target=server.serve_forever,
        name=f"test-server:{server.config.socket_path}",
        daemon=True,
    )
    thread.start()
    return thread


def stop_server_thread(server, thread: threading.Thread, *, timeout: float = 30) -> None:
    """Request shutdown, wait for full server cleanup, and reject a leak."""
    server.state.shutdown_requested.set()
    if server.dispatcher.is_alive():
        server.dispatcher.stop()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), (
        f"in-process server failed to stop: {server.config.socket_path}"
    )


def shutdown_server_thread(
    server,
    thread: threading.Thread,
    client=None,
    *,
    timeout: float = 30,
) -> None:
    """Request protocol shutdown when possible, then synchronously stop."""
    try:
        if client is not None:
            try:
                client.shutdown()
            except (ConnectionError, OSError):
                pass
            finally:
                client.close()
    finally:
        stop_server_thread(server, thread, timeout=timeout)
