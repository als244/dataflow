"""Small thread-lifecycle helpers shared by concurrent test drivers."""

import time


def join_threads(threads, timeout: float, *, label: str) -> None:
    """Join a group under one wall-clock deadline and name any stragglers."""
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    alive = [thread.name for thread in threads if thread.is_alive()]
    assert not alive, (f"{label} exceeded its {timeout:g}s thread deadline: "
                       + ", ".join(alive))
