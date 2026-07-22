"""Slice snapshots: save only the byte range a saver is RESPONSIBLE
for; restore composes ranges back into complete objects (the
responsibility model's checkpoint primitive). CPU-only (fake engine —
the slice logic is store-level).

Tests:
- test_slice_snapshot_roundtrip_and_compose: two half-range slice snapshots recompose bitwise in place and into a fresh daemon, equal the whole-object snapshot byte-for-byte, and an out-of-bounds range is refused.
- test_second_daemon_on_live_socket_refuses: a second server on a live socket refuses loudly instead of unlinking it, while a stale socket file is reclaimed.
"""
import threading
import time

import numpy as np
import pytest

from dataflow.service import EngineClient, EngineConfig, Server, ServiceError

N = 1 << 20          # 1 MiB object
HALF = N // 2


def boot(tmp, name):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(socket_path=sock, fake=True,
                                 slab_backing_gib=0.2))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(300):
        try:
            with EngineClient(sock, client_name="probe"):
                break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    return server, EngineClient(sock, client_name=name)


def wait_snap(client, out):
    snap_id = out["snap_id"]
    for _ in range(600):
        s = client.snapshot_status(snap_id)
        if s["state"] == "done":
            return
        if s["state"] == "error":
            raise AssertionError(s)
        time.sleep(0.01)
    raise AssertionError("snapshot timeout")


def test_slice_snapshot_roundtrip_and_compose(tmp_path):
    rng = np.random.default_rng(7)
    payload = rng.integers(0, 256, N, dtype=np.uint8).tobytes()

    server, c = boot(tmp_path, "slice-a")
    try:
        c.put_object("W_demo", payload)
        a = tmp_path / "slice_lo"
        b = tmp_path / "slice_hi"
        whole = tmp_path / "whole"
        wait_snap(c, c.snapshot("all", str(a), ids=["W_demo"],
                                ranges={"W_demo": (0, HALF)}))
        wait_snap(c, c.snapshot("all", str(b), ids=["W_demo"],
                                ranges={"W_demo": (HALF, N)}))
        wait_snap(c, c.snapshot("all", str(whole), ids=["W_demo"]))

        # (4) refused ranges
        with pytest.raises(ServiceError):
            c.snapshot("all", str(tmp_path / "bad"), ids=["W_demo"],
                       ranges={"W_demo": (HALF, N + 1)})

        # (1) zero the resident bytes, compose the two slices in place
        c.put_object("W_demo", b"\x00" * N)
        c.restore_snapshot(str(a), overwrite=True)
        c.restore_snapshot(str(b), overwrite=True)
        got = bytes(c.get_object("W_demo"))
        assert got == payload, "in-place slice compose diverged"

        # (2)+(3) fresh daemon: absent object -> create + partial fills
        server2, c2 = boot(tmp_path, "slice-b")
        try:
            c2.restore_snapshot(str(a))
            c2.restore_snapshot(str(b), overwrite=True)
            fresh = bytes(c2.get_object("W_demo"))
            assert fresh == payload, "fresh-daemon slice compose diverged"

            server3, c3 = boot(tmp_path, "slice-c")
            try:
                c3.restore_snapshot(str(whole))
                assert bytes(c3.get_object("W_demo")) == fresh, \
                    "slice compose != whole-object snapshot"
            finally:
                c3.shutdown()
        finally:
            c2.shutdown()
    finally:
        c.shutdown()


def test_second_daemon_on_live_socket_refuses(tmp_path):
    """The double-launch door is closed: a second Server on a LIVE
    socket refuses loudly instead of silently unlinking it (the
    two-runs-on-one-GPU class); a stale socket file is reclaimed."""
    server, c = boot(tmp_path, "solo")
    try:
        sock = str(tmp_path / "solo.sock")
        second = Server(EngineConfig(socket_path=sock, fake=True,
                                     slab_backing_gib=0.05))
        with pytest.raises(RuntimeError, match="already serving"):
            second.serve_forever()
    finally:
        c.shutdown()
    # stale socket (daemon gone): wait until nothing accepts, then a
    # fresh server reclaims the leftover file cleanly
    import socket as _socket
    import time as _t

    sock_path = str(tmp_path / "solo.sock")
    for _ in range(200):
        probe = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect(sock_path)
            probe.close()
            _t.sleep(0.05)
        except OSError:
            probe.close()
            break
    server3, c3 = boot(tmp_path, "solo")
    c3.shutdown()
