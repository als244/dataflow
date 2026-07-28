"""Slice snapshots: a snapshot saves SLICES — byte ranges of stored
objects mapped into logical objects — and restore places them by
logical coordinates after verifying per-slice hashes. CPU-only (fake
engine — the slice logic is store-level).

Tests:
- test_slice_roundtrip_and_compose: two half-src slices recompose bitwise in place and into a fresh daemon, equal the bulk whole-store snapshot byte-for-byte.
- test_shifted_dst_recompose: slices with dst != src land at logical coordinates — two shard objects compose one logical object on a fresh daemon, and a src range relocates into a differently-named smaller logical; stored names are not recreated.
- test_remap_extraction_restore: a remap plan extracts logical ranges into local objects — a full split into two shards and a partial mid-range window; targets outside the windows are not created.
- test_corrupt_payload_refused_store_untouched: a flipped payload byte refuses with VERIFY_FAILED before any placement; verify=False restores the corrupt bytes (the documented opt-out).
- test_refusal_mid_list_leaves_store_untouched: a size-mismatched resident target refuses the whole restore during validation — earlier valid entries are not placed.
- test_snapshot_json_schema_and_hashes: snapshot.json carries the dataflow-snapshot/v1 schema and fully materialized slice entries whose hashes match recomputed blake2b-16; a tampered schema string refuses with VERSION_SKEW; a dir without snapshot.json refuses with IO_ERROR.
- test_slice_validation_refusals: out-of-bounds src, dst/src length mismatch, dst beyond logical_bytes, conflicting logical_bytes, unknown id, and an empty slice list each refuse with BAD_REQUEST.
- test_duplicate_snapshots_full_and_independent: a duplicated object stores its own full payload (no reference segments, no lineage or version keys, no dedup count) and both objects round-trip independently on a fresh daemon.
- test_snapshot_has_no_group_concept: snapshot.json carries no group table, restore reports none, and the group verbs are gone from the client surface.
- test_restore_runs_in_background_and_parks_writers: a non-blocking restore returns a restore id while its payload work runs off the dispatcher; a concurrent write to a target parks until the restore's leases release, then lands.
- test_restore_status_lifecycle: restore_status tracks a restore to done with its restored list and client_meta; an unknown id refuses with UNKNOWN_RESTORE.
- test_second_daemon_on_live_socket_refuses: a second server on a live socket refuses loudly instead of unlinking it, while a stale socket file is reclaimed.
"""
import hashlib
import json
import threading
import time

import numpy as np
import pytest

from dataflow.service import EngineClient, EngineConfig, Server, ServiceError

N = 1 << 20          # 1 MiB object
HALF = N // 2
SMALL = 4096


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


def rng_bytes(seed, n):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, n, dtype=np.uint8).tobytes()


def snapshot_json(dest):
    return json.loads((dest / "snapshot.json").read_text())


def test_slice_roundtrip_and_compose(tmp_path):
    payload = rng_bytes(7, N)
    server, c = boot(tmp_path, "slice-a")
    try:
        c.put_object("W_demo", payload)
        a = tmp_path / "slice_lo"
        b = tmp_path / "slice_hi"
        whole = tmp_path / "whole"
        wait_snap(c, c.snapshot(str(a),
                                slices=[{"id": "W_demo",
                                         "src": [0, HALF]}]))
        wait_snap(c, c.snapshot(str(b),
                                slices=[{"id": "W_demo",
                                         "src": [HALF, N]}]))
        wait_snap(c, c.snapshot(str(whole)))     # bulk: whole store

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
                    "slice compose != bulk whole-store snapshot"
            finally:
                c3.shutdown()
        finally:
            c2.shutdown()
    finally:
        c.shutdown()


def test_shifted_dst_recompose(tmp_path):
    shard0 = rng_bytes(11, SMALL)
    shard1 = rng_bytes(13, SMALL)
    stored = rng_bytes(17, 2 * SMALL)
    server, c = boot(tmp_path, "shift-a")
    try:
        c.put_object("O_r0", shard0)
        c.put_object("O_r1", shard1)
        c.put_object("W_src", stored)
        s0 = tmp_path / "shard0"
        s1 = tmp_path / "shard1"
        sh = tmp_path / "shifted"
        wait_snap(c, c.snapshot(str(s0), slices=[
            {"id": "O_r0", "logical_id": "O",
             "dst": [0, SMALL], "logical_bytes": 2 * SMALL}]))
        wait_snap(c, c.snapshot(str(s1), slices=[
            {"id": "O_r1", "logical_id": "O",
             "dst": [SMALL, 2 * SMALL], "logical_bytes": 2 * SMALL}]))
        # src != dst within one stored object, into a SMALLER logical
        wait_snap(c, c.snapshot(str(sh), slices=[
            {"id": "W_src", "src": [SMALL, 2 * SMALL],
             "logical_id": "W_shift", "dst": [0, SMALL],
             "logical_bytes": SMALL}]))
    finally:
        c.shutdown()

    server2, c2 = boot(tmp_path, "shift-b")
    try:
        c2.restore_snapshot(str(s0))               # creates logical O
        c2.restore_snapshot(str(s1), overwrite=True)
        assert bytes(c2.get_object("O")) == shard0 + shard1, \
            "shifted-dst shards did not compose the logical object"
        assert c2.query_object("O_r0") is None, \
            "mapped restore must not recreate the stored name"

        c2.restore_snapshot(str(sh))
        assert bytes(c2.get_object("W_shift")) == stored[SMALL:], \
            "src range did not relocate to dst in the logical object"
    finally:
        c2.shutdown()


def test_remap_extraction_restore(tmp_path):
    payload = rng_bytes(19, 2 * SMALL)
    server, c = boot(tmp_path, "remap-a")
    try:
        c.put_object("O", payload)
        dest = tmp_path / "identity"
        wait_snap(c, c.snapshot(str(dest), slices=[{"id": "O"}]))
    finally:
        c.shutdown()

    # full split: logical O extracted into two shard-local objects
    server2, c2 = boot(tmp_path, "remap-b")
    try:
        res = c2.restore_snapshot(str(dest), remap={"O": [
            {"logical": [0, SMALL], "id": "O_shard0",
             "local": [0, SMALL], "bytes": SMALL},
            {"logical": [SMALL, 2 * SMALL], "id": "O_shard1",
             "local": [0, SMALL], "bytes": SMALL}]})
        assert set(res["restored"]) == {"O_shard0", "O_shard1"}
        assert bytes(c2.get_object("O_shard0")) == payload[:SMALL]
        assert bytes(c2.get_object("O_shard1")) == payload[SMALL:]
        assert c2.query_object("O") is None, \
            "remap restore must not create the logical name"
    finally:
        c2.shutdown()

    # partial window: only the mid-range is extracted, rest skipped
    server3, c3 = boot(tmp_path, "remap-c")
    try:
        c3.restore_snapshot(str(dest), remap={"O": [
            {"logical": [SMALL // 2, SMALL // 2 + SMALL],
             "id": "O_mid", "local": [0, SMALL], "bytes": SMALL}]})
        assert bytes(c3.get_object("O_mid")) == \
            payload[SMALL // 2:SMALL // 2 + SMALL]
        assert c3.query_object("O") is None
        assert c3.query_object("O_shard0") is None
    finally:
        c3.shutdown()


def test_corrupt_payload_refused_store_untouched(tmp_path):
    payload = rng_bytes(23, N)
    server, c = boot(tmp_path, "corrupt-a")
    try:
        c.put_object("W_demo", payload)
        dest = tmp_path / "ck"
        wait_snap(c, c.snapshot(str(dest)))
    finally:
        c.shutdown()

    blob = bytearray((dest / "payload.bin").read_bytes())
    blob[10] ^= 0xFF
    (dest / "payload.bin").write_bytes(bytes(blob))

    server2, c2 = boot(tmp_path, "corrupt-b")
    try:
        with pytest.raises(ServiceError) as ei:
            c2.restore_snapshot(str(dest))
        assert ei.value.code == "VERIFY_FAILED"
        assert c2.query_object("W_demo") is None, \
            "refused restore must leave the store untouched"

        c2.restore_snapshot(str(dest), verify=False)   # documented opt-out
        got = bytes(c2.get_object("W_demo"))
        assert got != payload
        assert got[:10] == payload[:10] and got[11:] == payload[11:], \
            "verify=False must restore exactly the on-disk bytes"
    finally:
        c2.shutdown()


def test_refusal_mid_list_leaves_store_untouched(tmp_path):
    server, c = boot(tmp_path, "midlist-a")
    try:
        c.put_object("A_one", rng_bytes(29, SMALL))
        c.put_object("B_two", rng_bytes(31, SMALL))
        dest = tmp_path / "pair"
        wait_snap(c, c.snapshot(str(dest)))
    finally:
        c.shutdown()

    server2, c2 = boot(tmp_path, "midlist-b")
    try:
        filler = rng_bytes(37, 2 * SMALL)
        c2.put_object("B_two", filler)        # wrong-size resident target
        with pytest.raises(ServiceError) as ei:
            c2.restore_snapshot(str(dest), overwrite=True)
        assert ei.value.code == "BINDING_MISMATCH"
        assert c2.query_object("A_one") is None, \
            "validation refusal must reject the WHOLE restore"
        assert bytes(c2.get_object("B_two")) == filler
    finally:
        c2.shutdown()


def test_snapshot_json_schema_and_hashes(tmp_path):
    payload = rng_bytes(41, N)
    server, c = boot(tmp_path, "schema-a")
    try:
        c.put_object("W_demo", payload)
        dest = tmp_path / "ck"
        wait_snap(c, c.snapshot(str(dest)))
    finally:
        c.shutdown()

    m = snapshot_json(dest)
    assert m["schema"] == "dataflow-snapshot/v1"
    entry = next(e for e in m["slices"] if e["id"] == "W_demo")
    assert entry["src"] == [0, N]
    assert entry["logical_id"] == "W_demo"
    assert entry["dst"] == [0, N]
    assert entry["logical_bytes"] == N
    seg = entry["payload"]
    blob = (dest / "payload.bin").read_bytes()
    data = blob[seg["offset"]:seg["offset"] + seg["size"]]
    assert hashlib.blake2b(data, digest_size=16).hexdigest() == \
        entry["hash"]

    # relic/foreign schema string refuses loudly
    tampered = dict(m)
    tampered["schema"] = "dataflow-snap/s1"
    (dest / "snapshot.json").write_text(json.dumps(tampered))
    server2, c2 = boot(tmp_path, "schema-b")
    try:
        with pytest.raises(ServiceError) as ei:
            c2.restore_snapshot(str(dest))
        assert ei.value.code == "VERSION_SKEW"

        empty = tmp_path / "not_a_snapshot"
        empty.mkdir()
        with pytest.raises(ServiceError) as ei:
            c2.restore_snapshot(str(empty))
        assert ei.value.code == "IO_ERROR"
    finally:
        c2.shutdown()


def test_slice_validation_refusals(tmp_path):
    server, c = boot(tmp_path, "refuse")
    try:
        c.put_object("W_demo", rng_bytes(43, N))
        dest = str(tmp_path / "bad")

        with pytest.raises(ServiceError) as ei:      # src out of bounds
            c.snapshot(dest, slices=[{"id": "W_demo", "src": [0, N + 1]}])
        assert ei.value.code == "BAD_REQUEST"

        with pytest.raises(ServiceError) as ei:      # dst/src length skew
            c.snapshot(dest, slices=[{"id": "W_demo", "src": [0, SMALL],
                                      "dst": [0, 2 * SMALL]}])
        assert ei.value.code == "BAD_REQUEST"

        with pytest.raises(ServiceError) as ei:      # dst beyond logical
            c.snapshot(dest, slices=[
                {"id": "W_demo", "src": [0, SMALL], "logical_id": "X",
                 "dst": [SMALL, 2 * SMALL], "logical_bytes": SMALL}])
        assert ei.value.code == "BAD_REQUEST"

        with pytest.raises(ServiceError) as ei:      # conflicting logical
            c.snapshot(dest, slices=[
                {"id": "W_demo", "src": [0, SMALL], "logical_id": "X",
                 "dst": [0, SMALL], "logical_bytes": SMALL},
                {"id": "W_demo", "src": [SMALL, 2 * SMALL],
                 "logical_id": "X", "dst": [SMALL, 2 * SMALL],
                 "logical_bytes": 2 * SMALL}])
        assert ei.value.code == "BAD_REQUEST"

        with pytest.raises(ServiceError) as ei:      # unknown id
            c.snapshot(dest, slices=[{"id": "missing_obj"}])
        assert ei.value.code == "BAD_REQUEST"

        with pytest.raises(ServiceError) as ei:      # empty slice list
            c.snapshot(dest, slices=[])
        assert ei.value.code == "BAD_REQUEST"
    finally:
        c.shutdown()


def test_duplicate_snapshots_full_and_independent(tmp_path):
    payload = rng_bytes(47, SMALL)
    server, c = boot(tmp_path, "dup-a")
    try:
        c.put_object("W_root", payload)
        c.duplicate_object("W_root", "W_root@ck")
        dest = tmp_path / "dup"
        out = c.snapshot(str(dest))
        assert "n_deduped" not in out
        wait_snap(c, out)
    finally:
        c.shutdown()

    m = snapshot_json(dest)
    assert len(m["slices"]) == 2
    for entry in m["slices"]:
        assert "ref" not in entry["payload"], \
            "duplicates must store their own full payload"
        assert "lineage" not in entry and "version" not in entry
    server2, c2 = boot(tmp_path, "dup-b")
    try:
        c2.restore_snapshot(str(dest))
        assert bytes(c2.get_object("W_root")) == payload
        assert bytes(c2.get_object("W_root@ck")) == payload
    finally:
        c2.shutdown()


def test_snapshot_has_no_group_concept(tmp_path):
    server, c = boot(tmp_path, "nogroup")
    try:
        assert not hasattr(c, "create_object_group")
        assert not hasattr(c, "duplicate_object_group")
        c.put_object("W_solo", rng_bytes(53, SMALL))
        dest = tmp_path / "flat"
        wait_snap(c, c.snapshot(str(dest)))
        assert "object_groups" not in snapshot_json(dest)
        c.wipe("all", force=True)
        res = c.restore_snapshot(str(dest))
        assert "object_groups_recreated" not in res
        with pytest.raises(TypeError):
            c.restore_snapshot(str(dest), duplicates="recreate")
    finally:
        c.shutdown()


def boot_wide(tmp, name):
    """A daemon with enough slab for a restore whose payload work is
    long enough to observe in flight."""
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(socket_path=sock, fake=True,
                                 slab_backing_gib=1.2))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(300):
        try:
            with EngineClient(sock, client_name="probe"):
                break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    return server, EngineClient(sock, client_name=name)


def test_restore_runs_in_background_and_parks_writers(tmp_path):
    big = 512 << 20
    payload = b"\xa5" * big
    server, c = boot_wide(tmp_path, "bg-a")
    try:
        c.put_object("W_big", payload)
        dest = tmp_path / "bg"
        wait_snap(c, c.snapshot(str(dest)))
        c.wipe("all", force=True)

        out = c.restore_snapshot(str(dest), block=False)
        rid = out["restore_id"]
        replacement = b"\x5a" * big
        ticket = c.put_object("W_big", replacement, wait=False)
        time.sleep(0.10)
        status = c.restore_status(rid)
        assert status["state"] in ("restoring", "done")
        if status["state"] == "restoring":
            assert not ticket.done.is_set(), \
                "a write to a restore target must PARK until it completes"
        done = c.wait_restore(rid)
        assert done["state"] == "done"
        assert done["restored"] == ["W_big"]
        c.wait(ticket, timeout=30)
        assert bytes(c.get_object("W_big")) == replacement, \
            "the parked write must land AFTER the restore"
    finally:
        c.shutdown()


def test_restore_status_lifecycle(tmp_path):
    server, c = boot(tmp_path, "lifecycle")
    try:
        c.put_object("W_demo", rng_bytes(59, SMALL))
        dest = tmp_path / "lc"
        wait_snap(c, c.snapshot(str(dest)))
        c.wipe("all", force=True)

        out = c.restore_snapshot(str(dest), block=False)
        done = c.wait_restore(out["restore_id"])
        assert done["state"] == "done"
        assert done["restored"] == ["W_demo"]
        assert done["client_meta"] == {}
        assert bytes(c.get_object("W_demo")) == rng_bytes(59, SMALL)

        with pytest.raises(ServiceError) as ei:
            c.restore_status("restore-999999")
        assert ei.value.code == "UNKNOWN_RESTORE"
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
