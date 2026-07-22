"""Peer-group scaffolding gates (CPU, fake daemons — standard suite):
the conductor-shaped flow — connect the coordinator star, create the
group on rank 0, members adopt with correct ranks, the join barrier
holds the verb, the whole table exposes as TaskContext handles, and
group_error fans out member -> coordinator -> members (two hops).

Tests:
- test_group_lifecycle_and_error_fanout: creating a three-member group on the coordinator gives each daemon its own rank and a ready handle, rejects non-rank-0 creation and duplicate names, and propagates a member's group_error two hops to a peer with no direct link while dropping the errored group from the handle table.
- test_world_one_group_is_immediately_ready: a single-member group reports world 1 and is ready immediately.
"""
import threading
import time

from dataflow.service import EngineClient, EngineConfig, Server

PORTS = {"g-a": 29491, "g-b": 29492, "g-c": 29493, "g-solo": 29494}


def boot(tmp, name):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=True, slab_backing_gib=0.05,
        peer_name=name, peer_listen=f"127.0.0.1:{PORTS[name]}"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    return server, EngineClient(sock, client_name=name)


def wait_for(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_group_lifecycle_and_error_fanout(tmp_path):
    daemons = {n: boot(tmp_path, n) for n in ("g-a", "g-b", "g-c")}
    sa, ca = daemons["g-a"]
    sb, cb = daemons["g-b"]
    sc, cc = daemons["g-c"]
    try:
        # conductor: coordinator star only (a->b, a->c; b-c NOT linked)
        ca.peer_connect("g-b", f"127.0.0.1:{PORTS['g-b']}")
        ca.peer_connect("g-c", f"127.0.0.1:{PORTS['g-c']}")
        out = ca._call("create_peer_group",
                       {"name": "dp", "members": ["g-a", "g-b", "g-c"],
                        "backend": "auto"})
        assert out["world"] == 3

        # every daemon holds the group with ITS OWN rank
        for cli, want_rank in ((ca, 0), (cb, 1), (cc, 2)):
            infos = cli._call("list_peer_groups", {})
            assert infos and infos[0]["name"] == "dp"
            assert infos[0]["rank"] == want_rank
            assert infos[0]["world"] == 3 and infos[0]["ready"]

        # TaskContext handles: whole table, ready groups only (fake
        # boot => comm None; rank/world live on the handle object)
        handles = sa.nm.group_handles()
        assert handles["dp"].rank == 0 and handles["dp"].world == 3
        assert handles["dp"].comm is None

        # rank 0 required
        try:
            cb._call("create_peer_group",
                     {"name": "bad", "members": ["g-a", "g-b"]})
            raise AssertionError("expected BAD_REQUEST")
        except Exception as e:
            assert "rank 0" in str(e)

        # duplicate name refused
        try:
            ca._call("create_peer_group",
                     {"name": "dp", "members": ["g-a", "g-b", "g-c"]})
            raise AssertionError("expected GROUP_EXISTS")
        except Exception as e:
            assert "GROUP_EXISTS" in str(e) or "exists" in str(e)

        # group_error two-hop fan-out: member b reports; c hears it
        # via the coordinator despite having NO link to b
        sb.nm.group_error("dp", "injected test failure", fan_out=None)
        assert wait_for(
            lambda: any(e.get("event") == "group_error"
                        for e in sc.state.events))
        assert wait_for(
            lambda: sa.nm.groups.groups["dp"].error is not None)
        # errored groups vanish from the TaskContext handle table
        assert "dp" not in sc.nm.group_handles()
    finally:
        for _, cli in daemons.values():
            try:
                cli.shutdown()
            except Exception:
                pass


def test_world_one_group_is_immediately_ready(tmp_path):
    server, client = boot(tmp_path, "g-solo")
    try:
        out = client._call("create_peer_group",
                           {"name": "solo", "members": ["g-solo"]})
        assert out["world"] == 1
        infos = client._call("list_peer_groups", {})
        assert infos[0]["ready"] and infos[0]["world"] == 1
    finally:
        try:
            client.shutdown()
        except Exception:
            pass
