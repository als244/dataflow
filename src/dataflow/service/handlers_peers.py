"""Peer-plane verbs (dataflow-peer/s2): connect/status, send_object,
transfer_status, and the INTERNAL inbound-commit op the NetworkManager
queues through the dispatcher. Enabled when EngineConfig.peer_name is
set; without it the verbs answer PEER_DISABLED and no NM threads exist.
"""
from __future__ import annotations

from .peer.nm import NetworkManager
from .wire import ServiceError


class SendCompletion:
    """Releases the outbound read-lease when the transfer resolves."""

    def __init__(self, store, oid: str):
        self.store = store
        self.oid = oid

    def __call__(self, ticket) -> None:
        self.store.release_leases([self.oid])


class BenchFinish:
    """p2p_bench per-transfer completion latch."""

    def __init__(self):
        import threading

        self.done = threading.Event()

    def __call__(self, ticket) -> None:
        self.done.set()


def install(server) -> None:
    store = server.store
    st = server.state
    cfg = server.config
    nm = None
    if getattr(cfg, "peer_name", None):
        nm = NetworkManager(server, peer_name=cfg.peer_name,
                            listen=getattr(cfg, "peer_listen", None),
                            chunk_bytes=getattr(cfg, "peer_chunk_bytes",
                                                128 << 20),
                            rdma_device=getattr(cfg, "peer_rdma_device",
                                                None))
        server.nm = nm
        nm.start()

    def require_nm() -> NetworkManager:
        if nm is None:
            raise ServiceError("PEER_DISABLED",
                               "boot with peer_name/peer_listen")
        return nm

    # ---------------------------------------------- queued verbs
    def peer_connect(call):
        a = call.args
        return require_nm().connect(a["name"], a["control_addr"])

    def peer_disconnect(call):
        a = call.args
        m = require_nm()
        with m.lock:
            link = m.links.get(a["peer_id"])
        if link is None:
            return {"ok": True, "already": True}
        m.drop_link(link, why="peer_disconnect")
        return {"ok": True}

    def send_object(call):
        a = call.args
        m = require_nm()
        oid = a["oid"]
        rec = store.objects.get(oid)
        if rec is None:
            raise ServiceError("UNKNOWN_OBJECT", oid)
        if a.get("release_after"):
            raise ServiceError("BAD_REQUEST",
                               "release_after lands with the leases/"
                               "release second-writer audit")
        dest = a.get("as_id") or oid
        store.acquire_leases([oid])
        src_ptr = store.host_ptr(rec) if store.slab is not None else None
        try:
            send_id = m.start_send(
                a["peer_id"], dest, store.view(rec),
                overwrite=bool(a.get("overwrite")),
                meta=dict(rec.meta), on_finish=SendCompletion(store, oid),
                src_ptr=src_ptr)
        except Exception:
            store.release_leases([oid])
            raise
        return {"send_id": send_id, "dest_id": dest,
                "bytes": rec.size_bytes}

    def send_object_group(call):
        a = call.args
        ids = store.resolve_object_group(a["ogid"])
        sent = []
        for oid in ids:
            sub = dict(call.args)
            sub["oid"] = oid
            sub.pop("ogid", None)
            call2 = type(call)(ticket=call.ticket, session_id=call.session_id,
                               op="send_object", args=sub, payload=None,
                               reply_to=call.reply_to)
            sent.append(send_object(call2)["send_id"])
        return {"send_ids": sent, "n": len(sent)}

    def peer_commit_inbound(call):
        a = call.args
        rec = store.adopt_inbound(a["dest_id"], a["extent"],
                                  a["size_bytes"], meta=a.get("meta"),
                                  from_peer=a.get("from_peer"))
        return {"ok": True, "oid": rec.id, "version": rec.version}

    def create_peer_group(call):
        a = call.args
        return require_nm().create_group(
            a["name"], list(a["members"]), a.get("backend", "auto"))

    def destroy_peer_group(call):
        a = call.args
        nm_now = require_nm()
        with nm_now.groups.lock:
            rec = nm_now.groups.groups.get(a["name"])
        handle = rec.handle if rec is not None else None
        if handle is not None and handle.comm is not None:
            handle.comm.close()
        nm_now.groups.drop(a["name"])
        return {"ok": True}

    def p2p_bench(call):
        """Point-to-point transfer bench — coll_bench's sibling for
        the TRANSFER path. Not a collective, so it runs on ONE
        daemon: timed sends of slab-scratch payloads to ``peer``
        over the link's engine-default lane (rdma when the link
        has it, socket otherwise — the lane real send_object
        traffic takes); each wall runs from start_send to the
        remote-commit acknowledgement with no verb-plane hops
        inside the window. args {peer, sizes: [bytes...], iters}.
        Returns the lane, per-size acked walls, and a pipelined
        sustained wall (all sends in flight, then wait all).
        Remote-side bench objects land as p2pbench_*."""
        import time as time_mod

        a = call.args
        m = require_nm()
        if store.slab is None:
            raise ServiceError("BAD_REQUEST",
                               "p2p_bench requires a real "
                               "(pinned) boot")
        peer = a["peer"]
        link = m.links.get(peer)
        if link is None or not link.alive:
            raise ServiceError("PEER_UNREACHABLE", peer)
        qp = getattr(link, "rdma_qp", None)
        lane = ("rdma" if m.rdma is not None and qp is not None
                and qp.ready else "socket")
        sizes = [int(s) for s in a["sizes"]]
        iters = int(a.get("iters", 5))
        from .hostmem import bytes_view

        rows = []
        for nbytes in sizes:
            ext = store.alloc_scratch(nbytes)
            try:
                ptr = store.slab.ptr + ext.offset
                view = bytes_view(ptr, nbytes)
                view[:1] = b"\x5a"
                view[-1:] = b"\xa5"
                walls = []
                for rep in range(iters + 1):
                    fin = BenchFinish()
                    t0 = time_mod.monotonic()
                    send_id = m.start_send(
                        peer, f"p2pbench_{nbytes}", view,
                        overwrite=True, meta={}, on_finish=fin,
                        src_ptr=ptr)
                    if not fin.done.wait(120.0):
                        raise ServiceError(
                            "INTERNAL", f"p2p_bench {nbytes}B "
                                        f"rep {rep}: timeout")
                    row = m.transfers.get(send_id, {})
                    if row.get("state") != "done":
                        raise ServiceError(
                            "INTERNAL", f"p2p_bench {nbytes}B "
                                        f"rep {rep}: "
                                        f"{row.get('state')}")
                    if rep:
                        walls.append(time_mod.monotonic() - t0)
                fins = []
                t0 = time_mod.monotonic()
                for k in range(iters):
                    fin = BenchFinish()
                    fins.append(fin)
                    m.start_send(peer, f"p2pbench_{nbytes}_p{k}",
                                 view, overwrite=True, meta={},
                                 on_finish=fin, src_ptr=ptr)
                for fin in fins:
                    if not fin.done.wait(300.0):
                        raise ServiceError(
                            "INTERNAL", f"p2p_bench {nbytes}B "
                                        f"sustained: timeout")
                sustained = time_mod.monotonic() - t0
                rows.append({"bytes": nbytes, "walls_s": walls,
                             "sustained_s": sustained})
            finally:
                store.release_scratch(ext)
        return {"lane": lane, "iters": iters, "rows": rows}

    def coll_bench(call):
        """Replay a transfer PATTERN through the real collective path
        (the same enqueue/exchange/reduce the optimizer tasks drive):
        args {group, sizes: [bytes...], dtype, reps}. Both group
        members must call concurrently (collectives are collective).
        Returns per-rep walls + the comm's phase-time breakdown —
        the microbench behind every idle-gap investigation."""
        import time as time_mod

        import torch

        from ..runtime.interop import TORCH_DTYPE_BY_NAME

        a = call.args
        m = require_nm()
        gh = m.group_handles().get(a["group"])
        if gh is None or gh.comm is None:
            raise ServiceError("BAD_REQUEST",
                               f"group {a['group']!r} not ready")
        dt = TORCH_DTYPE_BY_NAME[a.get("dtype", "bf16")]
        sizes = [int(s) for s in a["sizes"]]
        reps = int(a.get("reps", 3))
        verify = bool(a.get("verify"))
        fill = float(gh.rank + 1)
        want = float(gh.world * (gh.world + 1) / 2)
        tensors = []
        for nb in sizes:
            t = torch.full((nb // dt.itemsize,), fill, dtype=dt,
                           device="cuda")
            tensors.append(t)
        before = dict(getattr(gh.comm, "stats", {}))
        walls = []
        for rep in range(reps):
            if verify:
                for t in tensors:
                    t.fill_(fill)
            torch.cuda.synchronize()
            t0 = time_mod.monotonic()
            for t in tensors:
                gh.allreduce(t)
            gh.stream.synchronize()
            walls.append(round(time_mod.monotonic() - t0, 6))
            if verify:
                for t in tensors:
                    if (float(t[0]) != want or float(t[-1]) != want
                            or float(t[t.numel() // 2]) != want):
                        raise ServiceError(
                            "INTERNAL",
                            f"coll_bench verify failed: got "
                            f"{float(t[0])}/{float(t[t.numel()//2])}/"
                            f"{float(t[-1])}, want {want}")
        rs_ag_ok = None
        if a.get("rs_ag_identity"):
            # the ZeRO identity, backend-blind: rs into my slice then
            # ag back must equal the allreduce of the same fill
            n = int(a["rs_ag_identity"]) // dt.itemsize
            n -= n % gh.world
            full = torch.full((n,), fill, dtype=dt, device="cuda")
            own = torch.empty(n // gh.world, dtype=dt, device="cuda")
            gh.reduce_scatter(full, own)
            gathered = torch.empty(n, dtype=dt, device="cuda")
            gh.all_gather(own, gathered)
            gh.stream.synchronize()
            rs_ag_ok = (float(gathered[0]) == want
                        and float(gathered[-1]) == want)
            if not rs_ag_ok:
                raise ServiceError(
                    "INTERNAL",
                    f"rs+ag identity failed: {float(gathered[0])}/"
                    f"{float(gathered[-1])} want {want}")
        after = getattr(gh.comm, "stats", {})
        delta = {}
        for key, val in after.items():
            base = before.get(key, 0)
            delta[key] = round(val - base, 6) if isinstance(val, float) \
                else val - base
        total = sum(sizes)
        gbps = [round(total * 8 / w / 1e9, 2) for w in walls]
        lane = ("nccl" if type(gh.comm).__name__ == "NcclComm"
                else ("rdma" if gh.comm.rdma_qp() is not None
                      else "socket"))
        return {"walls_s": walls, "gbps_per_rep": gbps,
                "stats": delta, "bytes_per_rep": total,
                "lane": lane, "verified": verify,
                "rs_ag_ok": rs_ag_ok,
                "rdma_lane": lane == "rdma"}

    server.dispatcher.handlers.update({
        "coll_bench": coll_bench,
        "p2p_bench": p2p_bench,
        "create_peer_group": create_peer_group,
        "destroy_peer_group": destroy_peer_group,
        "peer_connect": peer_connect,
        "peer_disconnect": peer_disconnect,
        "send_object": send_object,
        "send_object_group": send_object_group,
        "peer_commit_inbound": peer_commit_inbound,
    })

    # ---------------------------------------------- fast verbs
    def list_peers(conn, args):
        if nm is None:
            return []
        with nm.lock:
            return [{"peer_id": l.peer_id, "state": "up" if l.alive
                     else "down"} for l in nm.links.values()]

    def peer_status(conn, args):
        m = require_nm()
        with m.lock:
            link = m.links.get(args["peer_id"])
            if link is None:
                raise ServiceError("PEER_UNREACHABLE", args["peer_id"])
            inflight = [r for r in m.transfers.values()
                        if r["peer_id"] == link.peer_id
                        and r["state"] in ("negotiating", "moving",
                                           "committing")]
            return {"peer_id": link.peer_id, "state": "up",
                    "transfers": len(inflight),
                    "peak_gbps": dict(link.peak_gbps)}

    def transfer_status(conn, args):
        return require_nm().transfer_status(args["send_id"])

    def profiler_control(conn, args):
        """start|stop capture on THIS daemon's profiler (the vendor
        annotator behind the backend) — the conductor brackets chosen
        steps; nsys --capture-range=cudaProfilerApi records only the
        bracketed region."""
        from . import execution

        ann = execution.get_backend(store).annotator
        action = args["action"]
        if action == "start":
            ann.start_capture()
        elif action == "stop":
            ann.stop_capture()
        else:
            raise ServiceError("BAD_REQUEST", f"action {action!r}")
        return {"ok": True, "enabled": getattr(ann, "enabled", False)}

    def list_peer_groups(conn, args):
        return [] if nm is None else nm.groups.infos()

    server.fast_handlers.update({
        "profiler_control": profiler_control,
        "list_peer_groups": list_peer_groups,
        "list_peers": list_peers,
        "peer_status": peer_status,
        "transfer_status": transfer_status,
    })
