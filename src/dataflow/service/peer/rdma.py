"""rdma-host transport: RDMA_WRITE into the remote store slab (pyverbs).

One RdmaEngine per daemon (device + PD + the WHOLE-SLAB MR registered
once at NM boot — allocation-is-reservation means any extent is a valid
target with zero per-object work). One RC QP per peer link, brought up
by exchanging {gid, gid_index, qpn} over the control link right after
HELLO. The data plane is ONE-SIDED: the sender's writer thread posts
chunked RDMA_WRITEs from its own slab (local MR lkey) straight into the
receiver's reserved extent (raddr + rkey from the ADDR reply); the
receiver's CPU never touches payload — DONE rides the control link and
the checksum verifies over the landed extent.

GID selection: sysfs scan for a RoCE v2 IPv4-mapped GID (the
::ffff:a.b.c.d form) on the configured device/port — exactly the entry
the direct-link address produces.
"""
from __future__ import annotations

import threading
from pathlib import Path

from pyverbs.addr import AHAttr, GID, GlobalRoute
from pyverbs.cq import CQ
from pyverbs.device import Context
from pyverbs.libibverbs_enums import (
    ibv_access_flags,
    ibv_mtu,
    ibv_qp_state,
    ibv_qp_type,
    ibv_send_flags,
    ibv_wr_opcode,
)
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPAttr, QPCap, QPInitAttr
from pyverbs.wr import SGE, SendWR

ACCESS = (ibv_access_flags.IBV_ACCESS_LOCAL_WRITE
          | ibv_access_flags.IBV_ACCESS_REMOTE_WRITE)
RDMA_CHUNK = 128 * 1024 * 1024
POLL_BUDGET = 200_000_000        # spins before declaring a wedge


def roce_v2_ipv4_gid(device: str, port: int = 1):
    """(gid_index, gid_str) of the RoCE v2 IPv4-mapped GID on an
    ACTIVE port, or None. Link-local (fe80) v2 GIDs are deliberately
    rejected: they are MAC-derived and exist even on ports that are
    down or address-less, so accepting one arms QPs that can never
    pass traffic — report no-RDMA instead and let the transport
    demote to socket."""
    base = Path(f"/sys/class/infiniband/{device}/ports/{port}")
    try:
        if not (base / "state").read_text().strip().startswith("4"):
            return None
    except OSError:
        return None
    for entry in sorted((base / "gid_attrs/types").iterdir(),
                        key=path_index):
        try:
            typ = entry.read_text().strip()
            gid = (base / "gids" / entry.name).read_text().strip()
        except OSError:
            continue
        if typ != "RoCE v2":
            continue
        if gid.startswith("0000:0000:0000:0000:0000:ffff"):
            return int(entry.name), gid
    return None


def path_index(p: Path) -> int:
    return int(p.name)


def port_link_layer(device: str, port: int = 1) -> str:
    """The port's link layer, from sysfs: 'Ethernet' (RoCE) or
    'InfiniBand' (IB). RoCE and IB share the RDMA verbs but address a
    connection differently — RoCE by GID, IB by LID — so the engine
    branches on this."""
    try:
        return (Path(f"/sys/class/infiniband/{device}/ports/{port}")
                / "link_layer").read_text().strip()
    except OSError:
        return "unknown"


class RdmaLinkQP:
    """One peer link's RC QP (+ its CQ). Serialized by ``wlock`` — the
    writer thread owns posts; connect happens once from the reader."""

    def __init__(self, engine: "RdmaEngine"):
        self.engine = engine
        self.cq = CQ(engine.ctx, 256)
        cap = QPCap(max_send_wr=64, max_recv_wr=8, max_send_sge=1,
                    max_recv_sge=1)
        init = QPInitAttr(qp_type=ibv_qp_type.IBV_QPT_RC, scq=self.cq,
                          rcq=self.cq, cap=cap)
        self.qp = QP(engine.pd, init)
        self.wlock = threading.Lock()
        self.ready = False

    def local_info(self) -> dict:
        return {"gid": self.engine.gid_str,
                "gid_index": self.engine.gid_index,
                "qpn": self.qp.qp_num, "psn": 0,
                "mtu": self.engine.active_mtu}

    def connect(self, remote: dict) -> None:
        attr = QPAttr(qp_state=ibv_qp_state.IBV_QPS_INIT)
        attr.pkey_index = 0
        attr.port_num = self.engine.port
        attr.qp_access_flags = ACCESS
        self.qp.to_init(attr)
        # RoCE addressing: a global route over the peer's GID. The IB
        # drop-in branches here to LID addressing instead —
        # AHAttr(dlid=remote["lid"], is_global=0), no GlobalRoute.
        gr = GlobalRoute(dgid=GID(remote["gid"]),
                         sgid_index=self.engine.gid_index)
        ah = AHAttr(gr=gr, is_global=1, port_num=self.engine.port)
        attr = QPAttr(qp_state=ibv_qp_state.IBV_QPS_RTR)
        # both ends compute the same min; peers that predate the mtu
        # field are driven at the old fixed 1024
        mtu = min(self.engine.active_mtu,
                  int(remote.get("mtu", ibv_mtu.IBV_MTU_1024)))
        if remote["gid"] == self.engine.gid_str:
            # mlx5 internal loopback is not reliably wire-equivalent,
            # and BOX-DEPENDENT: of two same-NIC/same-fw hosts, one
            # fails RC bring-up to its own GID at path MTU 4096 and
            # hits data timeouts under sustained collectives at 2048,
            # while the other runs 4096 self-loop clean (suspect
            # board/IOMMU, not the NIC). Self-connections stay at
            # 1024 — the only value proven under sustained collectives
            # everywhere; wire connections keep the negotiated MTU.
            mtu = min(mtu, int(ibv_mtu.IBV_MTU_1024))
        attr.path_mtu = mtu
        attr.dest_qp_num = int(remote["qpn"])
        attr.rq_psn = int(remote.get("psn", 0))
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 12
        attr.ah_attr = ah
        self.qp.to_rtr(attr)
        attr = QPAttr(qp_state=ibv_qp_state.IBV_QPS_RTS)
        attr.sq_psn = 0
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.max_rd_atomic = 1
        self.qp.to_rts(attr)
        self.ready = True

    def write(self, src_ptr: int, raddr: int, rkey: int, length: int,
              on_progress=None) -> None:
        """Chunked signaled RDMA_WRITEs; blocks the calling (writer)
        thread polling the CQ per chunk. Raises RuntimeError on any
        non-success completion (fail-stop)."""
        lkey = self.engine.slab_mr.lkey
        done = 0
        with self.wlock:
            while done < length:
                n = min(RDMA_CHUNK, length - done)
                sge = SGE(src_ptr + done, n, lkey)
                wr = SendWR(wr_id=done,
                            opcode=ibv_wr_opcode.IBV_WR_RDMA_WRITE,
                            num_sge=1, sg=[sge],
                            send_flags=ibv_send_flags.IBV_SEND_SIGNALED)
                wr.set_wr_rdma(rkey, raddr + done)
                self.qp.post_send(wr)
                spins = 0
                while True:
                    got, wcs = self.cq.poll(1)
                    if got:
                        if wcs[0].status != 0:
                            raise RuntimeError(
                                f"RDMA_WRITE wc status {wcs[0].status}")
                        break
                    spins += 1
                    if spins > POLL_BUDGET:
                        raise RuntimeError("RDMA_WRITE completion wedge")
                done += n
                if on_progress is not None:
                    on_progress(done)


class RdmaEngine:
    def __init__(self, device: str, *, port: int = 1):
        # The engine wires RDMA over RoCE (Ethernet link layer, GID
        # addressing). InfiniBand (LID addressing) is a documented
        # drop-in: select the port's LID here instead of a RoCE GID, and
        # use LID-based AHAttr in RdmaLinkQP.connect — the RC QP /
        # RDMA_WRITE / MR data plane is identical. Until then a
        # non-Ethernet port stops here loudly rather than arming a QP
        # that cannot pass traffic.
        link_layer = port_link_layer(device, port)
        if link_layer != "Ethernet":
            raise RuntimeError(
                f"{device} port {port} link layer is {link_layer!r}; only "
                "RoCE (Ethernet) is wired so far — InfiniBand is a clean "
                "drop-in at the RdmaEngine/RdmaLinkQP seams")
        picked = roce_v2_ipv4_gid(device, port)
        if picked is None:
            raise RuntimeError(f"no RoCE v2 GID on {device} port {port}")
        self.gid_index, self.gid_str = picked
        self.device = device
        self.port = port
        self.ctx = Context(name=device)
        self.active_mtu = int(self.ctx.query_port(port).active_mtu)
        self.pd = PD(self.ctx)
        self.slab_mr: MR | None = None
        self.slab_base = 0

    def register_slab(self, ptr: int, size: int) -> None:
        self.slab_mr = MR(self.pd, size, ACCESS, address=ptr)
        self.slab_base = ptr

    def make_link_qp(self) -> RdmaLinkQP:
        return RdmaLinkQP(self)

    def rkey(self) -> int:
        return self.slab_mr.rkey
