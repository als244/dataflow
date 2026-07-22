# Engine networking

How dataflow engines talk: to their clients, to each other, and as
collective groups. This is the networking layer's story — its
planes, threads, protocols, and the interfaces everything else
consumes. Companion docs: [engine_service.md](engine_service.md)
(the verb surface), [distributed_training.md](distributed_training.md)
(how training composes all of this), [topology](usage.md) for the
per-setup file the conductor reads.

## The three planes

Every byte an engine moves travels one of three planes, and keeping
them straight explains most of the design:

- **The verb plane** — client ↔ daemon, over the daemon's unix
  socket. Framed JSON requests and replies (`put_object`, `run`,
  `create_peer_group`, ...). Local to a box; remote daemons are
  reached by forwarding the socket over ssh. The conductor lives
  entirely on this plane.
- **The control plane** — daemon ↔ daemon, over TCP **peer links**.
  Framed control messages: hellos, heartbeats, transfer rendezvous
  and completion, group join/ack/error, collective headers. Small,
  latency-critical frames (links set `TCP_NODELAY` — a sub-MSS
  exchange must never sit out a delayed ACK).
- **The data plane** — the payload paths. Object transfer payloads
  ride the link socket (small payloads inline in their RTS frame;
  large ones chunked) or, when both ends brought RDMA up, one-sided
  `RDMA_WRITE`s from the sender's pinned slab straight into the
  receiver's reserved extent — the CPU never touches payload and
  DONE rides the control plane behind the RC ack. The nccl backend
  is its own data plane (device-direct collectives on the group
  stream over NCCL's transports).

## Peer links

`peer_connect(name, "ip:port")` dials a daemon's peer listener and
performs a HELLO exchange: each end learns the other's peer name
and whether it offers RDMA. One link per daemon pair, shared by
everything — transfers, group traffic, probes all multiplex over
it, discriminated by frame kind (and, for collective frames, the
group name they carry). Links are kept honest by heartbeats: the
housekeeping thread PINGs each link on a fixed cadence and drops a
peer that goes silent past the down-threshold, which parks its transfers and fans
errors to any groups that spanned it.

When both ends advertised RDMA, the link additionally brings up a
reliable-connected QP pair. RDMA is transport-agnostic at the verbs
level — the same RC QPs and one-sided `RDMA_WRITE`s run over either
RoCE (Ethernet link layer) or InfiniBand — but a connection is
*addressed* differently on each: RoCE by GID, InfiniBand by LID. The
engine wires the RoCE path today (an ACTIVE Ethernet port with an
IPv4-mapped RoCE v2 GID — anything less reports no-RDMA rather than
arming a QP that cannot pass traffic) and stops loudly on a
non-Ethernet port; InfiniBand is a clean drop-in at that seam (LID
selection and LID-based addressing, nothing else changes). Bring-up
is an `RDMA_INFO` exchange over the control plane carrying each end's
qpn, addressing, and path-MTU (the QPs run at the min of both ends'
active MTUs), then an `RDMA_UP` confirmation once both sides reach
RTS. RDMA that fails to come up
**demotes the link to the socket data plane, loudly and
symmetrically — never fatally**; both ends make the same lane
decision from the same handshake facts, because a link whose two
ends disagree about the lane deadlocks.

## Threading architecture

Per daemon, every thread the networking layer runs, what it owns,
and how the count scales:

| thread | count | role |
|---|---|---|
| main | 1 | accepts verb-plane connections |
| conn | per client | serves one client's verb stream |
| dispatcher | per active run | executes the run (task dispatch + completion polling) |
| nm-accept | 1 | accepts inbound peer links |
| nm-link-\<peer\> | per link | blocking reader: dispatches every inbound frame |
| nm-housekeeping | 1 | heartbeats, peer-down sweeps, protocol ticks |
| nm-coll-\<group\> | per hostmem group | the collective exchange worker |
| nm-comm-build-\<group\> | transient | attaches a group's backend, then exits |
| nm-rdma-write-\<transfer\> | transient | one-sided writer for one outbound rdma transfer |
| snapshot-writer | 1 | background snapshot streaming |

The census is therefore: a **fixed four** (main, nm-accept,
nm-housekeeping, snapshot-writer) **+ one per client connection +
one per active run + one per peer link + one per hostmem group**,
with two short-lived kinds: a builder per group creation and a
writer per outbound rdma transfer. Links add NO persistent sender
threads: socket-lane sends (frames and payload chunks alike) are
issued inline by whichever thread holds the frame, serialized per
link, while an rdma transfer spawns its transient writer precisely
so the reader never blocks on the write's completion polling. nccl
groups add no persistent threads either — collectives enqueue
device-direct on the group stream.

Two rules make the concurrency tractable. **The reader never
blocks**: a link's reader thread dispatches frames and returns —
it never builds group state (frames for a group whose backend is
still attaching are parked and drained, in order, when the attach
completes) and never waits on RDMA bring-up (which its own frame
processing services). **Group state has one writer**: records
mutate under the group table's lock, and each group's backend is
attached by exactly one builder.

## Groups

A **peer group** is a named, ordered member list with a backend:
rank = position in the list. Training declares its groups in the
topology file (`[groups.dp] members = [...] backend = "auto"`) and
the conductor materializes them at bring-up; any client can create
ad-hoc groups the same way.

**The star overlay.** Groups are an overlay on the link mesh: the
group's **coordinator** — rank 0's daemon — must hold a link to
every member (its *star*). Links are shared, so overlapping groups
reuse them; the conductor connects exactly the stars its groups
need, never a forced full mesh.

**One verb, an attachment barrier.** There is no join verb. The
client calls `create_peer_group(name, members, backend)` once, on
the coordinator:

```python
coord.peer_connect("m1", "10.0.0.2:29700")   # the star
coord.peer_connect("m2", "10.0.0.3:29700")
coord._call("create_peer_group",
            {"name": "dp", "members": ["m0", "m1", "m2"],
             "backend": "auto"})
```

The coordinator pushes a `GROUP_JOIN` frame down each star link;
members adopt the group from the frame (learning their rank from
their position), **attach their backend, and only then** answer
`GROUP_ACK`. When the barrier fills, the coordinator attaches its
own backend and the verb returns. The barrier is therefore an
*attachment* barrier: `create_peer_group` returning means every
rank's backend is live and the group can carry collectives
immediately. A member that fails to attach withholds its ack and
the verb fails on the barrier timeout; a member that fails later
fans `GROUP_ERROR` member → coordinator → members, and the errored
group refuses further work.

## Backends: from record to handle to comm

Each daemon tracks its groups as records; a READY record carries a
**GroupHandle** — `{name, rank, world, backend, members, stream,
comm}` plus the collective methods (`allreduce`, `broadcast`,
`reduce`, `reduce_scatter`, `all_gather`) — and under the handle
sits the backend's comm:

- **hostmem** — the engine's own staged exchange. Operands stage
  through pinned slab regions; a per-group worker thread runs the
  wire exchange (RDMA lane when the link has it, socket frames
  otherwise — the same static, symmetric lane decision as
  transfers) and reduces; a mapped flag word releases the parked
  group stream when a result is ready, so the caller's stream
  ordering never involves the CPU. Always available; the fallback
  and CI lane.
- **nccl** — the production lane for real GPUs. The communicator is
  bootstrapped *inside* the join barrier (the uniqueId rides the
  join frame; init is collective), and operations enqueue
  device-direct on the group stream. `backend = "auto"` resolves to
  nccl on any real boot and hostmem otherwise.

Both backends attach through the same path and the same barrier;
the backend only changes what "attach" builds.

## Tasks, programs, and the TaskContext

Programs bind to groups by **role**, not by name: a task declares
`comm_groups={"dp": "<group name>"}` in its spec
([program_schema.md](program_schema.md)), and at run time the
engine injects the live handles into the task's context — block
code looks up its role and drives the handle:

```python
gh = ctx.groups.get(spec.comm_groups["dp"])
gh.allreduce(grad_view)        # on the group stream
```

The producer contract for what a block may hand a collective —
engine-owned buffers, stream-edge discipline — lives in
[task-contract.md](task-contract.md). Because group creation is an
attachment barrier, a run launched after `create_peer_group`
returns can use its groups from its first task; there is no
bring-up race to order against.

## The conductor's role

The conductor is a verb-plane client and never appears on the peer
plane: it launches daemons, connects the stars its topology's
groups require, creates each group with one verb on that group's
coordinator, and then drives runs. Group *membership* is topology
data; group *existence* is engine state the conductor materializes.
Before any fleet run it enforces the version handshake — every
member on the same repo commit and torch/cuda stack, the conductor
itself clean — because the failure modes of skew are silent
([distributed_training.md](distributed_training.md)).

## Benchmarking the planes

Two verbs measure the peer plane from inside the engine — through
the real transports, with no verb-plane round trips inside the
timed window.

**`coll_bench` — the collective path.** Replays a transfer pattern
through the same enqueue/exchange/reduce machinery the optimizer
tasks drive. Collectives are collective: **both group members must
be inside the verb concurrently** (drive the second member's client
from its own process). Args: `{group, sizes: [bytes, ...], dtype,
reps, verify, rs_ag_identity}` — `verify` fills known patterns and
checks the reduced result; `rs_ag_identity` additionally asserts
the reduce_scatter + all_gather == allreduce identity at the given
size. Returns per-rep walls plus the comm's phase-time breakdown
(staging, exchange, reduce) and which lane served it.

**`p2p_bench` — the transfer path.** Point-to-point is not a
collective, so this verb runs on **one** daemon: it drives timed
transfers to a named peer over the link's engine-default lane
(rdma when the link has it, socket otherwise — the same lane real
`send_object` traffic takes) and times each transfer to its
remote-commit acknowledgement inside the daemon. Args:
`{peer, sizes: [bytes, ...], iters}`. Returns per-transfer walls, a
sustained back-to-back figure per size, and the lane that served
the run — the honest engine-side answer to "what does an object
transfer cost on this link".

Both verbs report the lane they actually used; a demoted link
benches as what it is. The standalone sweep harness in the bench
tooling drives these across size grids and renders
perftest-style tables.

## Interface quick reference

| surface | calls |
|---|---|
| links | `peer_connect`, `peer_disconnect`, `list_peers` |
| groups | `create_peer_group` (coordinator only; returns = all attached) |
| transfers | `send_object`, `send_object_group`, `transfer_status`, `wait_transfer` |
| diagnostics | `coll_bench`, `p2p_bench` (see Benchmarking the planes), `subscribe_events` (`peer_up/peer_down/peer_rdma_up/group_*`), `engine_status` |
