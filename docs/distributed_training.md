# Distributed training: groups, sharding, and checkpointing

This guide explains how multi-daemon training works end to end: the
communication layer, the sharding API that expresses *who owns what*,
the parallelism configurations built on it, and how checkpointing and
resume behave for each. Everything here is driven from
`tools/train/train.py` and configured by `topology.toml` — no machine
facts live in code.

## 1. The pieces

**Daemons and the conductor.** Every GPU runs one `dataflowd` daemon.
A *conductor* (`dataflow_training/run/conductor.py`, driven by `train.py train`) launches
daemons over the hosts in `topology.toml`, registers a per-rank
program with each, feeds token rounds, and drives lockstep steps.
Daemons never talk to the conductor's Python state — everything
crosses the wire as programs, objects, and run calls.

**Groups and backends.** Collectives run over a *peer group* created
by the conductor — one `create_peer_group` verb on the coordinator
daemon; members join and attach their backends inside the barrier
(the protocol lives in
[engine_networking.md](engine_networking.md)). The group's backend comes from the topology
(`backend = "auto"` resolves to **nccl** on any real boot; the
hostmem lane remains for CI/loopback). Before any fleet run the
conductor performs a **handshake**: every member must be on the same
repo commit *and* the same torch/cuda/cudnn versions, and the
conductor itself must have no uncommitted tracked changes. This is
enforced because the failure modes of skew are silent (mixed-version
collectives corrupt quietly; version-skewed kernels break replicated
compute — see §3, tensor parallelism).

## 2. The sharding API (`dataflow_training/distributed/sharding.py`)

A `ShardPlan` is a group-scoped list of assignments over the packed
weight layouts:

```python
Region(object_root, field, rows=(lo, hi), dim=0)   # a slice of one field
Assignment(region, owner, resident=ALL_RANKS)
```

Each assignment carries **two independent axes**:

- **`owner`** — *who runs the optimizer* for the region: that rank
  holds the region's optimizer state (m/v — nobody else has those
  bytes), computes the update each step, and propagates the result.
  `owner=ALL_RANKS` means every rank updates redundantly (sound only
  when gradients are identical across ranks, i.e. after a data-
  parallel allreduce).
- **`resident`** — *who holds the parameter bytes at all*.
  `ALL_RANKS` = replicated weights. A single rank = the shard
  physically lives only there (tensor parallelism); per-rank layouts
  then materialize the field at shard shape.

`dim` selects the shard axis: `dim=0` slices rows (byte-contiguous —
the form the optimizer byte-range machinery uses), `dim=1` slices
columns (a pure layout transform, used by resident-narrowed TP
fields).

Plans are built by helpers, validated (`plan.validate()` checks full
cover, bounds, one axis per field, and optimizer-aware rules like
muon's whole-matrix requirement), and checked against a consumer
contract (`plan.consumable("optimizer")` or `"tp"`). Everything a
task needs at runtime is baked into the task as plain data — compute
keys select code, `block_params` carries the geometry (shard regions,
slices), and `comm_groups` maps a comm purpose to the NAME of the
peer group serving it (`{"dp": <group name>}` for the gradient
exchange, `{"tp": <group name>}` for tensor-parallel collectives —
the name is whatever the topology calls the group, e.g. `node`). The
program stores names, never handles: the live group binds per run,
and a run without the group executes the same task standalone —
which is exactly what the fleet warm-up does.

## 3. The parallelism configurations

### Plain data parallelism (default)

```
python tools/train/train.py train --preset l3_1b --steps 1000 --topology topology.toml \
    --rounds 6,2 --out results/pretrain/run.json
```

Every rank holds full replicas; each trains on its share of the
step's rounds (`--rounds` splits the global batch); optimizer tasks
allreduce gradients before updating. The allreduce makes DP
**structurally immune to cross-rank numeric differences** — every
rank lands the same sum, so there is exactly one trajectory.

### ZeRO-1 optimizer sharding (`--opt-shard zero1`)

A `zero1_halves` plan: weights stay replicated (`resident=ALL`),
optimizer state is partitioned by owner (field-snapped near-equal
buckets; single-matrix roots like embed/head split by rows). Each
optimizer task reduces every sharded gradient region to its owner,
the owner updates, and the updated params broadcast back. Per-rank
optimizer bytes halve (world 2); the loopback gates hold it
bitwise-equal to plain DP.

Field-snapped buckets lose balance as world approaches the per-root
field count (a llama3 block has ~7 large fields; at world 8 the
worst/best owned-bytes ratio degrades toward 0.2). For world > 2,
prefer `zero1rs`.

### Byte-equal ZeRO-1 (`--opt-shard zero1rs`)

The bandwidth-optimal formulation, and the one that scales to any
world size. Instead of per-field owner buckets, each eligible weight
object is treated as ONE flat array of `total` elements (alignment
gaps included), cut into `world` byte-equal slices plus a
`total % world` tail:

1. ONE `reduce_scatter` of the flattened gradient buffer — each rank
   receives the sum for its own slice (the tail, if any, is
   allreduced and updated redundantly on every rank);
2. the rank updates its slice with flat AdamW moments (`m`/`v` are
   sized to the slice, so per-rank `O_*` bytes are `~1/world`);
3. ONE `all_gather` re-assembles the full updated parameters
   everywhere.

Elementwise sums and updates over identically-reduced values: the
result is bitwise-identical to plain DP per weight field (the flat
update also rewrites the packing gaps — unread noise, identical
across ranks, only ever seen by a whole-buffer byte diff). Balance is
exact at every world size, and the two collectives touch each byte
once — no per-field round trips.

**Eligibility is per weight object, checked at plan time**
(`rs_eligible`): one flat update means ONE optimizer story for the
whole object — every field must resolve to `adamw` at that layer
(after `layer_overrides` routing), match no `hyper_overrides`
pattern (a per-field lr/wd would silently apply to the whole range),
and share uniform param/grad/opt dtypes with param == grad (that
byte-coincidence is what lets one geometry describe both W and dW).
Policy names are namespaced exactly as the update task sees them
(`embed.w`, `head.final_norm_w`, block fields bare). Objects that
fail any test silently keep the field-snapped `zero1` treatment —
the two modes compose per root. The update task re-verifies the
uniformity at run time and refuses to train if the policy drifted
from what the plan assumed.

### Tensor parallelism (`--tp-mlp`)

A `tp_mlp_shards` plan: the MLP fields shard over `d_ff` with
`resident == owner == rank` (w1/w3 by columns, w2 by rows);
everything else stays replicated with zero1-style owners (no
`ALL_RANKS` updates — replicated fields are re-pinned by the owner's
broadcast every step). TP splits *compute*, not data: every rank
runs the full batch; the forward allreduces each layer's MLP partial
product and the backward allreduces dx. Gradients for replicated
fields are *replicas*, not partial sums, so the optimizer runs in
replica mode (no reduce — a sum would double them).

**Hard requirement: arch-homogeneous ranks.** TP's correctness
depends on the replicated compute actually replicating. Different
GPU generations run different kernels (expect ~3-5e-4 nats/round
forward divergence at identical weights between generations), and TP
then sums a mixture of two model-variants every layer — training
degrades deterministically. DP and ZeRO-1 do not care (the grad
allreduce collapses ranks onto one trajectory); TP does. Valid TP
substrates are architecture-homogeneous: a matched pair or a
same-generation node, never a mixed pair.

## 4. Checkpointing

### What a snapshot is

The service has a first-class snapshot API: `snapshot(scope, dest,
ids=..., client_meta=...)` leases the selected store objects,
copies their bytes to `dest/payload.bin` on a dedicated writer
thread (training-adjacent verbs park rather than error while an
object is leased), and writes `manifest.json` last — atomically.
`restore_snapshot(path, overwrite=...)` recreates the objects and
returns the stored `client_meta`.

The engine is **stateless per step**: all trajectory state lives in
store objects (weights `W_*`, optimizer state `O_*`) plus the
driver-supplied step index. That is why a checkpoint record is small — the step counter, seed,
fleet layout, and artifact locations; config and program re-derive
from the invocation — and why restore + re-register + continue
reproduces training exactly.

### Fleet checkpoints

```
python tools/train/train.py train ... --topology topology.toml \
    --checkpoint-every 100 \
    --checkpoint-redundancy 2 \
    --checkpoint-keep-last 3 \
    --out results/pretrain/myrun.json
```

At every N-step boundary the conductor has each rank snapshot to a
**host-local** path (`results/pretrain/checkpoints/<run>/step_XXXXXX/`
on that rank's own disk — the run name is your `--out` stem), waits
for all writers (`save_checkpoint`), then writes `checkpoint_record.json` on the conductor **last**
(the full save/restore mechanics, lease protection, and usage
examples live in [checkpointing.md](checkpointing.md)).
That file is the completeness marker: a crash mid-snapshot leaves no
marker and the checkpoint is invisible to resume. It is the checkpoint record
(one format at every world size) and records: the
RESPONSIBILITY save plan, the relative per-rank artifact dirs, the
global data cursor, the loss curve so far, and the LAUNCH RECORD —
the literal argv, resolved settings, data identity, git and
torch/CUDA identity, per-rank host/device, and every rank's PLANNED
PROGRAM saved beside the artifacts (`programs/rankN.json`): a
checkpoint captures plan-time decisions, not just weights, and any
run can be re-invoked exactly from its record.

### The parallelism stack: scheme, sharding, layouts, groups

Parallelism is configured by ONE value object — the
**ParallelismScheme** (`distributed/parallelism.py`) — and executed
by a stack of layers, each consuming only the DATA of the layer
above:

```
ParallelismScheme        WHAT the parallel structure is: an ordered
  (the contract)         tuple of named AXES (name, size, role).
                         Axis names are comm PURPOSE keys ("dp",
                         "tp"; future "ep"/"pp"). world = product of
                         axis sizes; composition = more axes.
    │
    ▼
group annotation         The lowered (parallelism-blind) program
  (annotation pass)      gains {purpose: group} comm handles and
                         shard/tp task params, keyed by the axis
                         purposes — family code never involved.
    │
    ▼
sharding API             The GEOMETRY under an axis: ShardPlan (which
  (sharding.py)          fields split, where), per-rank views, flat
                         slice boundaries. Computed once; the scheme
                         CARRIES its plan as data.
    │
    ▼
layouts                  The coordinate system geometry compiles
  (layout registry +     onto: named per-(family, kind) field
   narrow_layouts)       tables; narrowing applies slice geometry;
                         every byte size follows.
    │
    ▼
responsibility           Who steps and therefore saves each slice —
  (responsibility.py)    derived from the same axes; becomes the
                         checkpoint's save plan.
```

The conductor (`dataflow_training/run/conductor.py`) takes the scheme as
input — `run(cfg, recipe, pipeline, steps, scheme=...)` — validates
it up front (inconsistent schemes refuse before any daemon
launches), and never hardcodes a parallelism name. `train.py`'s
flags compile to a scheme (`--rounds`/`--opt-shard` → a data axis;
`--tp-mlp` → a tensor axis carrying its ShardPlan; no flags →
`solo()`).

Composition (e.g. dp × tp meshes) is contract-ready — a multi-axis
scheme is well-formed data, and comm sub-groups derive per axis per
orthogonal coordinate — but the mesh group machinery is deliberately
unbuilt until a composed configuration first matters; `validate`
refuses multi-axis schemes loudly rather than half-running them.
A NEW parallelism adds an axis role plus its program-layer machinery
(an annotation-pass extension and a comm purpose), never a new
conductor entry point.

### Responsibility: who steps, saves

Every parameter slice has a RESPONSIBLE rank — it holds the slice's
optimizer state, performs its step, and saves its checkpoint bytes —
plus optional BACKUP ranks (multiplicity is recorded from day one;
fault-tolerant k-responsibility is the same dial later). "Sharding"
keeps meaning LAYOUT (how tensors split); "responsibility" is the
who-steps/who-saves vocabulary:

- **zero1rs (the DP default)**: parameter bytes partition at the
  optimizer's own flat-slice boundaries — who saves == who steps,
  and checkpoint IO balance is automatic. Each rank writes its param
  byte RANGES (slice snapshots) plus its own whole `O_*` shards.
- **co (diagnostic)**: every rank holds identical bytes (certified
  bitwise); one responsible rank per object, byte-balanced, the rest
  recorded as backups.
- **TP**: each rank's narrowed `W_*`/`O_*` objects are wholly its
  responsibility.
- **world 1**: rank 0, everything.

Restore replays every artifact a rank needs in order (its own last),
and the engine's native range compose reassembles complete objects.

Data objects (`tokens_*`, `targets_*`, `loss_*`) are never saved:
resume re-derives them from the deterministic feed position, which
is a pure function of the step index.

`--checkpoint-redundancy k` copies artifacts to k distinct hosts at
save time; backups are sourced first (they already hold the bytes).
`--checkpoint-keep-last K` prunes older complete checkpoints on
every host after each new one lands.

### Resume

```
python tools/train/train.py train ... --resume auto --out results/pretrain/myrun.json
```

`--resume auto` picks the newest *complete* checkpoint for the run
name (or pass a specific `step_XXXXXX` directory). The conductor
validates the record against the current invocation (world,
rounds, backend, seed — mismatches refuse loudly), pulls any
remote-written artifact from its recorded writer host and fans the
full set out to every rank, boots daemons, runs the
kernel warm-up, and only **then** restores (`overwrite=True`) — the
warm-up mutates state, and restoring afterwards makes it harmless.
Training continues from the checkpoint step; the loss curve is
stitched from the record so the saved run output stays continuous.

Resume is *in-place*: same world, same plan. Restoring into a
different world/plan requires a gathered full-state artifact (a
SavePlan configuration reserved for future migration tooling).

### What to expect numerically

Restore fidelity is exact: the first resumed step reproduces the
uninterrupted run's step **bitwise**. Beyond that, fleet runs have
inherent run-to-run nondeterminism (~1e-4 loss-level within a dozen
steps at 125M scale, from execution-order-sensitive kernels — e.g.
embedding scatter-add atomics), so two continuations of the same
checkpoint can drift within that envelope while remaining
statistically identical. Parity judgments therefore use EMA bands,
which sit far above this floor.

## 5. Verifying a setup

- `tests/fleet/test_checkpoint_drill.py` — single-daemon
  snapshot/kill/restore/resume, bitwise gate.
- The fleet drill sequence: run with checkpointing → kill the fleet
  mid-run → relaunch with `--resume auto` → compare the resumed curve
  against the run's own uninterrupted continuation.
- `tests/fleet/test_coll_dtypes_crossbox.py` — per-(backend, dtype)
  verified collectives on the real wire.
- The handshake runs automatically at every fleet launch.

## Fleet flags beyond the basics

`train.py train` composes the run from `topology.toml` plus
per-rank overrides: `--group` picks the topology group, `--rounds`
assigns each rank its share of the step's DATA, in round units (rank
order = member order; unequal shares = weighted data parallelism —
a rank's local grad-accum count is simply its share), `--fast-budget`/`--backing-budget` (comma per
rank) override the topology's per-rank device budgets and host
memory, `--backend {hostmem,nccl,auto}`
overrides the group backend, `--opt-shard` selects the optimizer
sharding mode (DEFAULT at world>1: `zero1rs` — each rank steps
params/n; `co` is the co-responsible DIAGNOSTIC lane), and
`--tp-mlp` switches to tensor-parallel MLPs through the sharding
API. the conductor's `attach=` argument (repeatable) attaches to pre-launched
daemons instead of launching them — the profiling rigs use this to
nsys-wrap each daemon themselves. `train.py compare` overlays the
loss curves of finished runs. Omitting the topology flags entirely
is ZERO-CONFIG SOLO: one local child daemon, the same machinery at
world 1.

### Profiling a fleet run

```
python tools/train/train.py train --preset l3_1b --steps 10 --topology topology.toml \
    --rounds 6,2 --profile --profile-start-before-step 5 \
    --profile-stop-after-step 8 --out results/pretrain/prof.json
```

`--profile` wraps every launched daemon in Nsight Systems with the
canonical trace set, brackets the requested step window on each rank
through its daemon's `profiler_control` verb (cudaProfilerApi
capture ranges, [benchmarking.md](benchmarking.md)), and fetches
the per-rank
`.nsys-rep` reports back to the conductor's log directory.

## 6. Bringing up a new machine (single node, N GPUs)

Day-1 ladder for a fresh multi-GPU box (e.g. an 8-GPU node). Each
rung is cheap and localizes its own failure class; don't skip rungs
on a new GPU architecture.

**Setup (once).**

```
git clone <repo> && cd dataflow
pip install -e .                       # same interpreter for daemons
ln -s /path/to/fineweb10B datasets/fineweb10B   # llm.c GPT-2 shards
cp topology.example.toml topology.toml # multi-GPU pattern is in the
                                       # example: one [hosts.*] per
                                       # GPU, device = 0..N-1,
                                       # distinct ports, one
                                       # [groups.node] backend "nccl"
```

Only the conductor's machine needs the dataset — token slices ride
the control plane to every rank.

**Rung 1 — re-bless the architecture (pure battery).** Kernels have
only been certified on the architectures we run; a new one must
re-pass the goldens and the kernel-audit battery before any fleet
work:

```
python -m pytest -q
```

**Rung 2 — single-GPU training sanity.**

```
python tools/train/train.py smoke --steps 20
```

(one engine, one GPU, real corpus tokens — no fleet machinery).

**Rung 3 — the fleet lane on this node.** Loopback gates boot real
daemons and exercise groups, sharding, and checkpointing end to end:

```
python -m pytest -m fleet -q \
    tests/fleet/test_zero1_loopback.py \
    tests/fleet/test_zero1rs_loopback.py \
    tests/fleet/test_checkpoint_drill.py
```

**Rung 4 — small fleet smoke, then scale the world.** Start at
world 2 (`--group` of two members), then the full node:

```
python tools/train/train.py train --preset l3_125m --steps 60 --topology topology.toml \
    --group node --backend nccl --rounds 1,1,1,1,1,1,1,1 \
    --out results/pretrain/node_smoke.json
```

`--rounds` must list one entry per member (each rank's share of the
step's grad-accum rounds; equal on a homogeneous node). The version
handshake, link probes, and warm-up dance run automatically.

**Rung 5 — the real run, sharded + checkpointed.**

```
python tools/train/train.py train --preset l3_1b --steps 1000 --topology topology.toml \
    --group node --backend nccl --rounds 1,1,1,1,1,1,1,1 \
    --opt-shard zero1rs \
    --checkpoint-every 100 --checkpoint-keep-last 3 \
    --out results/pretrain/l3_1b_node.json
```

Resume after any interruption with the same command plus
`--resume auto`. Compare the loss curve to a recorded single-box or
crossbox run of the same preset per section 4's numeric
expectations.

**Known limits on a new node** (state of the world; the fleet will
refuse or warn rather than corrupt):

- world > 2 requires `backend = "nccl"` — hostmem collectives are
  pairwise.
- NCCL env defaults in `daemons.py` were tuned for a socket fabric
  (`NCCL_IB_DISABLE=1` etc.). Harmless intra-node (NVLink/P2P
  ignores them), but a MULTI-node IB/RoCE cluster needs them moved
  to per-fabric topology settings first.
- Tensor parallelism (`--tp-mlp`) is valid on this substrate
  (matched GPUs) but is a correctness track, not a throughput one.
- Fleet mode requires `--rounds` explicitly (launches refuse without it); always pass it
  explicitly for other world sizes.
