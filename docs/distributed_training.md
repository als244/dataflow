# Distributed training: groups, sharding, and checkpointing

This guide explains how multi-daemon training works end to end: the
communication layer, the sharding API that expresses *who owns what*,
the parallelism configurations built on it, and how checkpointing and
resume behave for each. Everything here is driven from
`tools/train_fleet.py` and configured by `topology.toml` — no machine
facts live in code.

## 1. The pieces

**Daemons and the conductor.** Every GPU runs one `dataflowd` daemon.
A *conductor* (the driver process, `train_fleet.py`) launches
daemons over the hosts in `topology.toml`, registers a per-rank
program with each, feeds token rounds, and drives lockstep steps.
Daemons never talk to the conductor's Python state — everything
crosses the wire as programs, objects, and run calls.

**Groups and backends.** Collectives run over a *peer group* created
by the conductor. The group's backend comes from the topology
(`backend = "auto"` resolves to **nccl** on any real boot; the
hostmem lane remains for CI/loopback). Before any fleet run the
conductor performs a **handshake**: every member must be on the same
repo commit *and* the same torch/cuda/cudnn versions, and the
conductor itself must have no uncommitted tracked changes. This is
enforced because the failure modes of skew are silent (mixed-version
collectives corrupt quietly; version-skewed kernels break replicated
compute — see §3, tensor parallelism).

## 2. The sharding API (`dataflow/pretrain/sharding.py`)

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
python tools/train_fleet.py train --preset l3_1b --steps 1000 \
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
optimizer bytes halve (world 2); certified bitwise-equal to plain DP
and at the 1000-step horizon.

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
GPU generations run different kernels (measured on this fleet:
3–5e-4 nats/round forward divergence at identical weights between
sm86 and sm120), and TP then sums a mixture of two model-variants
every layer — training degrades deterministically. DP and ZeRO-1 do
not care (the grad allreduce collapses ranks onto one trajectory);
TP does. The certification run used two ranks on one GPU; a matched
pair or an H100 node is a valid substrate, a mixed pair is not.

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
driver-supplied step index. That is why a resume manifest is tiny —
`{step, seed, cfg, prog_id}` — and why restore + re-register +
continue reproduces training exactly.

### Fleet checkpoints

```
python tools/train_fleet.py train ... \
    --checkpoint-every 100 \
    --checkpoint-redundancy 2 \
    --checkpoint-keep-last 3 \
    --out results/pretrain/myrun.json
```

At every N-step boundary the conductor has each rank snapshot to a
**host-local** path (`results/pretrain/checkpoints/<run>/step_XXXXXX/`
on that rank's own disk — the run name is your `--out` stem), waits
for all writers, then writes `fleet.json` on the conductor **last**.
That file is the completeness marker: a crash mid-snapshot leaves no
marker and the checkpoint is invisible to resume. It records the
fleet layout (hosts, rounds, budgets, backend, seed), the save plan,
artifact locations, and the loss curve so far.

### The SavePlan: who saves what

Saving is deduplicated by the same ownership algebra that defines
the parallelism — derived automatically from the run's plan:

| configuration | shared artifact (written once by rank 0) | per-rank artifacts |
|---|---|---|
| plain DP  | `W_*` and `O_*` (fully replicated) | — (other ranks write nothing) |
| ZeRO-1 (`zero1` or `zero1rs`) | `W_*` (replicated weights) | each rank's owned `O_*` shards |
| TP        | —                                  | each rank's `W_*` + `O_*` shards |

Data objects (`tokens_*`, `targets_*`, `loss_*`) are never saved:
resume re-derives them from the deterministic stream position, which
is a pure function of the step index.

`--checkpoint-redundancy k` additionally copies the shared artifact
to k distinct hosts at save time; if the primary disk is lost,
resume recovers from any surviving copy automatically.
`--checkpoint-keep-last K` prunes older complete checkpoints on
every host after each new one lands.

### Resume

```
python tools/train_fleet.py train ... --resume auto --out results/pretrain/myrun.json
```

`--resume auto` picks the newest *complete* checkpoint for the run
name (or pass a specific `step_XXXXXX` directory). The conductor
validates the manifest against the current invocation (world,
rounds, backend, seed — mismatches refuse loudly), distributes the
shared artifact to any host that lacks it, boots daemons, runs the
kernel warm-up, and only **then** restores (`overwrite=True`) — the
warm-up mutates state, and restoring afterwards makes it harmless.
Training continues from the checkpoint step; the loss curve is
stitched from the manifest so the saved run output stays continuous.

Resume is *in-place*: same world, same plan. Restoring into a
different world/plan requires a gathered full-state artifact (a
SavePlan configuration reserved for future migration tooling).

### What to expect numerically

Restore fidelity is exact: the first resumed step reproduces the
uninterrupted run's step **bitwise**. Beyond that, fleet runs have
inherent run-to-run nondeterminism (~1e-4 loss-level by step 12 at
125M, from execution-order-sensitive kernels — e.g. embedding
scatter-add atomics), so two continuations of the same checkpoint
can drift within that envelope while remaining statistically
identical. All certifications use EMA bands, which are far above
this floor.

## 5. Verifying a setup

- `tests/fleet/test_checkpoint_drill.py` — single-daemon
  snapshot/kill/restore/resume, bitwise gate.
- The fleet drill sequence (see the CK entries in the findings
  ledger): checkpointed run → kill → `--resume auto` → compare
  against the run's own continuation.
- `tests/fleet/test_coll_dtypes_crossbox.py` — per-(backend, dtype)
  verified collectives on the real wire.
- The handshake runs automatically at every fleet launch.

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
python tools/train_solo.py smoke --steps 20
```

(one engine, one GPU, real fineweb tokens — no fleet machinery).

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
python tools/train_fleet.py train --preset l3_125m --steps 60 \
    --group node --backend nccl --rounds 1,1,1,1,1,1,1,1 \
    --out results/pretrain/node_smoke.json
```

`--rounds` must list one entry per member (the per-rank split of the
step's grad-accum rounds; equal on a homogeneous node). The version
handshake, link probes, and warm-up dance run automatically.

**Rung 5 — the real run, sharded + checkpointed.**

```
python tools/train_fleet.py train --preset l3_1b --steps 1000 \
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
- NCCL env defaults in `hostops.py` were tuned for a socket fabric
  (`NCCL_IB_DISABLE=1` etc.). Harmless intra-node (NVLink/P2P
  ignores them), but a MULTI-node IB/RoCE cluster needs them moved
  to per-fabric topology settings first.
- Tensor parallelism (`--tp-mlp`) is valid on this substrate
  (matched GPUs) but is a correctness track, not a throughput one.
- The `--rounds` default is a 2-rank split; always pass it
  explicitly for other world sizes.
