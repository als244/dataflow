# Checkpointing

Persistence is an engine capability first: the daemon exposes a
small snapshot / restore API, asynchronous but lease-protected, and
everything else — including distributed training checkpoints — is
composed on top of it. This page describes the engine API and its
concurrency contract, then shows usage, ending with how the
distributed training layer drives it.

## The engine API

The verbs on `EngineClient`:

```python
out = client.snapshot(dest, slices=None,
                      client_meta=None)     # -> {"snap_id": ...}
client.snapshot_status(snap_id)             # -> {"state", "bytes_done", ...}
client.wait_snapshot(snap_id, timeout=...)  # poll until done/error
client.restore_snapshot(path, overwrite=False, verify=True, remap=None)
    # blocking by default; block=False -> {"restore_id": ...}
client.restore_status(restore_id)
client.wait_restore(restore_id, timeout=...)
```

- `slices=None` saves every resident object whole. An explicit
  `slices` list selects exactly: each slice names a stored object
  `id` and may map a `src` byte range into a logical object
  (`logical_id`, `dst`, `logical_bytes` — all defaulting to the
  identity mapping) — the primitive partitioned-responsibility
  saves use. `client_meta` is an arbitrary JSON dict that
  round-trips through the snapshot (steps, ranks, tags).
- `dest` becomes one **snapshot directory**: `snapshot.json` (schema
  `dataflow-snapshot/v1` — the slice index with sizes, mappings,
  payload offsets, one blake2b-16 hash per slice, and your
  `client_meta`) plus `payload.bin` (raw bytes).

### Under the hood: asynchronous, lease-protected

`snapshot` does no copying on the calling path. Admission validates
the request, acquires a **read lease** on every object it will save
— taken last and exception-safe, so a rejected request can never
leak one — enqueues a copy job, and returns the `snap_id`
immediately. A dedicated writer thread then streams the leased
objects' bytes from host backing into `payload.bin` (hashing each
slice as it streams), writes
`snapshot.json` to a temp file and renames it into place (a crashed
save never leaves a plausible-looking snapshot), and finally
releases every lease, on success or failure alike.

The lease is the whole concurrency contract: while held, the saved
bytes are guaranteed stable — object extents cannot move and no
writer can touch them.

### Residency contract: snapshots read host backing, always

A store resident's canonical home is its **pinned host-backing
extent** — that is what "resident" means in the service model; there
is no fast-only persistent object. Device (fast) copies are
per-run transients: a run uploads what it needs from backing and
offloads mutated persistent state back to backing as part of the
run itself. Snapshot admission then happens on the same single
dispatcher that runs execute on, so between runs the backing bytes
ARE the post-step state — the payload writer copies straight from
the backing extents and never touches the device; no
device-to-host staging path exists in the snapshot machinery
because none is needed. Snapshotting a non-resident id fails
validation loudly.

### How waiting works

**Engine side (implicit — callers cannot get this wrong).** Any verb
that would disturb a leased object — `put_object`, a release, or an
entire **run** whose program binds one (runs are checked
object-by-object before any state mutates) — is not rejected but
**parked**: the dispatcher holds the call and retries it
automatically when the leases release. The client of that verb
observes latency, never an error. At a training step boundary this
means the next step may be submitted immediately; it simply does not
execute until the save's payload copy is off the state it needs:

```
step N compute ──────────┐
                         ├─ snapshot admitted, W_/O_ leased
payload copy (writer) ───┼────────────────┐
step N+1 run submitted ──┤ PARKED (leased) │
                         │                 ├─ leases released
                         │                 └─ step N+1 unparks, runs
```

Work that touches no leased object proceeds concurrently with the
copy.

**Ordering against runs (both directions, no caller effort).**
A run is complete only when the engine's end-of-run drain has
consumed a completion token for EVERY piece of in-flight device
work — compute tasks and host/device transfer jobs alike, each
tracked by its own event — so a tail of device-to-host offloads on
the transfer stream is waited for exactly like compute. The drain
then refuses to return if any transfer is still queued (a loud
deadlock error, never a silent drop) and verifies every persistent
object landed at its planned final location before the service
declares the run done.
Runs occupy the dispatcher end-to-end, so a snapshot submitted while
a program is running is admitted only after that run — including its
final-state offload to backing — completes: `run` then `snapshot` in
submission order always saves the post-run state, with no explicit
synchronization by the caller. The reverse direction is the lease
park described above: a run submitted during an in-flight save
executes only after the copy releases its objects. Ordering is
dispatcher submission order; leases close the one remaining window
(the asynchronous payload copy).

**Client side (explicit — for artifact consumers).** The `snap_id`
is the handle: poll `snapshot_status` or block in `wait_snapshot`.
Waiting is only required before *reading the artifact* (or declaring
a checkpoint complete) — never for correctness of subsequent
training, which the leases already guarantee.

### Restore

`restore_snapshot(path, overwrite=True)` reads one snapshot and puts
its slices back in three passes — validate every placement, verify
every slice hash (`verify=False` opts out), then place — so any
refusal leaves the store untouched. Like snapshot, the payload work
(verify + placement) runs on the writer thread behind leases on
every target — a concurrent writer touching a target parks until
the restore completes — while admission (validation, creation of
absent targets, lease acquisition) stays on the dispatcher, so a
large restore never freezes the daemon's other verbs. The blocking
call returns the finished result; `block=False` returns a
`restore_id` for `restore_status` / `wait_restore`. Identity slices overwrite (or
create) their object whole; mapped slices land at `dst` in their
logical object, creating it at `logical_bytes` if absent; an
optional `remap` plan extracts logical ranges into local objects
instead. `client_meta` comes back in
the result, so a restorer can verify what it loaded.

## Example usage

Single engine, save and restore:

```python
out = client.snapshot("/ckpt/step_000420/rank0",
                      slices=[{"id": "W_0"}, {"id": "O_0"}],
                      client_meta={"step": 420, "rank": 0})
client.wait_snapshot(out["snap_id"], timeout=600.0)   # snapshot ready
...
res = client.restore_snapshot("/ckpt/step_000420/rank0",
                              overwrite=True)
assert res["client_meta"]["step"] == 420
```

A sliced save (only the first MiB of `W_0`, plus all of `O_0`):

```python
client.snapshot(dest,
                slices=[{"id": "O_0"},
                        {"id": "W_0", "src": [0, 1 << 20]}],
                client_meta={"rank": 0})
```

## The checkpoint record schema

`checkpoint_record.json` (`read_record` refuses foreign schemas
loudly) is the one cross-writer file; everything else in a step
directory is engine snapshots and program dumps. Annotated:

```jsonc
{
  "schema": "dataflow-checkpoint/v1",
  "step": 420,                  // the step this state follows
  "seed": 11,
  "scheme": {...},              // OPAQUE caller scheme; training stores
                                //   world, responsibility, source_policy
  "logical_objects": {          // the record's namespace: named byte
    "W_0": {"bytes": 2113024}   //   spans (+ field-schema digests)
  },
  "snapshots": [                // one engine snapshot per writer,
    {"path": "rank0",           //   with its resident-object sizes:
     "writer": "0",             //   shard layouts carry alignment
     "objects": {"W_0": 2113024, "O_0": 6439424}},  // padding, so the
    {"path": "rank1", "writer": "1",  // rank view recreates objects
     "objects": {"W_0": 2113024, "O_0": 6439424}}   // at EXACT local
  ],                            //   geometry, never a derived guess
  "slices": {                   // snapshot bytes -> logical bytes,
    "W_0": [                    //   hashed, authoritative-flagged
      {"snapshot": 0, "snapshot_range": [0, 2113024],
       "object_range": [0, 2113024], "hash": "...",
       "authoritative": true},
      {"snapshot": 1, "snapshot_range": [0, 2113024],
       "object_range": [0, 2113024], "hash": "..."}
    ]
  },
  "engine_spec": {"0": {"backing_gib": 87.0}, ...},
  "client_payload": {...},      // OPAQUE; training: losses, data cursor
  "launch": {
    "argv": [...],              // exact invocation
    "resolved": {"preset": ..., "seed": ..., "opt_shard": ...,
                  "world": ..., "rank_rounds": ..., "backend": ...},
    "data": {...},              // pipeline description
    "git": "...",
    "env": {"torch": ..., "cuda": ...},
    "ranks": [{"host": "chicago", "device": 0}, ...],
    "programs": ["programs/rank0.json", ...]   // exact lowered programs
  }
}
```

The record is validated totally at write and at read: slices must
union to each logical object's full span, overlapping bytes need an
authoritative winner, identical-span copies must hash-equal (the
replication drift certificate), and slice endpoints must land on
element boundaries when field schemas are given. `launch` makes a
checkpoint auditable and re-invocable without guessing; the client
payload makes the resumed curve continuous.

## Resuming

One flag, at any world size:

```bash
python tools/train/train.py train --preset l3_125m \
  --checkpoint-every 50 --out results/pretrain/mine --resume auto
```

`--resume auto` picks the newest step directory containing a
`checkpoint_record.json`; an explicit step-directory path pins one.
Every dynamic quantity a continued run needs is reconstructed from
the record, not guessed: the loop restarts at the recorded step, so
the per-step `step` run argument — which drives the learning-rate
schedule and the optimizer's bias-correction term as pure functions
of the step index — continues exactly as an uninterrupted run would;
the data pipeline resumes from `data_cursor`; prior `losses` ride
the record so the saved curve stays continuous; and the invocation
is validated against `launch.resolved` (world, seed, preset), with
mismatches refused rather than silently retrained.

Certification: the resume drills train with checkpoints, resume on
FRESH daemons, and compare the resumed tail against the
uninterrupted run's losses. The same-box world-2 drill demands the
tails EQUAL BIT FOR BIT — exact equality is the claim that the
record captures every byte the tail depends on — while the
cross-box drills allow a tight ambient envelope (5e-4 worst-step),
all on top of the bitwise slice-reassembly gates.

## Single GPU is the world-1 special case

There is deliberately one checkpoint format at every world size.
A single-GPU run writes the same step directory with `world: 1`:
one `rank0/` snapshot whose slices cover every object whole, one
program dump — and the same `read_record` / `load_checkpoint` path
reads it back. Nothing about resume, validation, or the completeness
marker is distributed-specific; the distributed composition below is
the general case this degenerates from.

## The distributed training composition

Training checkpoints are exactly this API, driven per rank. One
directory per saved step:

```
checkpoints/<run_name>/step_000420/
  rank0/  rank1/          one engine snapshot per rank
  programs/rankN.json     the exact lowered program each rank ran
  checkpoint_record.json  the checkpoint record — written LAST
```

**Saving** (`save_checkpoint`, driven by the conductor at each
checkpoint boundary)**.** The configured source policy compiles the
responsibility map into per-writer save sets — `simple` (the
default) saves everything each writer holds, whole, for
self-sufficient per-rank snapshots with writer 0 authoritative on
replicated objects; `dedup` saves one copy of each replicated object
as disjoint responsibility slices — and the composer fans out one
snapshot per writer, waits for all of them, collects the per-slice
hashes each daemon streamed, and runs the replication-drift
certificate (identical-span copies must hash-equal; disagreement
refuses the checkpoint naming both writers). State that each rank
accumulates privately — per-rank counters no step synchronizes —
saves under writer-qualified logical ids (`Aux_0@1`) instead of
being falsely certified as replicated; resume returns each rank its
own copy. Minimal form of what
the training layer does at a step boundary:

```python
logical, per_writer = compile_source_policy(
    policy="simple", world=world, writer_specs=writer_specs,
    plan=responsibility, opt_slices=opt_slices)
save_checkpoint(writers, step_dir, step=step, seed=seed,
                logical_objects=logical, scheme=scheme,
                client_payload={"losses": losses})   # record LAST
```

`checkpoint_record.json` is written only after every writer reports
done and the drift certificate passes, so its presence means the
checkpoint is whole and certified; it also carries the seed, the
opaque scheme and client payload (data cursor, loss history), and a
full launch record (argv, git and torch/cuda identity, per-rank
host/device, program paths). Readers open it first — `read_record`
refuses foreign schemas loudly.

**Restoring.** Targets resolve against the record into per-snapshot
restore plans in the engine's remap shape — every requested byte
sourced exactly once, authoritative slices preferred where coverage
overlaps:

```python
record, client = load_checkpoint(step_dir, targets=["W_0"])  # weights
record, client = load_checkpoint(step_dir, targets="all")    # logical
record, client = load_checkpoint(step_dir,
                                 targets={"1": ["W_0", "O_0"]})  # rank
```

A weights-only load simply targets the parameter objects — optimizer
bytes never enter the store, so there is nothing to release. The
logical view (`"all"`, or bare id lists) reassembles complete
objects from every writer's slices; a writer key restores that
rank's own view, with logical slices remapped into its local shard
geometry. Resume under the simple policy is the keyed form against
the rank's own snapshot — self-sufficient, no cross-writer bytes.
The checkpoint evaluation tool is exactly the weights-only helper
plus a forward pass.

Cross-box runs add one move: snapshots stay on the box that wrote
them until a resume needs foreign slices (a remapped topology, or a
logical load of another box's shards), when the conductor pulls and
fans them out.

**From the training tool** all of this is two flags:

```bash
python tools/train/train.py train --preset l3_125m \
  --checkpoint-every 50 --out results/pretrain/mine          # save
python tools/train/train.py train --preset l3_125m \
  --checkpoint-every 50 --out results/pretrain/mine --resume auto   # resume
```

`--resume` takes `auto` (newest directory containing a
`checkpoint_record.json`) or an explicit step directory; resume
validates the record against the invocation (world, seed, preset)
and refuses mismatches, pre-checkpoint losses ride the record so the
saved curve stays continuous, and the data pipeline restarts from
the recorded cursor.

## Current limits

A *conflicting* next step stalls for the payload-copy window —
leases guarantee safety, not stall-freedom. The stall is
run-granular by design: runs are admitted atomically (the bind
pre-pass declares mutation intent for every bound persistent object
before anything executes), and the engine has no mid-run wait
primitive, so the only safe enforcement point is before the run
starts — even though only the optimizer tail actually dirties the
saved objects. Older pre-record checkpoint layouts are not readable
by this tooling; there is deliberately no converter.

## Future improvements

Two optimizations for the stall window, in recommended order:

1. **Duplicate-then-snapshot** (preferred first). Duplicate each
   persistent object (`duplicate_object` — a fast on-device copy),
   let training continue immediately, and snapshot the copies under
   their real identities via slice mapping
   (`slices=[{"id": oid + "@ck", "logical_id": oid}, ...]`), then
   release the copies. Shrinks the stall to the device-copy time
   regardless of payload IO and needs no engine dispatch changes —
   the training layer wires duplicate -> snapshot-the-copies ->
   release into its step-boundary save.

2. **Task-granular lease waits.** Leases are READ leases, so the
   forward/backward — which only reads the weights — could legally
   overlap the payload copy, with just the mutating optimizer tasks
   waiting at their dispatch point (task.mutates intersecting the
   leased set -> wait on release). Because the dispatcher is
   single-threaded, a snapshot can never be admitted mid-run, so
   leases only ever pre-exist a run and the wait needs no re-park
   machinery — a condition wait at engine task dispatch suffices.
   Hides most of the copy behind step compute; worth doing only if
   the duplicate-then-snapshot stall ever still matters.

## What certifies this

- lease behavior — parked writers wake on release, snapshots see a
  stable image (`tests/dataflow/service/test_service_snapshot.py`)
- ranged saves and slice round-trips
  (`tests/dataflow/service/test_slice_snapshots.py`)
- record format, own-artifact-last reassembly, completeness marker
  (`tests/dataflow_training/training/surfaces/test_checkpoint_record.py`)
- end-to-end resume drills — single box, same-box world 2 with
  partitioned saves, cross-box with artifact redistribution — each
  asserting the resumed tail reproduces the uninterrupted run
  (`tests/fleet/checkpoint_resume/test_world1_resume_drill.py`,
  `test_world2_resume_drill.py`, `test_crossbox_resume_drill.py`)
