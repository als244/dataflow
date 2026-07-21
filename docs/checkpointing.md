# Checkpointing

How training state is saved and restored: the engine's snapshot /
restore verbs, the lease protection that makes overlapped saving
safe, the sliced (ranged) saves that responsibility partitioning
produces, and the training-run layer that assembles per-rank
artifacts into one resumable checkpoint.

## What a checkpoint is on disk

One directory per saved step:

```
checkpoints/<run_name>/step_000420/
  rank0/                  engine snapshot artifact (one per rank)
    manifest.json         artifact index: objects, ranges, offsets
    payload.bin           raw object bytes
  rank1/ ...
  programs/rank0.json     the exact lowered program each rank ran
  checkpoint_record.json  the record (format 2) — written LAST
```

`checkpoint_record.json` is the completeness marker: it is written
only after every rank's snapshot reports done, so its presence means
the checkpoint is whole. Readers must open it first (`read_record`
refuses unknown formats loudly). It carries the step, seed, world,
data cursor, loss history, the responsibility **save plan** (which
rank saved which byte range of which object), the artifact list, and
a full launch record (argv, resolved settings, git identity,
torch/cuda versions, per-rank host/device, program paths) — enough
to re-invoke or audit the run without guessing.

## Engine layer: snapshot / restore

`snapshot(scope, dest, ids=..., ranges=..., client_meta=...)` is
**asynchronous**: the daemon validates the request, acquires leases
(see below), enqueues a copy job, and returns a `snap_id`
immediately. A dedicated writer thread then copies the object bytes
from host backing into `payload.bin` and writes `manifest.json` via
a temp-file rename, so a crashed save never leaves a
plausible-looking artifact. Poll with `snapshot_status(snap_id)` or
block with `wait_snapshot`.

```python
out = client.snapshot("all", "/ckpt/step_000420/rank0",
                      ids=["W_0", "O_0"],
                      ranges={"W_0": (0, 1 << 20)},   # byte range
                      client_meta={"step": 420, "rank": 0})
client.wait_snapshot(out["snap_id"], timeout=600.0)
```

`restore_snapshot(path, overwrite=True)` reads an artifact and puts
its objects back: a full entry overwrites (or creates) the object; a
**ranged** entry fills just that byte range in place, creating a
full-size object first if none exists. `client_meta` round-trips, so
a restorer can verify the step it loaded.

## Lease protection: why overlapped saving is safe

The payload copy reads object bytes directly from host backing, so
those bytes must not move or change mid-copy. The snapshot admission
therefore takes a **read lease** on every object it will copy —
acquired last and exception-safe, so a rejected request can never
leak a lease — and the writer thread releases all of them when the
job finishes, success or failure.

While an object is leased, any verb that would disturb it — a
`put_object`, a `release_object`, or an entire **run** whose program
binds it (runs are checked object-by-object before any state
mutates) — is not failed but **parked**: the dispatcher holds the
call and automatically retries it when the leases release. The
client sees only latency, never an error. Timeline for a step
boundary:

```
step N compute ──────────┐
                         ├─ snapshot admitted, leases W_/O_
payload copy (writer) ───┼────────────────┐
step N+1 run submitted ──┤ PARKED (leased) │
                         │                 ├─ leases released
                         │                 └─ step N+1 unparks, runs
```

So correctness never depends on the caller's timing: the next step
cannot read or write half-saved state, by construction. The current
cost is that a *conflicting* next step stalls for the payload-copy
window; work that touches no leased object proceeds concurrently.
The engine also exposes `duplicate_object_group` (an on-device copy
of a whole object group under a tag), which supports a future
stall-free pattern — copy the persistent state to a background group
quickly, release the live objects, and snapshot the copy while
training proceeds. The training layer does not use it yet.

## Sliced saves and reassembling restores

Under partitioned optimizer responsibility, each rank saves only the
parameter byte ranges it is responsible for, plus its own optimizer
shard — that is what the `ranges=` argument and the record's
`save_plan` express. Restore reverses it: each rank replays, in
order, every artifact the record lists for the checkpoint —
**its own artifact last**, so its own optimizer shard and ranges win
any overlap — and complete objects reassemble bitwise from the
slices:

```python
record = read_record(step_dir)
for artifact in artifacts_for_restore(record, rank):
    client.restore_snapshot(str(step_dir / artifact), overwrite=True)
```

Cross-box runs add one move: artifacts live on the box that wrote
them until resume, when the conductor pulls each writer's artifacts
and fans the full step directory out to every member.

## Training-run usage

```bash
# save every 50 steps; resume from the newest complete checkpoint
python tools/train/train.py train --preset l3_125m \
  --checkpoint-every 50 --run-name mine
python tools/train/train.py train --preset l3_125m \
  --checkpoint-every 50 --run-name mine --resume auto
```

`--resume` accepts `auto` (newest directory containing a
`checkpoint_record.json`) or an explicit step directory. Resume
validates the record against the invocation (world, seed, preset)
and refuses mismatches; losses before the checkpoint ride the record
so the saved curve stays continuous; the data pipeline restarts from
the recorded cursor.

## What certifies this

- lease behavior: parked writers wake on release, snapshots see a
  stable image (`tests/dataflow/service/test_service_snapshot.py`)
- ranged saves and slice round-trips
  (`tests/dataflow/service/test_slice_snapshots.py`)
- record format, own-artifact-last reassembly, completeness marker
  (`tests/dataflow_training/training/test_checkpoint_record.py`)
- end-to-end resume drills — single box, same-box world 2 with
  partitioned saves, and cross-box with artifact redistribution —
  each asserting the resumed tail reproduces the uninterrupted run
  (`tests/fleet/test_world1_resume_drill.py`,
  `test_world2_resume_drill.py`, `test_crossbox_resume_drill.py`)

Older pre-record checkpoint layouts are not readable by this
tooling; there is deliberately no converter.
