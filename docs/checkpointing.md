# Checkpointing

Persistence is an engine capability first: the daemon exposes a
small snapshot / restore API, asynchronous but lease-protected, and
everything else — including distributed training checkpoints — is
composed on top of it. This page describes the engine API and its
concurrency contract, then shows usage, ending with how the
distributed training layer drives it.

## The engine API

Four verbs on `EngineClient`:

```python
out = client.snapshot(scope, dest, ids=None, ranges=None,
                      client_meta=None)     # -> {"snap_id": ...}
client.snapshot_status(snap_id)             # -> {"state", "bytes_done", ...}
client.wait_snapshot(snap_id, timeout=...)  # poll until done/error
client.restore_snapshot(path, overwrite=False)
```

- `scope` selects objects ("all" plus an explicit `ids=` list is the
  common form). `ranges={oid: (lo, hi)}` saves only that byte range
  of an object — the primitive that partitioned-responsibility
  saves use. `client_meta` is an arbitrary JSON dict that
  round-trips through the artifact (steps, ranks, tags).
- `dest` becomes one **artifact directory**: `manifest.json` (the
  object index — ids, sizes, ranges, payload offsets, your
  `client_meta`) plus `payload.bin` (raw bytes).

### Under the hood: asynchronous, lease-protected

`snapshot` does no copying on the calling path. Admission validates
the request, acquires a **read lease** on every object it will save
— taken last and exception-safe, so a rejected request can never
leak one — enqueues a copy job, and returns the `snap_id`
immediately. A dedicated writer thread then streams the leased
objects' bytes from host backing into `payload.bin`, writes
`manifest.json` to a temp file and renames it into place (a crashed
save never leaves a plausible-looking artifact), and finally
releases every lease, on success or failure alike.

The lease is the whole concurrency contract: while held, the saved
bytes are guaranteed stable — object extents cannot move and no
writer can touch them.

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

**Client side (explicit — for artifact consumers).** The `snap_id`
is the handle: poll `snapshot_status` or block in `wait_snapshot`.
Waiting is only required before *reading the artifact* (or declaring
a checkpoint complete) — never for correctness of subsequent
training, which the leases already guarantee.

### Restore

`restore_snapshot(path, overwrite=True)` reads one artifact and puts
its objects back: full entries overwrite (or create) the object;
**ranged** entries fill just their byte range in place, creating a
full-size object first if none exists. `client_meta` comes back in
the result, so a restorer can verify what it loaded.

## Example usage

Single engine, save and restore:

```python
out = client.snapshot("all", "/ckpt/step_000420/rank0",
                      ids=["W_0", "O_0"],
                      client_meta={"step": 420, "rank": 0})
client.wait_snapshot(out["snap_id"], timeout=600.0)   # artifact ready
...
res = client.restore_snapshot("/ckpt/step_000420/rank0",
                              overwrite=True)
assert res["client_meta"]["step"] == 420
```

A ranged save (only the first MiB of `W_0`, plus all of `O_0`):

```python
client.snapshot("all", dest, ids=["O_0", "W_0"],
                ranges={"W_0": (0, 1 << 20)},
                client_meta={"rank": 0})
```

## The distributed training composition

Training checkpoints are exactly this API, driven per rank. One
directory per saved step:

```
checkpoints/<run_name>/step_000420/
  rank0/  rank1/          one engine artifact per rank
  programs/rankN.json     the exact lowered program each rank ran
  checkpoint_record.json  the record (format 2) — written LAST
```

**Saving.** Under partitioned optimizer responsibility each rank
saves the parameter byte ranges it is responsible for plus its own
optimizer shard; the record's `save_plan` is that map. Minimal form
of what the training layer does at a step boundary:

```python
ids, ranges = rank_save_args(save_plan, rank, own_objects=["O_0"])
out = client.snapshot("all", f"{step_dir}/rank{rank}",
                      ids=ids, ranges=ranges,
                      client_meta={"step": step, "rank": rank})
# ... all ranks' copies overlap each other; then:
client.wait_snapshot(out["snap_id"], timeout=600.0)
write_record(step_dir, step=step, save_plan=save_plan,
             artifacts=["rank0", "rank1"], ...)   # completeness marker
```

`checkpoint_record.json` is written only after every rank reports
done, so its presence means the checkpoint is whole; it also carries
the seed, world, data cursor, loss history, and a full launch record
(argv, git and torch/cuda identity, per-rank host/device, program
paths). Readers open it first — `read_record` refuses unknown
formats loudly.

**Restoring.** Each rank replays every artifact the record lists,
**its own last**, so its own optimizer shard and ranges win any
overlap; complete objects reassemble bitwise from the slices:

```python
record = read_record(step_dir)
for artifact in artifacts_for_restore(record, rank):
    client.restore_snapshot(str(step_dir / artifact), overwrite=True)
```

Cross-box runs add one move: artifacts stay on the box that wrote
them until resume, when the conductor pulls each writer's artifacts
and fans the completed step directory out to every member.

**From the training tool** all of this is two flags:

```bash
python tools/train/train.py train --preset l3_125m \
  --checkpoint-every 50 --run-name mine          # save
python tools/train/train.py train --preset l3_125m \
  --checkpoint-every 50 --run-name mine --resume auto   # resume
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

1. **Duplicate-then-snapshot** (preferred first). Use the engine's
   `duplicate_object_group` verb: fast on-device copy of the
   persistent group under a tag, release the live objects
   immediately, snapshot the background copy while training
   proceeds. Shrinks the stall to the device-copy time regardless of
   payload IO, needs no engine dispatch changes, and the verb
   already exists — the training layer just has to wire
   duplicate -> snapshot-the-copy -> drop-the-copy into its
   step-boundary save.

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
  (`tests/dataflow_training/training/test_checkpoint_record.py`)
- end-to-end resume drills — single box, same-box world 2 with
  partitioned saves, cross-box with artifact redistribution — each
  asserting the resumed tail reproduces the uninterrupted run
  (`tests/fleet/test_world1_resume_drill.py`,
  `test_world2_resume_drill.py`, `test_crossbox_resume_drill.py`)
