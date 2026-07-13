# The dataflow engine service

A persistent daemon that owns pinned host memory (the **store
slab**), holds named objects (weights, optimizer state, data,
losses) as **residents**, and executes registered dataflow
**programs** against them. Clients connect over a unix socket,
put/fetch objects, register programs once, and run them many times
— training state lives in the store between runs, so a training
job is "run the step program N times", each step microseconds of
control overhead away from the in-process engine (measured: tok/s
within ±1%, device/host peaks at parity, llama3-8B 1K×64).

Start it:

```bash
python tools/dataflowd.py start --socket /tmp/dfd.sock --backing-gib 145
python tools/dataflowd.py status
python tools/dataflowd.py stop
```

`--backing-gib` is the ONE pinned budget: residents AND run
transients (gradients, saved activations) draw from the same slab.
Size it to `residents + worst-plan transients` (the plan's demand
bound); the daemon refuses to pin into the system's last 24 GiB.

## Client in five verbs

```python
from dataflow.service import EngineClient

with EngineClient("/tmp/dfd.sock", client_name="driver") as c:
    # 1. state into the store (materialize = server-side seeded init)
    c.materialize_group({"kind": "family_init_all", "family": "llama3",
                         "cfg": cfg_dict, "seed": 11})
    c.put_object("tokens_0_0", token_bytes)     # data chunks

    # 2. register once (content-hashed id; placement cached)
    reg = c.register_program(plan_path,
                             resolver={"family": "llama3", "cfg": cfg_dict})

    # 3. run many (args reach tasks via run_args: step, lr, ...)
    for k in range(steps):
        c.put_object(f"tokens_{k+1}", next_chunk)          # overlap ok
        r = c.run(reg["prog_id"], args={"step": k},
                  rebind={"tokens_0_0": f"tokens_{k}"},    # per-step data
                  fetch=["loss_0_0"])
        print(k, r["fetched"]["loss_0_0"], r["makespan_us"])

    # 4. checkpoint (leases keep the set stable; copy runs in background)
    c.duplicate_object_group("model_state", tag="ck")
    s = c.snapshot("all", "/ckpts/step100",
                   client_meta={"step": 100, "cursor": [3, 128]})
    c.wait_snapshot(s["snap_id"])

    # 5. resume later (client_meta comes back in the same call)
    meta = c.restore_snapshot("/ckpts/step100")["client_meta"]
```

## The model in one paragraph

Objects are engine-global and flat-namespaced: any client sees
`W_3`. A program's **initial objects** bind to residents at run
start (strict size match); whatever the program's
`final_locations` declares comes OUT resident (losses); everything
else the run creates (gradients, activation staging) is a
**transient** — named in the program, never in the catalog, carved
lazily from the same slab, recycled across steps, returned at
`unregister_program`. `rebind` points a program input id at a
different resident per run (streaming data). Each run also snapshots
the daemon's live peer-group table: tasks that declare `comm_groups`
resolve their group by NAME at that moment, and run standalone when
it isn't there (distributed_training.md). Runs execute FIFO on
one dispatcher; status/query verbs answer instantly from a fast
path; `cancel_run` takes effect at the next task boundary; a
failed run poisons nothing (abort drain + boundary unwind).

## Snapshots

`snapshot(scope, dest)` freezes an id set under **read-leases**
(reads proceed; writers — puts, wipes, runs touching those ids —
wait, parked, until the background writer finishes), streams
payload + manifest to `dest`, and dedups clean duplicates against
their parent via version counters (a `W@ck` whose parent later
trained stores its own bytes — soundness over savings).
`restore_snapshot` recreates residents and object groups and hands
back your `client_meta` — step counter, LR state, data cursor —
so resume is one call.

## Watching

`subscribe_events()` streams service events (`run_started/done`,
`snapshot_*`, `engine_*`); reconnect with `since_seq` to replay
what you missed. `engine_status()` / `run_status(run_id)` /
`query_backing()` (residents + per-program transients) answer from
the fast path even mid-run. Run traces (per-task timings) export
via `export_trace(run_id)`.

## Safety rails

The slab refuses to pin into the last 24 GiB of host memory; the
whole daemon runs fine under a systemd cgroup cap
(`systemd-run --scope -p MemoryMax=...`), which is the recommended
way to launch anything large. Device-side, the daemon boots with
`expandable_segments` and shares one stream set across programs so
long-lived multi-program service does not accumulate allocator
cache.

Full API + design rationale: `docs/notes/server_engine_api_s1.md`
and `docs/notes/engine_service_design.md` (local design notes).
