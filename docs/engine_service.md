# The dataflow engine service

A persistent daemon that owns pinned host memory (the **store
slab**), holds named objects as **residents**, and executes
registered dataflow **programs** against them. Clients connect over
a unix socket, put/fetch objects, register programs once, and run
them many times — persistent state lives in the store between runs,
so a long job is "run the registered program N times", each run
microseconds of control overhead away from the in-process engine
(the parity gates hold service-hosted runs to in-process throughput
and identical device/host memory peaks).

Start it:

```bash
python tools/train/dataflowd.py start --socket /tmp/dfd.sock --backing-gib 145
python tools/train/dataflowd.py status --socket /tmp/dfd.sock
python tools/train/dataflowd.py stop --socket /tmp/dfd.sock
```

`--backing-gib` is the ONE pinned budget (default `auto`): residents AND
run transients draw from the same slab.
Size it to `residents + worst-plan transients` (the plan's demand
bound); the daemon refuses to pin into the system's last 24 GiB.
`--fake` boots without CUDA for tests/dev; `--device N` picks the GPU;
`--kernels <set>` pins the kernel set; `--peer-name`/`--peer-listen`/
`--peer-rdma-device` arm the peer plane
([distributed_training.md](distributed_training.md)).

## The resolver registry (the workload seam)

The engine executes programs; what a task's `compute_block_key` MEANS
comes from a **registered resolver kind**
(`dataflow.service.registry`, contract:
[program_contract.md](program_contract.md)). Registration is a
workload-side import-time act — `register_program_resolver(kind,
build)` — and the daemon learns kinds three ways:

- **default**: `dataflowd.py start` loads
  `dataflow_training.register.register_all()`, which registers the
  builtin kind `"model_family"` — model-family programs resolve out of
  the box;
- **`--no-default-workloads`**: skip that — a bare engine daemon that
  knows NO kinds until a plugin registers some;
- **`--plugin <module>`** (repeatable, at boot) or the **`load_plugin`
  verb** (at runtime: `client.load_plugin({"module": "mypkg.plugin"})`
  or `{"path": "/abs/file.py"}`): import a module that self-registers;
  the verb's reply reports `kinds_registered` — the kinds that
  appeared during the import.

`client.list_resolvers()` returns `{"kinds": [...]}` — what the daemon
currently knows. Registering a program with an unknown
`resolver_spec["kind"]` fails loudly, naming the registered kinds.
(Bulk state initialization rides this same seam: the client
`create_object`s the residents, then registers + runs a HOST-task
program whose tasks resolve through a registered kind and fill those
extents in place — server-side init needs no engine vocabulary. The
training package's wrapper for exactly that, and its in-process
profiling helpers, live workload-side: [usage.md](usage.md).)

## Client in five verbs

```python
from dataflow.service import EngineClient

with EngineClient("/tmp/dfd.sock", client_name="driver") as c:
    # 1. state into the store. put_object uploads bytes outright;
    #    create_object allocates a resident with NO payload for a
    #    HOST-task init program to fill in place (bulk state exists
    #    once, never as transient + copy).
    for oid, nbytes in state_sizes.items():
        c.create_object(oid, nbytes)
    ini = c.register_program(init_program_dict, resolver=my_resolver)
    c.run(ini["prog_id"], args={})
    c.unregister_program(ini["prog_id"])
    c.put_object("in/x_0", chunk_bytes)          # external inputs

    # 2. register once (content-hashed id; placement cached). The
    #    resolver spec is opaque to the engine except for "kind".
    reg = c.register_program(program_dict, resolver=my_resolver)

    # 3. run many (args reach tasks as opaque run_args)
    for k in range(steps):
        c.put_object(f"in/x_{k+1}", next_chunk, wait=False)  # pre-stage
        r = c.run(reg["prog_id"], args={"step": k},
                  rebind={"in/x_0": f"in/x_{k}"},    # per-run inputs
                  fetch=["out/y_0"])
        print(k, r["fetched"]["out/y_0"], r["makespan_us"])

    # 4. checkpoint. snapshot freezes the saved ids under read-leases
    #    and streams payload + snapshot.json to disk in the background.
    s = c.snapshot("/ckpts/step100",
                   client_meta={"step": 100, "cursor": [3, 128]})
    c.wait_snapshot(s["snap_id"])

    # 4b. SLICE snapshots: an explicit slice list selects exactly which
    #     bytes to save and where they belong. Each slice names a
    #     stored object and may map a src byte range into a logical
    #     object (logical_id + dst + logical_bytes; all default to the
    #     identity mapping). Restoring each responsible saver's
    #     snapshot in turn REASSEMBLES the complete object. Explicit
    #     slice lists never dedup.
    c.snapshot("/ckpts/step100-r0",
               slices=[{"id": "state/w3", "src": [0, 1 << 20]}])

    # 5. resume later (client_meta comes back in the same call).
    #    Slice hashes are verified BEFORE any placement; a remap plan
    #    can extract logical ranges into differently-named local
    #    objects instead of the default placement.
    meta = c.restore_snapshot("/ckpts/step100")["client_meta"]
```

The training package wraps this whole sequence — init, per-step data
feed, checkpoints, resume — in its own driver: [usage.md](usage.md).

## The semantics in one paragraph

Objects are engine-global and flat-namespaced: any client sees
`state/w3`. A program's **initial objects** bind to residents at run
start (strict size match); whatever the program's
`final_locations` declares comes OUT resident; everything
else the run creates is a
**transient** — named in the program, never in the catalog, carved
lazily from the same slab, recycled across runs, returned at
`unregister_program`. `rebind` points a program input id at a
different resident per run (per-run input feed). Each run also snapshots
the daemon's live peer-group table: tasks that declare `comm_groups`
resolve their group by NAME at that moment, and run standalone when
it isn't there (distributed_training.md). Runs execute FIFO on
one dispatcher; status/query verbs answer instantly from a fast
path; `cancel_run` takes effect at the next task boundary; a
failed run poisons nothing (abort drain + boundary unwind).

An ALL-HOST program (every task `host: true` — program_schema.md)
never enters the engine: no placement dry-run, no session/pool. Its
tasks run synchronously on the dispatcher thread against the bound
objects' store extents, so in-place writes land directly in the
catalogued residents (bulk fills exist once, not twice).
Registration rejects programs mixing host and device tasks, host
tasks with outputs, and host tasks touching non-initial objects.

## The object plane

Beyond `put_object`/`fetch`: `get_object(id)` returns bytes (or
writes straight to a `dest` path for big residents);
`create_object(id, size_bytes)` allocates a catalogued extent with NO
payload — content is unspecified until first written, the intended
writer being a run that mutates it in place (a host task fills the
extent directly; same-size re-create is idempotent, a size change
is BINDING_MISMATCH);
`materialize_object` fills a resident server-side; **object groups**
name id sets (`create_object_group(name, members=...)` or one fnmatch
`pattern`, nestable via `object_groups=`; `query_object_group` lists
the resolved members; the scope names `"all"` and `"backing"` are
reserved). `wipe(scope)` frees residents by scope (an object-group
name, `"backing"`, or `"all"`) — it skips objects marked with
`protect_object` unless called with `force`, and refuses ids a
snapshot currently holds under lease. `unprotect_object` lifts the
mark. `validate_program` dry-runs registration (schema + binding
checks, nothing retained).

## Snapshots

`snapshot(dest)` saves every resident object whole;
`snapshot(dest, slices=[...])` saves exactly the listed slices, each
mapping a `src` byte range of a stored object into a logical object
(`logical_id` + `dst` + `logical_bytes`, defaulting to the identity
mapping). Either way the saved ids freeze under **read-leases**
(reads proceed; writers — puts, wipes, runs touching those ids —
wait, parked, until the background writer finishes) while payload
plus `snapshot.json` (schema `dataflow-snapshot/v1`, written last as
the completeness marker, one streaming blake2b-16 hash per slice)
land at `dest`. Bulk snapshots dedup clean duplicates against their
parent via version counters (a duplicate whose parent was later
mutated stores its own bytes — soundness over savings).
`restore_snapshot` is three-pass — validate every placement, verify
every slice hash (`verify=False` opts out), then place — so a
refusal leaves the store untouched; identity slices recreate their
stored objects exactly (metadata included), an optional `remap` plan
extracts logical ranges into local objects instead, and your
`client_meta` — whatever resume record the client stored — comes
back in the same call.

## The peer plane

Daemons talk to each other over peer links (`peer_connect`), form
named collective **groups** with one `create_peer_group` verb on the
group's coordinator (no join verb — members join and attach their
backends inside the barrier, so the verb returning means every
rank's backend is live), and carry object transfers and collectives
over socket, RDMA, or nccl lanes. The full design — planes, links,
threading, groups, backends, and how tasks consume groups — lives
in [engine_networking.md](engine_networking.md).

## Watching

`subscribe_events()` streams service events (`run_started/done`,
`snapshot_*`, `engine_*`); reconnect with `since_seq` to replay
what you missed. `engine_status()` / `run_status(run_id)` /
`query_backing()` (residents + per-program transients) answer from
the fast path even mid-run, as do `list_objects` / `list_programs` /
`list_runs` / `session_status` / `health`.

`profiler_control("start"/"stop")` flips the annotation layer and
`cudaProfilerStart/Stop` — under `nsys
--capture-range=cudaProfilerApi` the capture holds exactly the
bracketed runs; [benchmarking.md](benchmarking.md) packages the
recipe.

**Traces.** Every run records a per-task `RunTrace`; the daemon keeps
the last 200 events per run (`run_events(run_id)`,
`export_trace(run_id, dest)`). The `run` verb also takes a `trace`
flag — `c.run(pid, trace=True)` — to return the FULL trace in the
run reply (`r["trace"]`, the `trace_to_dict` form).

## Safety rails

The slab refuses to pin into the last 24 GiB of host memory; the
whole daemon runs fine under a systemd cgroup cap
(`systemd-run --scope -p MemoryMax=...`), which is the recommended
way to launch anything large. Device-side, the daemon boots with
`expandable_segments` and shares one stream set across programs so
long-lived multi-program service does not accumulate allocator
cache.

The workload<->engine contract (resolver kinds, host-task state
init, run_args opacity): [program_contract.md](program_contract.md).
The in-process engine surface underneath:
[engine_api.md](engine_api.md).
