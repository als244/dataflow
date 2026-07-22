# The dataflow engine service

A persistent daemon that owns pinned host memory (the **store
slab**), holds named objects (weights, optimizer state, data,
losses) as **residents**, and executes registered dataflow
**programs** against them. Clients connect over a unix socket,
put/fetch objects, register programs once, and run them many times
— training state lives in the store between runs, so a training
job is "run the step program N times", each step microseconds of
control overhead away from the in-process engine (the parity gates
hold service-hosted runs to in-process tok/s and identical
device/host memory peaks).

Start it:

```bash
python tools/train/dataflowd.py start --socket /tmp/dfd.sock --backing-gib 145
python tools/train/dataflowd.py status --socket /tmp/dfd.sock
python tools/train/dataflowd.py stop --socket /tmp/dfd.sock
```

`--backing-gib` is the ONE pinned budget (default `auto`): residents AND
run transients (gradients, saved activations) draw from the same slab.
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
(Model init also rides this seam: `init_model` builds the family's
one-task init program and runs it through the ordinary verbs — its
task resolves through the `"model_family"` kind's `family_init`
compute key, so server-side init needs no engine vocabulary. Family
enumeration is likewise workload-side:
`dataflow_training.model_families.families` / `tools/gen_model_docs/list_models.py`.
Task-cost profiling never needs the daemon at all — it drives
executables in-process: `dataflow_training.run.profiling
.load_or_profile` / `apply_measured_costs`.)

## Client in five verbs

```python
from dataflow.service import EngineClient
from dataflow_training.run.driver import init_model     # workload-side sugar

with EngineClient("/tmp/dfd.sock", client_name="driver") as c:
    # 1. state into the store. INIT IS A PROGRAM: init_model builds the
    #    family's one-task init program, registers + runs it through the
    #    ordinary verbs, and the final-object capture persists every
    #    W_/O_/Aux_/data object as a resident.
    init_model(c, "llama3", cfg_dict, seed=11)
    c.put_object("tokens_0_0", token_bytes)     # data chunks

    # 2. register once (content-hashed id; placement cached). The
    #    resolver spec is opaque to the engine except for "kind".
    reg = c.register_program(program_dict,
                             resolver={"kind": "model_family",
                                       "family": "llama3", "cfg": cfg_dict})

    # 3. run many (args reach tasks as opaque run_args: step, valid_rows, ...)
    for k in range(steps):
        c.put_object(f"tokens_{k+1}", next_chunk, wait=False)  # pre-stage
        r = c.run(reg["prog_id"], args={"step": k},
                  rebind={"tokens_0_0": f"tokens_{k}"},    # per-step data
                  fetch=["loss_0_0"])
        print(k, r["fetched"]["loss_0_0"], r["makespan_us"])

    # 4. checkpoint. duplicate_object_group copies a named object group
    #    synchronously on the dispatcher; snapshot freezes ids under
    #    read-leases and streams to disk in the background.
    c.create_object_group("weights", pattern="W_*")   # fnmatch glob
    c.duplicate_object_group("weights", tag="ck")
    s = c.snapshot("all", "/ckpts/step100",
                   client_meta={"step": 100, "cursor": [3, 128]})
    c.wait_snapshot(s["snap_id"])

    # 4b. SLICE snapshots: save only the byte range this saver is
    #     RESPONSIBLE for (slice-granular save plans). A ranged entry
    #     records the FULL object size plus its [lo, hi); restore
    #     fills the range into the resident object (overwrite=True) or
    #     creates the object full-size and fills the range — restoring
    #     each responsible rank's artifact in turn REASSEMBLES the
    #     complete object. Ranged entries never dedup.
    c.snapshot("all", "/ckpts/step100-r0", ids=["W_3"],
               ranges={"W_3": (0, 1 << 20)})

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
different resident per run (per-step data feed). Each run also snapshots
the daemon's live peer-group table: tasks that declare `comm_groups`
resolve their group by NAME at that moment, and run standalone when
it isn't there (distributed_training.md). Runs execute FIFO on
one dispatcher; status/query verbs answer instantly from a fast
path; `cancel_run` takes effect at the next task boundary; a
failed run poisons nothing (abort drain + boundary unwind).

## The object plane

Beyond `put_object`/`fetch`: `get_object(id)` returns bytes (or
writes straight to a `dest` path for big residents);
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

`snapshot(scope, dest)` freezes an id set under **read-leases**
(reads proceed; writers — puts, wipes, runs touching those ids —
wait, parked, until the background writer finishes), streams
payload + manifest to `dest`, and dedups clean duplicates against
their parent via version counters (a `W@ck` whose parent later
trained stores its own bytes — soundness over savings).
`restore_snapshot` recreates residents and object groups and hands
back your `client_meta` — step counter, LR state, data cursor —
so resume is one call.

## The peer plane: links, groups, backends

Daemons speak to each other over **peer links** — TCP control
connections the client wires up with `peer_connect(name,
"ip:port")`. A link carries framed control messages, object
transfers (`send_object`; payloads ride the link on the socket lane
or one-sided RDMA writes when both ends brought RC QPs up), and the
collective frames below. For a fleet, the conductor connects a
**star**: the daemon that will coordinate each group holds a link to
every other member.

**Groups are created by one verb, on one daemon.** There is no join
verb: the client calls `create_peer_group(name, members, backend)`
once, on the daemon it thereby makes the **coordinator** (rank 0 —
member ranks are positions in the member list). The coordinator
pushes a `GROUP_JOIN` frame down its star links; each member adopts
the group from the frame, **attaches its backend, and only then**
answers `GROUP_ACK`. When the ack barrier fills, the coordinator
attaches its own backend and the verb returns. The join barrier is
therefore an **attachment barrier**: `create_peer_group` returning
means every rank's backend is live and the group can carry
collectives immediately — no rank ever races bring-up against
traffic.

**Backends attach the same way on both lanes.** With
`backend="nccl"`, attachment is the NCCL bootstrap (the uniqueId
rides the join frame; init is collective, so members bootstrap on
dedicated threads between join and ack). With `backend="hostmem"`,
attachment builds the staged-exchange machinery — pinned staging
regions, the exchange worker, the stream-release flag — and rides
RDMA between members when their link has it, sockets otherwise.
Either way a member that fails to attach withholds its ack, the
coordinator's barrier times out, and the verb fails loudly; a
member that dies later fans `GROUP_ERROR` through the coordinator
to every rank and the group refuses further work.

Tasks reach their groups by *role*: a program task declares
`comm_groups={"dp": <group name>}` and the engine injects the live
handles into its context — see
[distributed_training.md](distributed_training.md) for how training
composes this.

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
bracketed steps; `train.py --profile` packages the recipe
([benchmarking.md](benchmarking.md)).

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

The workload<->engine contract (resolver kinds, init-as-program,
run_args opacity): [program_contract.md](program_contract.md). The
in-process engine surface underneath: [engine_api.md](engine_api.md).
