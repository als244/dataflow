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
python tools/dataflowd.py start --socket /tmp/dfd.sock --slab-gib 145
python tools/dataflowd.py status
python tools/dataflowd.py stop
```

`--slab-gib` is the ONE pinned budget (default `auto`): residents AND
run transients (gradients, saved activations) draw from the same slab.
Size it to `residents + worst-plan transients` (the plan's demand
bound); the daemon refuses to pin into the system's last 24 GiB.
`--fake` boots without CUDA for tests/dev; `--peer-name`/`--peer-listen`
arm the peer plane ([distributed_training.md](distributed_training.md)).

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
the fast path even mid-run.

**Traces.** Every run records a per-task `RunTrace`; the daemon keeps
the last 200 events per run (`run_events(run_id)`,
`export_trace(run_id, dest)`). The `run` verb also takes a `trace`
flag to return the FULL trace in the run reply (`r["trace"]`, the
`trace_to_dict` form). Wiring quirk, faithfully: the daemon currently
reads that flag from the run-args dict, so request it as
`c.run(pid, args={"step": k, "trace": True})` — the client's
`trace=True` keyword is sent top-level and does not reach the check
(tasks ignore the extra opaque run-arg).

## Retired verbs

Three early service verbs are gone; their jobs moved across the seam:

- **`materialize_group` -> `init_model`** (init-as-program): server-side
  seeded init was workload vocabulary inside the engine. Now the init
  is an ordinary one-task program resolved through the registered kind
  (`family_init`), driven by `dataflow_training.run.driver.init_model`
  — byte-identical to in-process init, and the engine stays blind.
- **`list_families` -> `list_resolvers`**: the daemon does not know
  what a "family" is; it knows registered resolver KINDS. Family
  enumeration lives workload-side
  (`dataflow_training.model_families.families`, `tools/list_models.py`).
- **`profile_program` -> in-process `load_or_profile`**: profiling
  drives executables directly and caches on disk
  (`dataflow_training.run.profiling.load_or_profile` /
  `apply_measured_costs`); it never needed the daemon.

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
