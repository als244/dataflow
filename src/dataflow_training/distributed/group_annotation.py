"""Group annotation: parallelism applied to an ALREADY-LOWERED program.

Family lowering is parallelism-blind (``fam.lower(cfg)`` emits the pure
single-rank program). This pass attaches the distributed execution's
group addressing and per-rank parameters afterwards, keying ONLY on the
validated task-naming shape and the W-root each task consumes — never
on family internals. Per-rank object SIZES are the exact-sizes pass's
job (``object_size_factory`` with the rank view); this pass touches
``comm_groups`` and ``block_params`` and nothing else.

The three annotations (byte-for-byte what the shaped builder's retired
``dp_group``/``shard_params``/``tp_params`` parameters used to emit):

- every optimizer task gains ``comm_groups={"dp": group}`` — the
  present handle is what switches its executable to the collective
  update variant;
- an optimizer task whose W-root appears in ``shard_params`` gains
  ``block_params["shard"]`` (region reduce -> owned-only update ->
  broadcast, the ZeRO-1 form); in ``tp_params`` it gains
  ``block_params["tp_slices"]`` (replica-grads local update);
- a block fwd/recompute/bwd task whose W-root appears in ``tp_params``
  gains ``block_params["tp_slices"]`` and ``comm_groups={"tp": group}``
  (the sharded-MLP partial+allreduce variants).
"""
from dataclasses import replace

from dataflow.core.program import Program, TaskSpec

# optimizer compute kinds; the W-root is the task's first input
_OPT_KINDS = ("optimizer_embed", "optimizer_block", "optimizer_head")
# block-chain kinds carry their W-root as the second input (x/W/...);
# recognized by suffix so family kind prefixes (tp_, warmup_, family
# names) stay transparent
_BLOCK_SUFFIXES = ("_fwd", "_recompute", "_bwd")


def _w_root(task: TaskSpec) -> str | None:
    for oid in task.inputs:
        if oid == "W_embed" or oid == "W_head" or (
                oid.startswith("W_") and oid[2:].isdigit()):
            return oid
    return None


def annotate_groups(program: Program, *, group: str,
                    shard_params=None, tp_params=None) -> Program:
    """The lowered program with group addressing attached; a NEW
    Program (specs are frozen). ``group`` is the peer-group NAME
    (purpose keys "dp"/"tp" address it per task role)."""
    if not group:
        raise ValueError("annotate_groups needs a peer-group name")
    shards = shard_params or {}
    tps = tp_params or {}
    out = []
    for t in program.tasks:
        kind = t.compute_block_key
        root = _w_root(t)
        if kind in _OPT_KINDS:
            extra = {}
            if root in shards:
                extra["shard"] = shards[root]
            if root in tps:
                extra["tp_slices"] = tps[root]
            t = replace(
                t,
                block_params={**t.block_params, **extra},
                comm_groups={**t.comm_groups, "dp": group},
            )
        elif kind.endswith(_BLOCK_SUFFIXES) and root in tps:
            # the tp executable VARIANT: kind gains the tp_ prefix (the
            # builder's key_prefix swap), alongside the annotations
            t = replace(
                t,
                compute_block_key=f"tp_{kind}",
                block_params={**t.block_params, "tp_slices": tps[root]},
                comm_groups={**t.comm_groups, "tp": group},
            )
        out.append(t)
    # kind references live OUTSIDE the task list too: the recompute
    # rewrite table names the fwd/recompute kinds the planner uses to
    # build variants — swap those in lockstep with the task kinds
    # (the equivalence gate caught this the first time)
    swapped = {t.id: t.compute_block_key for t in out}
    rewrites = []
    for rw in program.recompute_rewrites:
        f_kind = swapped.get(rw.f_task_id, rw.f_compute_block_key)
        r_kind = rw.r_compute_block_key
        if f_kind != rw.f_compute_block_key:
            # recompute tasks are PLANNER-inserted — r_task_id names a
            # future task, so the r kind follows its fwd's swap
            r_kind = f"tp_{r_kind}"
            rw = replace(rw, f_compute_block_key=f_kind,
                         r_compute_block_key=r_kind)
        rewrites.append(rw)
    return replace(program, tasks=tuple(out),
                   recompute_rewrites=tuple(rewrites))
