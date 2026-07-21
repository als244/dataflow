"""The responsibility map: who STEPS and therefore SAVES each byte.

Every parameter slice has a RESPONSIBLE rank — it holds the slice's
optimizer state, performs its step, and saves its checkpoint bytes —
plus optional BACKUP ranks (fault tolerance, multiplicity recorded
from day one even while everything runs at 1). The map is pure
derived data:

    {object_id: [{"rank": r, "lo": a, "hi": b,
                  "role": "responsible" | "backup"}, ...]}

Modes:
- world 1: rank 0 responsible for everything, full ranges.
- "zero1rs" (the DP DEFAULT at world > 1): parameter bytes partition
  at the SAME flat-slice boundaries the zero1rs optimizer shards by
  (each rank steps params/n — reduce-scatter in, all-gather out), so
  checkpoint IO balance is automatic and save == step ownership.
  Each rank's own O objects are fully its responsibility.
- "co" (the co-responsible DIAGNOSTIC lane, was plain DP): every rank
  holds identical bytes (certified bitwise); ONE primary per object,
  assigned largest-first to the least-loaded rank (pure IO-balance
  choice), remaining ranks recorded as backups.

Sizes come from the layout registry (the coordinate system); zero1rs
boundaries from the sharding module's own math — this module invents
neither.
"""
from dataflow_training.model_families.layout_registry import registered_layouts

_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "int32": 4, "int64": 8}


def _full(rank: int, size: int, role: str = "responsible") -> dict:
    return {"rank": rank, "lo": 0, "hi": size, "role": role}


def _root_sizes(cfg) -> dict:
    """{W_root: total_bytes} via the layout registry."""
    out = {}
    for rl in registered_layouts(cfg).values():
        for root in rl.roots:
            out[root] = rl.total_bytes
    return out


def responsibility_map(cfg, world: int, *, mode: str = "zero1rs",
                       shard_params=None) -> dict:
    """The save plan. ``shard_params`` (required for zero1rs at
    world > 1) is sharding.zero1rs_block_params' output — the SAME
    boundaries the optimizer steps by."""
    sizes = _root_sizes(cfg)
    roots = sorted(sizes)
    if world == 1:
        # O objects ride their owner's artifact wholesale (rank_save_args)
        return {root: [_full(0, sizes[root])] for root in roots}
    if mode == "zero1rs":
        if not shard_params:
            raise ValueError("zero1rs responsibility needs shard_params")
        plan = {}
        for root in roots:
            sh = shard_params.get(root)
            if sh is None:
                # root not byte-equal eligible: falls back to
                # co-responsible with rank 0 primary
                plan[root] = ([_full(0, sizes[root])] +
                              [_full(r, sizes[root], "backup")
                               for r in range(1, world)])
                continue
            esize = _DTYPE_BYTES[sh["opt_dtype"]]
            n_slice, n_tail = sh["n_slice"], sh["n_tail"]
            entries = []
            lo_e = 0
            for r in range(world):
                n = n_slice + (n_tail if r == world - 1 else 0)
                hi_e = lo_e + n
                entries.append({"rank": r, "lo": lo_e * esize,
                                "hi": hi_e * esize,
                                "role": "responsible"})
                lo_e = hi_e
            plan[root] = entries
        return plan
    if mode == "co":
        loads = [0] * world
        plan = {}
        for root in sorted(roots, key=sizes.get, reverse=True):
            primary = loads.index(min(loads))
            loads[primary] += sizes[root]
            plan[root] = ([_full(primary, sizes[root])] +
                          [_full(r, sizes[root], "backup")
                           for r in range(world) if r != primary])
        return plan
    raise ValueError(f"unknown responsibility mode {mode!r}")


def rank_save_args(plan: dict, rank: int, own_objects) -> tuple:
    """What THIS rank snapshots: (ids, ranges). Param ranges come from
    the plan; the rank's own per-rank objects (its O shards, listed in
    ``own_objects``) ride wholesale."""
    ids, ranges = [], {}
    for oid, entries in plan.items():
        size = max(e["hi"] for e in entries)
        for e in entries:
            if e["rank"] == rank and e["role"] == "responsible":
                ids.append(oid)
                if not (e["lo"] == 0 and e["hi"] == size):
                    # partial responsibility -> ranged save; full-object
                    # responsibility snapshots whole (dedup-eligible)
                    ranges[oid] = (e["lo"], e["hi"])
    for oid in own_objects:
        if oid not in ids:
            ids.append(oid)
    return sorted(ids), ranges
