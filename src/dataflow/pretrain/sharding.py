"""The sharding API: conductor-owned ownership algebra over packed
layouts, group-scoped by construction.

Vocabulary (plan doc, Z0/Z0a/T0): a ShardPlan assigns REGIONS —
(object_root, field, optional (lo, hi) slice along ``dim``) — with
two independent axes per assignment:

- ``owner``: the rank that RUNS THE OPTIMIZER for the region — it
  holds the region's optimizer state (nobody else has those bytes),
  computes the new params each step, and propagates them. ALL_RANKS
  = every rank updates redundantly from full state (the cheap choice
  for tiny fields; no propagation traffic).
- ``resident``: which ranks hold the PARAMETER BYTES at all.
  ALL_RANKS = replicated weights (the zero1 configuration). A single
  rank = the shard physically lives only there (tensor parallelism;
  the per-rank layouts materialize it at shard shape via
  ``tp_view``). dim-1 (column) slices exist ONLY as such layout
  transforms — the byte-range machinery is dim-0.

Ownership is the vocabulary; consumers decide meaning
(``consumable(mode)``): the ``optimizer`` consumer (zero1) requires
resident=ALL_RANKS everywhere; the ``tp`` consumer additionally
accepts owner == resident == one rank.

Collective realizability: reduce_scatter/all_gather need EQUAL
per-rank counts; field-snapped shards are unequal by up to one field,
so plans report ``equal_shards(root)`` honestly and the v1 comm_seq
composes from per-shard reduce+broadcast instead (wire-neutral,
world-N, whole-field optimizer state preserved).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

ALL_RANKS = "all"


@dataclass(frozen=True)
class Region:
    object_root: str               # e.g. "W_{i}" instantiated per layer
    field: str
    rows: tuple | None = None      # (lo, hi) slice along ``dim``
    dim: int = 0                   # shard axis: 0 = rows (byte-
                                   # contiguous), 1 = columns (a layout
                                   # TRANSFORM — resident-narrowed
                                   # consumers only; the zero1 byte-
                                   # range machinery requires dim == 0)

    def key(self) -> tuple:
        return (self.object_root, self.field,
                self.rows if self.rows else None, self.dim)


@dataclass(frozen=True)
class Assignment:
    region: Region
    owner: object                  # rank int | ALL_RANKS: who runs
                                   # the optimizer for this region
    resident: object = ALL_RANKS   # who holds the param bytes:
                                   # ALL_RANKS (replicated) | rank int
                                   # (a physical shard, tp consumer)


@dataclass(frozen=True)
class FieldInfo:
    name: str
    shape: tuple
    dtype: str
    offset_bytes: int
    nbytes: int

    def rows_total(self) -> int:
        return self.shape[0] if self.shape else 1

    def row_bytes(self) -> int:
        return self.nbytes // max(self.rows_total(), 1)


def field_infos(layout) -> list:
    out = []
    for f in layout.fields:
        n = 1
        for s in f.shape:
            n *= s
        item = {"bf16": 2, "f16": 2, "fp16": 2, "f32": 4, "fp32": 4,
                "int32": 4, "int64": 8}.get(f.dtype, 2)
        out.append(FieldInfo(f.name, tuple(f.shape), f.dtype,
                             f.offset_bytes, n * item))
    return out


@dataclass
class ShardPlan:
    """Group-scoped ownership plan. ``layouts`` maps object_root ->
    the packed WEIGHT layout the roots' dW/O mirror."""

    group: str
    world: int
    assignments: tuple
    fields_by_root: dict = dc_field(default_factory=dict)

    # ---------------------------------------------------- queries
    def owned(self, rank: int) -> list:
        return [a.region for a in self.assignments if a.owner == rank]

    def redundant(self) -> list:
        return [a.region for a in self.assignments
                if a.owner == ALL_RANKS]

    def resident(self, rank: int) -> list:
        return [a.region for a in self.assignments
                if a.resident == ALL_RANKS or a.resident == rank
                or (isinstance(a.resident, (set, frozenset, tuple))
                    and rank in a.resident)]

    def field_owner(self, root: str, field_name: str):
        for a in self.assignments:
            if (a.region.object_root == root
                    and a.region.field == field_name):
                return a.owner
        return None

    def roots(self) -> list:
        seen = []
        for a in self.assignments:
            if a.region.object_root not in seen:
                seen.append(a.region.object_root)
        return seen

    def owned_fields(self, rank: int, root: str) -> list:
        """Field names of root fully or partially owned by rank."""
        return [a.region.field for a in self.assignments
                if a.region.object_root == root
                and (a.owner == rank or a.owner == ALL_RANKS)]

    def sharded_assignments(self, root: str) -> list:
        return [a for a in self.assignments
                if a.region.object_root == root
                and a.owner != ALL_RANKS]

    def v1_consumable(self) -> None:
        self.consumable("optimizer")

    def consumable(self, mode: str) -> None:
        """Consumer contracts. ``optimizer`` (the zero1 configuration):
        everything resident everywhere, dim-0 regions only.  ``tp``
        (tensor parallelism): a region may narrow residency iff its
        owner IS its (single-rank) resident — the shard physically
        lives only there; replicated fields follow the optimizer
        contract unchanged."""
        for a in self.assignments:
            if a.resident == ALL_RANKS:
                if a.region.dim != 0:
                    raise ValueError(
                        f"plan region {a.region.key()}: dim-"
                        f"{a.region.dim} sharding of a REPLICATED "
                        f"field — column shards exist only as "
                        f"resident-narrowed layout transforms")
                continue
            if mode == "optimizer":
                raise ValueError(
                    f"plan region {a.region.key()} narrows residency "
                    f"({a.resident!r}) — parameter placement is not "
                    f"executable by the optimizer-sharding consumer")
            if mode == "tp":
                if a.owner != a.resident or not isinstance(
                        a.resident, int):
                    raise ValueError(
                        f"plan region {a.region.key()}: tp consumer "
                        f"needs owner == resident == one rank, got "
                        f"owner={a.owner!r} "
                        f"resident={a.resident!r}")
                if a.region.rows is None:
                    raise ValueError(
                        f"plan region {a.region.key()}: resident-"
                        f"narrowed fields carry an explicit (lo, hi) "
                        f"slice per rank")
                continue
            raise ValueError(f"unknown consumer mode {mode!r}")

    # ------------------------------------------------ realizability
    def owned_ranges(self, rank: int, root: str) -> list:
        """Byte ranges of root's packed buffer owned by rank (rows are
        contiguous in row-major layouts)."""
        infos = {f.name: f for f in self.fields_by_root[root]}
        ranges = []
        for a in self.sharded_assignments(root):
            if a.owner != rank:
                continue
            if a.region.dim != 0:
                raise ValueError(
                    f"{root}.{a.region.field}: byte-range queries are "
                    f"dim-0 only; dim-{a.region.dim} shards are layout "
                    f"transforms (tp_view)")
            fi = infos[a.region.field]
            if a.region.rows is None:
                ranges.append((fi.offset_bytes,
                               fi.offset_bytes + fi.nbytes))
            else:
                lo, hi = a.region.rows
                rb = fi.row_bytes()
                ranges.append((fi.offset_bytes + lo * rb,
                               fi.offset_bytes + hi * rb))
        ranges.sort()
        merged = []
        for lo, hi in ranges:
            if merged and lo == merged[-1][1]:
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append((lo, hi))
        return merged

    def equal_shards(self, root: str) -> bool:
        """True iff every rank owns ONE contiguous run, runs are
        rank-major ordered, and counts are equal — the rs/ag fast-path
        precondition."""
        runs = [self.owned_ranges(r, root) for r in range(self.world)]
        if any(len(rr) != 1 for rr in runs):
            return False
        sizes = {rr[0][1] - rr[0][0] for rr in runs}
        ordered = all(runs[r][0][1] == runs[r + 1][0][0]
                      for r in range(self.world - 1))
        return len(sizes) == 1 and ordered

    # ---------------------------------------------------- validation
    def validate(self, opt_policy=None) -> None:
        """Full cover, no overlap, bounds, and the optimizer-aware
        rule: muon-ruled fields must be whole-matrix units."""
        from dataflow.tasks.optim import resolve_opt_policy

        op = resolve_opt_policy(opt_policy) if opt_policy is not None \
            else None
        for root, infos in self.fields_by_root.items():
            known = {f.name for f in infos}
            for a in self.assignments:
                if (a.region.object_root == root
                        and a.region.field not in known):
                    raise ValueError(f"{root}: unknown field "
                                     f"{a.region.field!r}")
            for fi in infos:
                self.validate_field(root, fi, op)

    def validate_field(self, root: str, fi, op) -> None:
        """One field's assignments, checked in plain order: a single
        shard axis, in-bounds spans, the muon whole-matrix rule, then
        gap/overlap-free cover of that axis."""
        assigns = [a for a in self.assignments
                   if a.region.object_root == root
                   and a.region.field == fi.name]
        dims = sorted({a.region.dim for a in assigns})
        if len(dims) > 1:
            raise ValueError(f"{root}.{fi.name}: mixed shard axes "
                             f"{dims} — one axis per field")
        dim = dims[0] if dims else 0
        if dim >= max(len(fi.shape), 1):
            raise ValueError(f"{root}.{fi.name}: dim {dim} out of "
                             f"range for shape {fi.shape}")
        extent = fi.shape[dim] if fi.shape else 1
        spans = []
        for a in assigns:
            if a.region.rows is None:
                if dim != 0:
                    raise ValueError(f"{root}.{fi.name}: whole-field "
                                     f"regions use dim 0")
                spans.append((0, extent))
                continue
            lo, hi = a.region.rows
            if not (0 <= lo < hi <= extent):
                raise ValueError(
                    f"{root}.{fi.name}: slice {a.region.rows} out of "
                    f"bounds (0, {extent}) on dim {dim}")
            if (op is not None and a.owner != ALL_RANKS
                    and op.for_field(fi.name, None, fi.shape)
                    == "muon"):
                raise ValueError(
                    f"{root}.{fi.name}: muon-ruled fields are "
                    f"whole-matrix units — splits break "
                    f"Newton-Schulz")
            spans.append((lo, hi))
        spans.sort()
        pos = 0
        for lo, hi in spans:
            if lo != pos:
                raise ValueError(f"{root}.{fi.name}: cover "
                                 f"gap/overlap at {pos} (next span "
                                 f"starts {lo})")
            pos = hi
        if pos != extent:
            raise ValueError(f"{root}.{fi.name}: covered to {pos} "
                             f"of {extent}")

    # --------------------------------------------- group derivation
    def required_groups(self) -> list:
        """SHAPE-now/substance-later hook (plan doc Z0a): v1 returns
        exactly the root group; purposes sharing a member set share a
        comm."""
        return [{"name": self.group, "purpose": "root"}]

    # ---------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {"group": self.group, "world": self.world,
                "assignments": [
                    {"root": a.region.object_root,
                     "field": a.region.field,
                     "rows": list(a.region.rows) if a.region.rows
                     else None,
                     "dim": a.region.dim,
                     "owner": a.owner,
                     "resident": a.resident}
                    for a in self.assignments]}

    @staticmethod
    def from_dict(d: dict, fields_by_root: dict) -> "ShardPlan":
        assigns = tuple(
            Assignment(Region(x["root"], x["field"],
                              tuple(x["rows"]) if x["rows"] else None,
                              dim=int(x.get("dim", 0))),
                       x["owner"],
                       resident=x.get("resident", ALL_RANKS))
            for x in d["assignments"])
        return ShardPlan(group=d["group"], world=d["world"],
                         assignments=assigns,
                         fields_by_root=fields_by_root)


@dataclass(frozen=True)
class ParallelConfig:
    """THE single parallelism argument a lowering takes: the group
    (mesh), the plan (placement), and my position in it. plan=None =
    plain data parallelism (replicated everything)."""

    group: str
    rank: int
    world: int
    plan: ShardPlan | None = None


def update_regions(plan: ShardPlan, rank: int) -> dict:
    """{object_root -> {field -> None | (lo, hi)}}: the regions RANK
    updates (its own plus every ALL_RANKS assignment) — the exact map
    ``opt_state_layout(update_regions=...)`` sizes O slots from.
    Optimizer-consumer semantics: replicated params, dim-0 regions
    against the FULL layouts; resident-narrowed (tp) fields are
    handled through per-rank layouts instead and are rejected here."""
    out: dict = {}
    for a in plan.assignments:
        if a.resident != ALL_RANKS:
            raise ValueError(
                f"{a.region.object_root}.{a.region.field}: resident-"
                f"narrowed region in the optimizer-consumer map — tp "
                f"fields size their O through the per-rank layouts "
                f"(tp_view), not update_regions")
        if a.owner != rank and a.owner != ALL_RANKS:
            continue
        per = out.setdefault(a.region.object_root, {})
        if a.region.field in per:
            raise ValueError(
                f"{a.region.object_root}.{a.region.field}: rank {rank} "
                f"assigned twice — one region per (rank, field)")
        per[a.region.field] = a.region.rows
    return out


def shard_block_params(plan: ShardPlan, rank: int) -> dict:
    """{object_root -> the JSON-able ``shard`` dict baked into that
    root's optimizer task block_params}:

      {"update": {field: None | [lo, hi]},     # what RANK updates
       "comm":   [{"field", "rows", "owner"}, ...]}  # sharded regions

    ``comm`` is identical on every rank and in plan order — the
    collective sequences must match across the group. Replicated
    (ALL_RANKS) fields appear only in ``update``: they keep the plain
    redundant-update allreduce."""
    upd = update_regions(plan, rank)
    out: dict = {}
    for root in plan.roots():
        comm = [{"field": a.region.field,
                 "rows": list(a.region.rows) if a.region.rows else None,
                 "owner": a.owner}
                for a in plan.sharded_assignments(root)]
        update = {name: (list(rows) if rows else None)
                  for name, rows in upd.get(root, {}).items()}
        out[root] = {"update": update, "comm": comm}
    return out


def tp_view(plan: ShardPlan, rank: int) -> dict:
    """{object_root -> {field -> (dim, lo, hi)}}: the resident-
    narrowed layout transform for RANK. Per-rank family layouts
    materialize these fields at shard shape; a field resident
    entirely on OTHER ranks is absent from this rank's view (and so
    from its layouts)."""
    out: dict = {}
    for a in plan.assignments:
        if a.resident == ALL_RANKS or a.resident != rank:
            continue
        root = a.region.object_root
        if root not in out:
            out[root] = {}
        if a.region.field in out[root]:
            raise ValueError(
                f"{root}.{a.region.field}: rank {rank} holds two "
                f"resident slices — one shard per (rank, field)")
        lo, hi = a.region.rows
        out[root][a.region.field] = (a.region.dim, int(lo), int(hi))
    return out


def tp_opt_block_params(plan: ShardPlan, rank: int) -> dict:
    """Per-root optimizer shard dicts for a TENSOR-PARALLEL run.
    Grads are REPLICAS (tp splits compute, not data), so the dicts
    carry grads="replica": comm entries skip the reduce and only
    broadcast after the owner's update. Resident-narrowed fields are
    fully local — whole shard-shaped fields of the PER-RANK layout,
    update-only, no comm. Replicated fields keep the standard
    owner+broadcast configuration."""
    out: dict = {}
    for root in plan.roots():
        update: dict = {}
        comm = []
        for a in plan.assignments:
            if a.region.object_root != root:
                continue
            if a.resident != ALL_RANKS:
                if a.resident == rank:
                    # this rank's physical shard: a whole field in the
                    # per-rank layout; grads and update fully local
                    update[a.region.field] = None
                continue
            rows = (list(a.region.rows) if a.region.rows else None)
            if a.owner == ALL_RANKS:
                update[a.region.field] = rows
                continue
            comm.append({"field": a.region.field, "rows": rows,
                         "owner": a.owner})
            if a.owner == rank:
                update[a.region.field] = rows
        out[root] = {"update": update, "comm": comm,
                     "grads": "replica"}
    return out


def tp_mlp_shards(fields_by_root: dict, group: str, world: int, *,
                  col_fields: tuple = ("w1", "w3"),
                  row_fields: tuple = ("w2",)) -> ShardPlan:
    """The tensor-parallel MLP plan: the named MLP fields shard over
    their d_ff axis with resident == owner == rank (``col_fields``
    slice dim 1, ``row_fields`` dim 0; equal slices, so the extent
    must divide by world). Every OTHER field stays replicated and
    rides the standard sharded-optimizer configuration (zero1
    bucketing) — what matters there is replication, not TP.

    EVERY replicated field gets an owner (no ALL_RANKS redundant
    updates): under tp the grads are per-rank REPLICAS, and on a
    heterogeneous pair they differ by gemm ulps — redundantly-updated
    replicas drift apart one bf16 quantum at a time with nothing to
    re-pin them, silently corrupting training (each rank's shard
    learns against its own drifted norms while the allreduced
    activations keep per-rank losses identical — found the hard way
    at 1B x 300 steps). Owner+broadcast re-pins every replicated
    field every step; the extra wire for norm-sized fields is
    negligible."""
    assignments = []
    replicated_by_root: dict = {}
    for root, infos in fields_by_root.items():
        rest = []
        for f in infos:
            if f.name not in col_fields and f.name not in row_fields:
                rest.append(f)
                continue
            dim = 1 if f.name in col_fields else 0
            extent = f.shape[dim]
            if extent % world:
                raise ValueError(
                    f"{root}.{f.name}: extent {extent} on dim {dim} "
                    f"does not divide by world {world}")
            per = extent // world
            for r in range(world):
                assignments.append(Assignment(
                    Region(root, f.name, rows=(r * per, (r + 1) * per),
                           dim=dim),
                    owner=r, resident=r))
        if rest:
            replicated_by_root[root] = rest
    if replicated_by_root:
        sub = zero1_halves(replicated_by_root, group, world,
                           small_field_owners=True)
        assignments.extend(sub.assignments)
    return ShardPlan(group=group, world=world,
                     assignments=tuple(assignments),
                     fields_by_root=fields_by_root)


def zero1_halves(fields_by_root: dict, group: str, world: int,
                 replicate_below_bytes: int = 1 << 16,
                 small_field_owners: bool = False) -> ShardPlan:
    """The v1 builder: per root, greedy field-snapped byte buckets —
    approximately equal 1/world shards, whole fields only. Fields
    smaller than ``replicate_below_bytes`` (norms, biases) go
    owner=ALL_RANKS: redundant update is cheaper than propagating
    them — SOUND ONLY when grads are identical across ranks (the DP
    allreduce guarantees it). ``small_field_owners`` assigns them
    round-robin owners instead (tp/replica-grads runs, where
    redundant updates drift across an arch boundary)."""
    assignments = []
    for root, infos in fields_by_root.items():
        big = [f for f in infos if f.nbytes >= replicate_below_bytes]
        small = [f for f in infos if f.nbytes < replicate_below_bytes]
        if len(big) == 1 and world > 1:
            # single-matrix roots (embed/head): field snapping cannot
            # split them — shard by ROW ranges instead (elementwise
            # optimizers only; validate(opt_policy) rejects muon)
            f = big[0]
            rows = f.rows_total()
            per = rows // world
            for r in range(world):
                lo = r * per
                hi = rows if r == world - 1 else (r + 1) * per
                assignments.append(
                    Assignment(Region(root, f.name, rows=(lo, hi)), r))
        else:
            total = sum(f.nbytes for f in big)
            target = total / world if world else total
            rank, acc = 0, 0
            for f in big:                 # layout order => contiguity
                # advance when this field's CENTER crosses the bucket
                # boundary — plain `acc >= boundary` only fires after
                # the bucket overshot, silently skewing every root
                # toward rank 0 (observed 63/37 on real layouts)
                while (rank < world - 1
                       and acc + f.nbytes / 2 >= target * (rank + 1)):
                    rank += 1
                assignments.append(Assignment(Region(root, f.name),
                                              rank))
                acc += f.nbytes
        for k, f in enumerate(small):
            owner = (k % world) if small_field_owners else ALL_RANKS
            assignments.append(Assignment(Region(root, f.name), owner))
    return ShardPlan(group=group, world=world,
                     assignments=tuple(assignments),
                     fields_by_root=fields_by_root)


def expert_shards(fields_by_root: dict, group: str, world: int, *,
                  expert_field_of, replicated_fields: tuple
                  ) -> ShardPlan:
    """Expert-sharded OPTIMIZER STATE (NOT expert parallelism —
    resident stays ALL_RANKS): fields for which ``expert_field_of``
    returns an expert index are assigned owner = index % world;
    ``replicated_fields`` (router/norms) go ALL_RANKS; everything else
    falls back to zero1-style bucketing."""
    assignments = []
    for root, infos in fields_by_root.items():
        rest = []
        for f in infos:
            e = expert_field_of(f.name)
            if e is not None:
                assignments.append(
                    Assignment(Region(root, f.name), e % world))
            elif f.name in replicated_fields:
                assignments.append(
                    Assignment(Region(root, f.name), ALL_RANKS))
            else:
                rest.append(f)
        if rest:
            sub = zero1_halves({root: rest}, group, world)
            assignments.extend(sub.assignments)
    return ShardPlan(group=group, world=world,
                     assignments=tuple(assignments),
                     fields_by_root=fields_by_root)


def layer_fields_by_root(cfg) -> dict:
    """Conductor helper: {object_root -> [FieldInfo]} from a family's
    layouts (llama3-shaped families; extend per family as sharding
    consumers grow)."""
    from dataflow.training.models.llama3 import family_layouts

    dims, fl = family_layouts(cfg)
    out = {}
    for i, layer in enumerate(fl.layers):
        out[f"W_{i}"] = field_infos(layer.weights)
    out["W_embed"] = field_infos(fl.embed)
    out["W_head"] = field_infos(fl.head)
    return out
