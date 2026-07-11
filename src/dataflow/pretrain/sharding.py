"""The sharding API: conductor-owned ownership algebra over packed
layouts, group-scoped by construction.

Vocabulary (plan doc, Z0/Z0a): a ShardPlan assigns REGIONS —
(object_root, field, optional dim-0 row range) — to an UPDATER (a
rank of the plan's group, or ALL_RANKS = every rank updates
redundantly). ``resident`` is fixed at ALL_RANKS in v1 (weights live
everywhere; true expert parallelism narrows it later — the v1
consumer REJECTS such plans loudly).

Ownership is the vocabulary; consumers decide meaning. The v1
consumer is the optimizer configuration: updater=r means rank r holds
the region's optimizer state and computes its update; everyone else
holds ZERO optimizer bytes for it and receives the updated params via
the propagation collective. updater=ALL_RANKS means redundant updates
with full state everywhere and no propagation traffic (the cheap
choice for tiny fields).

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
    rows: tuple | None = None      # (lo, hi) dim-0 slice; None = whole

    def key(self) -> tuple:
        return (self.object_root, self.field,
                self.rows if self.rows else None)


@dataclass(frozen=True)
class Assignment:
    region: Region
    updater: object                # rank int | ALL_RANKS
    resident: object = ALL_RANKS   # v1: always ALL_RANKS


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
        return [a.region for a in self.assignments if a.updater == rank]

    def redundant(self) -> list:
        return [a.region for a in self.assignments
                if a.updater == ALL_RANKS]

    def resident(self, rank: int) -> list:
        return [a.region for a in self.assignments
                if a.resident == ALL_RANKS or a.resident == rank
                or (isinstance(a.resident, (set, frozenset, tuple))
                    and rank in a.resident)]

    def updater_of(self, root: str, field_name: str):
        for a in self.assignments:
            if (a.region.object_root == root
                    and a.region.field == field_name):
                return a.updater
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
                and (a.updater == rank or a.updater == ALL_RANKS)]

    def sharded_assignments(self, root: str) -> list:
        return [a for a in self.assignments
                if a.region.object_root == root
                and a.updater != ALL_RANKS]

    def v1_consumable(self) -> None:
        for a in self.assignments:
            if a.resident != ALL_RANKS:
                raise ValueError(
                    f"plan region {a.region.key()} narrows residency "
                    f"({a.resident!r}) — true parameter placement "
                    f"(expert parallelism) is not executable by the "
                    f"v1 optimizer-sharding consumer")

    # ------------------------------------------------ realizability
    def owned_ranges(self, rank: int, root: str) -> list:
        """Byte ranges of root's packed buffer owned by rank (rows are
        contiguous in row-major layouts)."""
        infos = {f.name: f for f in self.fields_by_root[root]}
        ranges = []
        for a in self.sharded_assignments(root):
            if a.updater != rank:
                continue
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
            cover: dict = {f.name: [] for f in infos}
            for a in self.assignments:
                if a.region.object_root != root:
                    continue
                fi = next((f for f in infos
                           if f.name == a.region.field), None)
                if fi is None:
                    raise ValueError(f"{root}: unknown field "
                                     f"{a.region.field!r}")
                if a.region.rows is None:
                    cover[fi.name].append((0, fi.rows_total()))
                else:
                    lo, hi = a.region.rows
                    if not (0 <= lo < hi <= fi.rows_total()):
                        raise ValueError(
                            f"{root}.{fi.name}: rows {a.region.rows} "
                            f"out of bounds (0, {fi.rows_total()})")
                    if op is not None and a.updater != ALL_RANKS:
                        rule = op.for_field(fi.name, None, fi.shape)
                        if rule == "muon":
                            raise ValueError(
                                f"{root}.{fi.name}: muon-ruled fields "
                                f"are whole-matrix units — row splits "
                                f"break Newton-Schulz")
                    cover[fi.name].append((lo, hi))
            for name, spans in cover.items():
                spans.sort()
                pos = 0
                for lo, hi in spans:
                    if lo != pos:
                        raise ValueError(
                            f"{root}.{name}: cover gap/overlap at row "
                            f"{pos} (next span starts {lo})")
                    pos = hi
                total = next(f.rows_total() for f in infos
                             if f.name == name)
                if pos != total:
                    raise ValueError(f"{root}.{name}: covered to row "
                                     f"{pos} of {total}")

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
                     "updater": a.updater}
                    for a in self.assignments]}

    @staticmethod
    def from_dict(d: dict, fields_by_root: dict) -> "ShardPlan":
        assigns = tuple(
            Assignment(Region(x["root"], x["field"],
                              tuple(x["rows"]) if x["rows"] else None),
                       x["updater"])
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


def zero1_halves(fields_by_root: dict, group: str, world: int,
                 replicate_below_bytes: int = 1 << 16) -> ShardPlan:
    """The v1 builder: per root, greedy field-snapped byte buckets —
    approximately equal 1/world shards, whole fields only. Fields
    smaller than ``replicate_below_bytes`` (norms, biases) go
    updater=ALL_RANKS: redundant update is cheaper than propagating
    them."""
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
                if rank < world - 1 and acc >= target * (rank + 1):
                    rank += 1
                assignments.append(Assignment(Region(root, f.name),
                                              rank))
                acc += f.nbytes
        for f in small:
            assignments.append(Assignment(Region(root, f.name),
                                          ALL_RANKS))
    return ShardPlan(group=group, world=world,
                     assignments=tuple(assignments),
                     fields_by_root=fields_by_root)


def expert_shards(fields_by_root: dict, group: str, world: int, *,
                  expert_field_of, replicated_fields: tuple
                  ) -> ShardPlan:
    """Expert-sharded OPTIMIZER STATE (NOT expert parallelism —
    resident stays ALL_RANKS): fields for which ``expert_field_of``
    returns an expert index are assigned updater = index % world;
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
