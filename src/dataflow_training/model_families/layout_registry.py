"""The named layout registry: every family's packed layouts as KEYED,
VALIDATED, ADDRESSABLE artifacts — the coordinate system the sharding
and responsibility layers speak.

Keys are ``"<family>/<kind>"`` for layer kinds plus the
``"<family>/embed"`` / ``"<family>/head"`` pseudo-kinds. Each entry
exposes the layout's FIELD TABLE as data (name, shape, dtype, offset,
size) plus the kind's production coupling (weight field -> the saved
activation it produces, the PER-KIND view of
``FamilyLayouts.activation_of_weight``).

The registry is DERIVED — it projects the family registry's
``family_layouts`` output; there is no second source of truth. The
contract validator makes "well-defined" machine-checkable: unique
field names, monotone non-overlapping offsets, positive sizes, and
coupling references that exist in the kind. ``layouts_digest`` pins
the whole surface byte-identically (init byte order rides field
order — a certified surface; the digest gate is what lets builder
internals migrate to declarative form incrementally without risk).
"""
import hashlib
import json
from dataclasses import dataclass

from dataflow_training.model_families.families import resolve_family


@dataclass(frozen=True)
class RegisteredLayout:
    """One keyed layout: the field table as data."""

    key: str                      # "<family>/<kind>"
    family: str
    kind: str
    fields: tuple                 # (name, shape, dtype, offset_bytes, size_bytes)
    total_bytes: int
    coupling: dict                # weight field -> produced activation field
    # which object roots this layout backs (layer kinds list layer
    # indices; embed/head list their single root)
    roots: tuple


def _field_table(layout) -> tuple:
    return tuple(
        (f.name, tuple(f.shape), f.dtype, f.offset_bytes, f.nbytes)
        for f in layout.fields)


def registered_layouts(cfg) -> dict:
    """{key: RegisteredLayout} for cfg's family — layer kinds (weights;
    the kind's activation table rides the same entry), embed, head."""
    fam = resolve_family(cfg)
    dims, fl = fam.family_layouts(cfg)
    coupling_all = fl.activation_of_weight or {}
    out = {}
    by_kind = {}
    for i, ll in enumerate(fl.layers):
        by_kind.setdefault(ll.kind, []).append((i, ll))
    for kind, entries in by_kind.items():
        i0, ll = entries[0]
        wnames = {f.name for f in ll.weights.fields}
        coupling = {w: a for w, a in coupling_all.items() if w in wnames}
        out[f"{fam.name}/{kind}"] = RegisteredLayout(
            key=f"{fam.name}/{kind}", family=fam.name, kind=kind,
            fields=_field_table(ll.weights),
            total_bytes=ll.weights.total_bytes,
            coupling=coupling,
            roots=tuple(f"W_{i}" for i, _ in entries),
        )
    out[f"{fam.name}/embed"] = RegisteredLayout(
        key=f"{fam.name}/embed", family=fam.name, kind="embed",
        fields=_field_table(fl.embed), total_bytes=fl.embed.total_bytes,
        coupling={}, roots=("W_embed",),
    )
    out[f"{fam.name}/head"] = RegisteredLayout(
        key=f"{fam.name}/head", family=fam.name, kind="head",
        fields=_field_table(fl.head), total_bytes=fl.head.total_bytes,
        coupling={}, roots=("W_head",),
    )
    return out


def validate_layouts(cfg) -> list:
    """The strong contract, machine-checked. Returns problem strings
    (empty = conforming): unique field names, monotone non-overlapping
    offsets within each layout, positive sizes, total covers the last
    field, coupling references resolve within the kind."""
    problems = []
    fam = resolve_family(cfg)
    _, fl = fam.family_layouts(cfg)
    act_names_by_kind = {}
    for ll in fl.layers:
        act_names_by_kind.setdefault(
            ll.kind, {f.name for f in ll.activations.fields})
    for key, rl in registered_layouts(cfg).items():
        names = [f[0] for f in rl.fields]
        if len(names) != len(set(names)):
            problems.append(f"{key}: duplicate field names")
        prev_end = 0
        for name, shape, dtype, off, size in rl.fields:
            if size <= 0:
                problems.append(f"{key}.{name}: size {size}")
            if off < prev_end:
                problems.append(
                    f"{key}.{name}: offset {off} overlaps previous "
                    f"end {prev_end}")
            prev_end = off + size
        if prev_end > rl.total_bytes:
            problems.append(
                f"{key}: fields end {prev_end} past total "
                f"{rl.total_bytes}")
        acts = act_names_by_kind.get(rl.kind, set())
        for w, a in rl.coupling.items():
            if a not in acts:
                problems.append(
                    f"{key}: coupling {w}->{a} names an activation "
                    f"absent from the kind")
    return problems


def layouts_digest(cfg) -> str:
    """sha256 over the family's whole registered surface — field
    tables, offsets, couplings, roots. Byte-order stability is a
    CERTIFIED property (init draws ride field order); this digest is
    the gate that lets layout construction migrate incrementally."""
    reg = registered_layouts(cfg)
    blob = json.dumps(
        {k: {"fields": reg[k].fields, "total": reg[k].total_bytes,
             "coupling": sorted(reg[k].coupling.items()),
             "roots": reg[k].roots}
         for k in sorted(reg)},
        sort_keys=True, default=list)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
