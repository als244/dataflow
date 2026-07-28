"""checkpoint_record.json: read, write, validate.

A checkpoint is complete exactly when its record exists: writers land
their snapshot dirs first, the composer writes the record LAST,
atomically. The record is validated BEFORE it is written and AFTER it
is read — an invalid record never lands on disk and never leaves this
module. Refusals are total and loud, naming the offending object,
range, field or writers.

The record is workload-blind: logical objects are named byte spans
with optional field-schema digests; slices map snapshot bytes into
them; scheme, client_payload, summary and launch are opaque caller
state, stored and returned, never interpreted.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

RECORD_SCHEMA = "dataflow-checkpoint/v1"
RECORD_NAME = "checkpoint_record.json"

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1,
               "int32": 4, "int64": 8, "int8": 1, "uint8": 1}


class CheckpointError(RuntimeError):
    """Loud refusal from the record layer: incomplete checkpoints,
    foreign schemas, coverage holes, ambiguous overlap, replication
    drift, misaligned slices — always naming the offender."""


def schema_digest(fields: list) -> str:
    """blake2b-16 over a field schema's canonical JSON — what the
    record stores per logical object (the full schema rides the
    snapshots' object meta)."""
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


def write_record(dest, *, step, seed, logical_objects, slices,
                 snapshots, engine_spec, scheme=None,
                 client_payload=None, summary=None, launch=None,
                 field_schemas=None) -> dict:
    """Validate, then land the record atomically (tmp + rename). The
    caller composes this AFTER every snapshot has landed — the
    record's presence is the completeness marker."""
    doc = {
        "schema": RECORD_SCHEMA,
        "step": int(step), "seed": int(seed),
        "created_t": time.time(),
        "scheme": scheme or {},
        "logical_objects": logical_objects,
        "snapshots": snapshots,
        "slices": slices,
        "engine_spec": engine_spec,
        "launch": launch or {},
        "client_payload": client_payload or {},
        "summary": summary or {},
    }
    validate_record(doc, field_schemas=field_schemas)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / (RECORD_NAME + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.rename(dest / RECORD_NAME)
    return doc


def read_record(record_dir) -> dict:
    path = Path(record_dir) / RECORD_NAME
    if not path.is_file():
        raise CheckpointError(
            f"no {RECORD_NAME} under {record_dir} — the record is "
            f"written LAST, so this checkpoint is incomplete")
    doc = json.loads(path.read_text())
    if doc.get("schema") != RECORD_SCHEMA:
        raise CheckpointError(
            f"record schema {doc.get('schema')!r} != {RECORD_SCHEMA!r}")
    validate_record(doc)
    return doc


def validate_record(record: dict, *, field_schemas=None) -> None:
    """Total validation: every rule loud, every refusal named.

    1. schema guard;
    2. slice references stay inside the snapshots list, and each
       referenced snapshot's object inventory carries the slice's
       object at a size containing its snapshot_range;
    3. per logical object: ranges in bounds and length-consistent;
    4. completeness — object_ranges union to the full byte span;
    5. overlap legal ONLY with a winner: any multiply-covered byte
       needs an authoritative slice, and authoritative twins must be
       identical spans with EQUAL hashes (the replication drift
       certificate);
    6. with ``field_schemas``: schema digests match the record and
       every slice endpoint falls on an element boundary of its
       containing field.
    """
    if record.get("schema") != RECORD_SCHEMA:
        raise CheckpointError(
            f"record schema {record.get('schema')!r} != "
            f"{RECORD_SCHEMA!r}")
    snapshots = record["snapshots"]
    logical = record["logical_objects"]
    slices = record["slices"]

    for lid in slices:
        if lid not in logical:
            raise CheckpointError(
                f"slices for unknown logical object {lid}")
    for lid, spec in logical.items():
        total = int(spec["bytes"])
        entries = slices.get(lid, [])
        check_entry_shapes(lid, total, entries, snapshots)
        check_coverage(lid, total, entries)
        check_overlap(lid, entries, snapshots)
        check_replica_hashes(lid, entries, snapshots)
        if field_schemas and lid in field_schemas:
            check_digest(lid, spec, field_schemas[lid])
            check_alignment(lid, entries, field_schemas[lid])


def check_entry_shapes(lid, total, entries, snapshots) -> None:
    for e in entries:
        snap = e["snapshot"]
        if not (0 <= int(snap) < len(snapshots)):
            raise CheckpointError(
                f"{lid}: slice names snapshot {snap} but the record "
                f"lists {len(snapshots)}")
        src, dst = e["snapshot_range"], e["object_range"]
        if dst[1] - dst[0] != src[1] - src[0]:
            raise CheckpointError(
                f"{lid}: snapshot_range {src} length != "
                f"object_range {dst} length")
        if not (0 <= dst[0] < dst[1] <= total):
            raise CheckpointError(
                f"{lid}: object_range {dst} outside [0, {total})")
        bare = lid.rsplit("@", 1)[0] if "@" in lid else lid
        resident = (snapshots[int(snap)].get("objects") or {}).get(bare)
        if resident is None:
            raise CheckpointError(
                f"{lid}: snapshot {snap} lists no resident size for "
                f"{bare} — the rank view cannot recreate local "
                f"geometry")
        if not (0 <= src[0] < src[1] <= int(resident)):
            raise CheckpointError(
                f"{lid}: snapshot_range {src} outside the resident "
                f"object [0, {resident})")


def check_coverage(lid, total, entries) -> None:
    gaps = coverage_gaps(total, entries)
    if gaps:
        named = ", ".join(f"[{a}, {b})" for a, b in gaps)
        raise CheckpointError(
            f"{lid}: slices do not cover the object — missing {named}")


def coverage_gaps(total, entries) -> list:
    spans = sorted(tuple(e["object_range"]) for e in entries)
    gaps = []
    cursor = 0
    for lo, hi in spans:
        if lo > cursor:
            gaps.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < total:
        gaps.append((cursor, total))
    return gaps


def check_overlap(lid, entries, snapshots) -> None:
    """Sweep the elementary intervals; multiply-covered bytes need an
    authoritative winner, and authoritative twins must be
    identical-span with equal hashes."""
    bounds = sorted({x for e in entries for x in e["object_range"]})
    for lo, hi in zip(bounds, bounds[1:]):
        covering = [e for e in entries
                    if e["object_range"][0] <= lo
                    and hi <= e["object_range"][1]]
        if len(covering) < 2:
            continue
        auth = [e for e in covering if e.get("authoritative")]
        if not auth:
            raise CheckpointError(
                f"{lid}: bytes [{lo}, {hi}) covered by "
                f"{len(covering)} slices with no authoritative "
                f"winner")
        first = auth[0]
        for other in auth[1:]:
            if tuple(other["object_range"]) != \
                    tuple(first["object_range"]):
                raise CheckpointError(
                    f"{lid}: authoritative slices overlap with "
                    f"different spans {first['object_range']} vs "
                    f"{other['object_range']}")
            if other.get("hash") != first.get("hash"):
                writers = (snapshots[first["snapshot"]].get("writer"),
                           snapshots[other["snapshot"]].get("writer"))
                raise CheckpointError(
                    f"{lid}: replication drift — authoritative "
                    f"copies from writers {writers[0]} and "
                    f"{writers[1]} carry different hashes")


def check_replica_hashes(lid, entries, snapshots) -> None:
    """Identical-span slices are replicas by construction and must
    hash-equal whichever carries the authoritative flag — the flag
    picks the restore winner, equality certifies interchangeability
    (the replication drift certificate)."""
    by_span: dict = {}
    for e in entries:
        by_span.setdefault(tuple(e["object_range"]), []).append(e)
    for span, twins in by_span.items():
        first = twins[0]
        for other in twins[1:]:
            if other.get("hash") != first.get("hash"):
                a = snapshots[first["snapshot"]].get("writer")
                b = snapshots[other["snapshot"]].get("writer")
                raise CheckpointError(
                    f"{lid}: replication drift — copies of "
                    f"{list(span)} from writers {a} and {b} carry "
                    f"different hashes")


def check_digest(lid, spec, fields) -> None:
    want = spec.get("schema_digest")
    have = schema_digest(fields)
    if want != have:
        raise CheckpointError(
            f"{lid}: schema digest {want!r} does not match the "
            f"field schema ({have})")


def check_alignment(lid, entries, fields) -> None:
    for e in entries:
        for endpoint in e["object_range"]:
            check_element_boundary(lid, endpoint, fields)


def check_element_boundary(lid, offset, fields) -> None:
    for f in fields:
        lo = int(f["offset_bytes"])
        hi = lo + int(f["size_bytes"])
        if lo <= offset <= hi:
            esize = DTYPE_BYTES[f["dtype"]]
            if (offset - lo) % esize:
                raise CheckpointError(
                    f"{lid}: slice endpoint {offset} falls inside an "
                    f"element of field {f['name']} ({f['dtype']}) — "
                    f"endpoints must land on element boundaries")
            return
    raise CheckpointError(
        f"{lid}: slice endpoint {offset} lies outside every field "
        f"of the schema")
