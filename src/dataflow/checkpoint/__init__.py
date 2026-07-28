"""Workload-blind checkpoint records over engine snapshots.

Logical objects (named byte spans with field-schema digests) +
slices (snapshot bytes mapped into them) + engine specs + opaque
caller state, all in one atomically-last checkpoint_record.json —
the record IS the contract (docs/checkpointing.md). This package
validates records totally, refuses loudly with named offenders, and
resolves target sets into per-snapshot fetch plans in the engine's
remap wire shape. It imports nothing from any workload.

One namespace rule: a logical id may be SOURCE-QUALIFIED
(``Aux_0@1``) — per-source state that must never certify as
replicated. The rank view resolves a bare target to the source's
qualified object and restores it under the bare local name; the
logical view lists qualified objects as they are."""
from .compose import save_checkpoint
from .record import (CheckpointError, RECORD_NAME, RECORD_SCHEMA,
                     read_record, schema_digest, validate_record,
                     write_record)
from .targets import (persisted_objects, resolve_targets,
                      source_resident_bytes)

__all__ = ["CheckpointError", "RECORD_NAME", "RECORD_SCHEMA",
           "persisted_objects", "read_record", "resolve_targets",
           "save_checkpoint", "schema_digest", "validate_record",
           "source_resident_bytes", "write_record"]
