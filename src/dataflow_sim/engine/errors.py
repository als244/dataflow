"""Structured simulator failures consumed by planners and diagnostics."""
from __future__ import annotations


class SimulationCapacityError(ValueError):
    """Runtime capacity contradiction with machine-readable context.

    It remains a :class:`ValueError` for compatibility with existing callers,
    while planners can consume fields instead of parsing human-facing text.
    """

    __slots__ = (
        "actual_need_bytes",
        "capacity_bytes",
        "kind",
        "location",
        "missing_object_ids",
        "overage_bytes",
        "task_id",
    )

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        task_id: str,
        location: str,
        capacity_bytes: int | None,
        actual_need_bytes: int | None,
        overage_bytes: int | None,
        missing_object_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.task_id = task_id
        self.location = location
        self.capacity_bytes = capacity_bytes
        self.actual_need_bytes = actual_need_bytes
        self.overage_bytes = overage_bytes
        self.missing_object_ids = missing_object_ids
