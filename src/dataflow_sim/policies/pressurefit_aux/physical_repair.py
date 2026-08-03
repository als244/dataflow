"""Structured simulator-capacity repair support for PressureFit.

The analytic boundary model is intentionally conservative but not a complete
simulation.  If physical replay finds a capacity contradiction, the simulator
reports typed context and this module translates it into additional pressure
at the task-start boundary.  Human error wording is never part of the API.
"""
from __future__ import annotations

from dataflow_sim.engine.errors import SimulationCapacityError
from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _modeled_boundary_need,
)
from dataflow_sim.policies.pressurefit_aux.types import _ResidencySpan
from dataflow_sim.core.schema import TaskChain

_PHYSICAL_REPAIR_LIMIT = 12


def _apply_physical_repair(
    error: ValueError,
    bare: TaskChain,
    facts: _Facts,
    intervals: dict[str, list[_ResidencySpan]],
    extra_pressure: list[int],
) -> bool:
    """Convert one structured simulator failure into boundary pressure."""
    physical = _physical_pressure_from_error(error, facts, bare)
    if physical is None:
        return False
    boundary_idx, observed_need, observed_overage = physical
    modeled_need = _modeled_boundary_need(facts, intervals, boundary_idx)
    required_extra = observed_need - modeled_need
    if required_extra <= extra_pressure[boundary_idx]:
        required_extra = extra_pressure[boundary_idx] + observed_overage
    extra_pressure[boundary_idx] = max(1, required_extra)
    return True


def _physical_pressure_from_error(
    error: ValueError,
    facts: _Facts,
    bare: TaskChain,
) -> tuple[int, int, int] | None:
    """Return ``(boundary index, actual bytes, overage)`` when repairable."""
    if not isinstance(error, SimulationCapacityError):
        return None
    if error.location != "fast" or bare.fast_memory_capacity is None:
        return None
    boundary_idx = facts.task_index.get(error.task_id)
    if boundary_idx is None:
        return None

    actual_need = error.actual_need_bytes
    overage = error.overage_bytes
    if actual_need is None:
        missing_bytes = sum(
            facts.sizes.get(oid, 0) for oid in error.missing_object_ids
        )
        if missing_bytes <= 0:
            return None
        actual_need = bare.fast_memory_capacity + missing_bytes
        overage = missing_bytes
    if overage is None:
        overage = actual_need - bare.fast_memory_capacity
    return boundary_idx, actual_need, max(1, overage)
