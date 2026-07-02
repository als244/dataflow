"""Generic dataflow runtime: executes annotated programs over a DeviceBackend.

Knows nothing about DNNs, torch, or the simulator. See README.md for the
contract; see `dataflow.runtime.device` for the vendor boundary.
"""
from .engine import DeadlockError, Engine, ExecutionError, RunResult
from .executable import Executable, ExecutableResolver, SyntheticExecutable, TaskContext, synthetic_resolver
from .trace import RunTrace, compare_to_sim_eventlog

__all__ = [
    "Engine",
    "RunResult",
    "ExecutionError",
    "DeadlockError",
    "Executable",
    "ExecutableResolver",
    "SyntheticExecutable",
    "TaskContext",
    "synthetic_resolver",
    "RunTrace",
    "compare_to_sim_eventlog",
]
