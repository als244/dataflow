"""Program IR: the contract every other layer builds on.

Importing this package pulls in nothing beyond the standard library. The
simulator converters in ``dataflow.core.convert`` import ``dataflow_sim``
lazily inside their functions.
"""
from .program import (
    SCHEMA_VERSION,
    ObjectSpec,
    OutputSpec,
    Program,
    RecomputeOption,
    RecomputeRewrite,
    TaskSpec,
    TransferDirective,
)
from .types import DTYPE_BITS, TensorMeta, dtype_nbytes
from .validate import ValidationError, validate_program
from .jsonio import load_program, program_from_dict, program_to_dict, save_program

__all__ = [
    "SCHEMA_VERSION",
    "ObjectSpec",
    "OutputSpec",
    "Program",
    "RecomputeOption",
    "RecomputeRewrite",
    "TaskSpec",
    "TransferDirective",
    "TensorMeta",
    "DTYPE_BITS",
    "dtype_nbytes",
    "ValidationError",
    "validate_program",
    "load_program",
    "save_program",
    "program_from_dict",
    "program_to_dict",
]
