"""Shared scalar types for the program IR.

This module (and everything in ``dataflow.core``) must import nothing heavier
than the standard library.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Location = Literal["fast", "backing"]
LOCATIONS: tuple[str, ...] = ("fast", "backing")

# Object roles are semantic hints for tooling/visualization; the runtime and
# planners never branch on them.
Role = Literal[
    "parameter",
    "gradient",
    "optimizer_state",
    "activation",
    "input",
    "output",
    "temp",
    "other",
]
ROLES: tuple[str, ...] = (
    "parameter",
    "gradient",
    "optimizer_state",
    "activation",
    "input",
    "output",
    "temp",
    "other",
)

# Bits per element for supported dtypes. Bits (not bytes) so sub-byte formats
# stay representable.
DTYPE_BITS: dict[str, int] = {
    "fp64": 64,
    "fp32": 32,
    "tf32": 32,
    "fp16": 16,
    "bf16": 16,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "fp4": 4,
    "int64": 64,
    "int32": 32,
    "int16": 16,
    "int8": 8,
    "uint8": 8,
    "bool": 8,
}


def dtype_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    """Exact byte size of a dense tensor, rounded up to whole bytes."""
    bits = DTYPE_BITS.get(dtype)
    if bits is None:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BITS)}")
    n = math.prod(shape)
    return (n * bits + 7) // 8


@dataclass(frozen=True)
class TensorMeta:
    """Optional tensor interpretation of an object's bytes.

    Objects without tensor semantics (opaque blobs, packed contexts with
    internal layout) may omit any field. When both ``shape`` and ``dtype`` are
    present, validation asserts the dense size matches the object's
    ``size_bytes`` — sizes are exact by construction, never estimated.
    """

    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    strides: tuple[int, ...] | None = None

    def nbytes(self) -> int | None:
        if self.dtype is None or self.shape is None:
            return None
        if self.strides is not None:
            return None  # non-dense layouts are not size-checked
        return dtype_nbytes(self.shape, self.dtype)
