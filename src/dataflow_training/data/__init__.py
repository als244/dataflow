"""Data plane: fineweb streaming and the general sequence-packing
primitive (torch-free; the reference side vendors a byte-identical
copy)."""

from .packing import (  # noqa: F401,E402
    IGNORE_INDEX,
    PackedRound,
    PackedStep,
    pack_batch,
)

__all__ = ["IGNORE_INDEX", "PackedRound", "PackedStep", "pack_batch"]
