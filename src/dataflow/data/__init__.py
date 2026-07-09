"""Corpus-agnostic batching/packing utilities (torch-free).

Shared by the pretrain driver, future sft/rl drivers, and the
independent reference implementation (which VENDORS packing.py —
a checksum test pins the copies byte-identical).
"""
from .packing import (IGNORE_INDEX, PackedRound, PackedStep,
                      pack_batch)

__all__ = ["IGNORE_INDEX", "PackedRound", "PackedStep", "pack_batch"]
