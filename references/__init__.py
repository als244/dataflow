"""Isolated ground-truth PyTorch models for the pretraining parity study.

Everything in this package is plain ``torch`` + autograd and imports NOTHING
from ``dataflow``. These models are the black-box reference the engine is
validated against (a byte-identical init + the same data stream must produce
the same loss curve). They are intentionally kept out of ``src/dataflow`` so
they can never accidentally share code with the runtime.
"""
from .llama3 import Llama3, Llama3Config

__all__ = ["Llama3", "Llama3Config"]
