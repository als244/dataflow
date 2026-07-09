"""Isolated ground-truth PyTorch models for the pretraining parity study.

Every module here is plain ``torch`` + autograd and imports NOTHING from
``dataflow`` — and nothing from its sibling modules either: each family is a
COMPLETE, SELF-CONTAINED reference (shared primitives like RMSNorm / RoPE /
SwiGLU / MoE routing / MLA / DSA are reimplemented in each file, redundantly
and on purpose). These are the black-box references the engine is validated
against (a byte-identical init + the same data stream must produce the same
loss curve), kept out of ``src/dataflow`` so they can never accidentally
share code with the runtime.

The six MoE families (olmoe, qwen3moe, qwen35moe, dsv3, dsv32, glm52) expose
an OPTIONAL load-balancing auxiliary loss via ``loss(..., aux_coef=0.0)``.
"""
from .dsv3 import Dsv3, Dsv3Config
from .dsv32 import Dsv32, Dsv32Config
from .glm52 import Glm52, Glm52Config
from .llama3 import Llama3, Llama3Config
from .olmoe import Olmoe, OlmoeConfig
from .qwen3 import Qwen3, Qwen3Config
from .qwen35 import Qwen35, Qwen35Config
from .qwen35moe import Qwen35Moe, Qwen35MoeConfig
from .qwen3moe import Qwen3Moe, Qwen3MoeConfig

__all__ = [
    "Llama3", "Llama3Config",
    "Qwen3", "Qwen3Config",
    "Qwen35", "Qwen35Config",
    "Qwen3Moe", "Qwen3MoeConfig",
    "Qwen35Moe", "Qwen35MoeConfig",
    "Olmoe", "OlmoeConfig",
    "Dsv3", "Dsv3Config",
    "Dsv32", "Dsv32Config",
    "Glm52", "Glm52Config",
]
