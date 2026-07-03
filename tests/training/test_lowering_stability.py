"""Bit-identical lowering tripwire: generalizing the shared chain builder
(heterogeneous layer kinds, tied embeddings) must NOT change what existing
families emit — same ids, same order, same sizes, same directives-bare
structure. A legitimate lowering change updates these constants in the same
commit, deliberately."""
import hashlib
import json
from dataclasses import replace

from dataflow.core.jsonio import program_to_dict
from dataflow.training.llama3_lowering import lower_llama3
from dataflow.training.qwen3_lowering import lower_qwen3
from dataflow.training.shaped_llama3 import ShapedLlamaConfig
from dataflow.training.shaped_qwen3 import ShapedQwen3Config

EXPECTED = {
    "llama3-tiny-ga2-s2": "67cbd6d2dbe6e6bd",
    "llama3-tiny-tail": "ed3e3280fc2896eb",
    "qwen3-tiny-ga3": "999478c9e65e3345",
}


def _hash(program) -> str:
    return hashlib.sha256(
        json.dumps(program_to_dict(program), sort_keys=True).encode()
    ).hexdigest()[:16]


def test_lowered_programs_bit_identical():
    got = {
        "llama3-tiny-ga2-s2": _hash(
            lower_llama3(replace(ShapedLlamaConfig.tiny(), grad_accum_rounds=2, num_steps=2))
        ),
        "llama3-tiny-tail": _hash(
            lower_llama3(replace(ShapedLlamaConfig.tiny(), optimizer_placement="tail"))
        ),
        "qwen3-tiny-ga3": _hash(
            lower_qwen3(replace(ShapedQwen3Config.tiny(), grad_accum_rounds=3))
        ),
    }
    assert got == EXPECTED, {k: (got[k], EXPECTED[k]) for k in got if got[k] != EXPECTED[k]}
