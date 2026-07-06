"""Bit-identical lowering tripwire: generalizing the shared chain builder
(heterogeneous layer kinds, tied embeddings) must NOT change what existing
families emit — same ids, same order, same sizes, same directives-bare
structure. A legitimate lowering change updates these constants in the same
commit, deliberately."""
import hashlib
import json
from dataclasses import replace

from dataflow.core.jsonio import program_to_dict
from dataflow.training.llama3 import ShapedLlamaConfig, lower_llama3
from dataflow.training.olmoe import ShapedOlmoeConfig, lower_olmoe
from dataflow.training.qwen3 import ShapedQwen3Config, lower_qwen3
from dataflow.training.qwen35 import ShapedQwen35Config, lower_qwen35
from dataflow.training.qwen35moe import ShapedQwen35MoeConfig, lower_qwen35moe

# Constants last updated DELIBERATELY for the fused head_loss lowering
# (head_fwd/loss_bwd/head_bwd -> ONE token-chunked task; logits/dlogits
# objects removed from the grammar). olmoe rows ADDED with the MoE family
# (existing constants verified unchanged in the same commit).
EXPECTED = {
    "llama3-tiny-ga2-s2": "77805909b52d6959",
    "llama3-tiny-tail": "6eb2fdd93c7fd576",
    "qwen3-tiny-ga3": "13c4203931442fd2",
    # qwen35: heterogeneous kinds + both embedding modes
    "qwen35-tiny-ga2": "860de9f8f7b91d4a",
    "qwen35-tiny-tied": "ecd500539ee2e49d",
    # olmoe: first MoE family (moeattn kind, untied)
    "olmoe-tiny": "27b16815cf642d0a",
    "olmoe-tiny-ga2": "eef05ae0081b6dfc",
    # qwen35moe: hybrid MoE (linmoe/gattnmoe kinds + shared expert, untied)
    "qwen35moe-tiny-ga2": "a44e9cf9734a5da7",
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
        "qwen35-tiny-ga2": _hash(
            lower_qwen35(replace(ShapedQwen35Config.tiny(), grad_accum_rounds=2))
        ),
        "qwen35-tiny-tied": _hash(lower_qwen35(ShapedQwen35Config.tiny_tied())),
        "olmoe-tiny": _hash(lower_olmoe(ShapedOlmoeConfig.tiny())),
        "olmoe-tiny-ga2": _hash(
            lower_olmoe(replace(ShapedOlmoeConfig.tiny(), grad_accum_rounds=2))
        ),
        "qwen35moe-tiny-ga2": _hash(
            lower_qwen35moe(replace(ShapedQwen35MoeConfig.tiny(), grad_accum_rounds=2))
        ),
    }
    assert got == EXPECTED, {k: (got[k], EXPECTED[k]) for k in got if got[k] != EXPECTED[k]}
