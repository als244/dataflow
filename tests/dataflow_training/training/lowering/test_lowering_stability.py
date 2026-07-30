"""Bit-identical lowering tripwire: generalizing the shared chain builder
(heterogeneous layer kinds, tied embeddings) must NOT change what existing
families emit — same ids, same order, same sizes, same directives-bare
structure. A legitimate lowering change updates these constants in the same
commit, deliberately.

Tests:
- test_lowered_programs_bit_identical: every family/config variant's lowered program hashes to its pinned constant.
"""
import hashlib
import json
from dataclasses import replace

from dataflow.core.jsonio import program_to_dict
from dataflow_training.lowering.shaped_program import hardware_preset
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, lower_llama3
from dataflow_training.model_families.olmoe import ShapedOlmoeConfig, lower_olmoe
from dataflow_training.model_families.qwen3 import ShapedQwen3Config, lower_qwen3
from dataflow_training.model_families.qwen35 import ShapedQwen35Config, lower_qwen35
from dataflow_training.model_families.qwen35moe import ShapedQwen35MoeConfig, lower_qwen35moe
from dataflow_training.model_families.qwen3moe import ShapedQwen3MoeConfig, lower_qwen3moe
from dataflow_training.model_families.dsv3 import ShapedDsv3Config, lower_dsv3
from dataflow_training.model_families.dsv32 import ShapedDsv32Config, lower_dsv32
from dataflow_training.model_families.glm52 import ShapedGlm52Config, lower_glm52

# Constants last updated DELIBERATELY when initial objects gained the
# persistent marker: the field that makes checkpoint selection a
# property of the program rather than a name convention. Every
# family's digest moves, because every family marks its parameters,
# optimizer state and cross-step accumulators. The change was proved
# additive rather than semantic — stripping persistent from a new
# program's dict reproduces its previous digest exactly — so what
# moved is the metadata, not what any family lowers to.
EXPECTED = {
    "llama3-tiny-ga2-s2": "8826c0f91245927d",
    "llama3-tiny-tail": "6cce38e17646a28b",
    "qwen3-tiny-ga3": "591e4ed8245472ef",
    "qwen35-tiny-ga2": "3a0da0d6526d6531",
    "qwen35-tiny-tied": "098204ce403f4a4e",
    "olmoe-tiny": "04891ca13e918046",
    "olmoe-tiny-ga2": "0b35b3a84d65d902",
    "qwen35moe-tiny-ga2": "ebe52dd5033707d7",
    "qwen3moe-tiny": "830638022f968de8",
    "qwen3moe-tiny-ga2": "105626bd80c94c00",
    "dsv3-tiny": "27b7155074437df1",
    "dsv3-tiny-ga2": "452b4839b187e0a6",
    "dsv32-tiny": "689e3fc7e66bd169",
    "dsv32-tiny-ga2": "f5f8454169423702",
    "dsv32-tiny-dense": "7417250fe66159d8",
    "glm52-tiny": "c469d83c78d6d7d0",
    "glm52-tiny-ga2": "9942568902349a75",
    "glm52-tiny-warmup": "66853616d6b780f7",
}


# fixture pricing: the pinned digests must be machine-independent, so
# every lowering below prices with an explicitly NAMED preset (equal to
# the historical defaults, so the committed constants hold). Never a
# resolved device here — content stability is the contract.
HW = hardware_preset("rtx 5090")


def _hash(program) -> str:
    return hashlib.sha256(
        json.dumps(program_to_dict(program), sort_keys=True).encode()
    ).hexdigest()[:16]


def test_lowered_programs_bit_identical():
    got = {
        "llama3-tiny-ga2-s2": _hash(
            lower_llama3(replace(ShapedLlamaConfig.tiny(), grad_accum_rounds=2, num_steps=2), hw=HW)
        ),
        "llama3-tiny-tail": _hash(
            lower_llama3(replace(ShapedLlamaConfig.tiny(), optimizer_placement="tail"), hw=HW)
        ),
        "qwen3-tiny-ga3": _hash(
            lower_qwen3(replace(ShapedQwen3Config.tiny(), grad_accum_rounds=3), hw=HW)
        ),
        "qwen35-tiny-ga2": _hash(
            lower_qwen35(replace(ShapedQwen35Config.tiny(), grad_accum_rounds=2), hw=HW)
        ),
        "qwen35-tiny-tied": _hash(lower_qwen35(ShapedQwen35Config.tiny_tied(), hw=HW)),
        "olmoe-tiny": _hash(lower_olmoe(ShapedOlmoeConfig.tiny(), hw=HW)),
        "olmoe-tiny-ga2": _hash(
            lower_olmoe(replace(ShapedOlmoeConfig.tiny(), grad_accum_rounds=2), hw=HW)
        ),
        "qwen35moe-tiny-ga2": _hash(
            lower_qwen35moe(replace(ShapedQwen35MoeConfig.tiny(), grad_accum_rounds=2), hw=HW)
        ),
        "qwen3moe-tiny": _hash(lower_qwen3moe(ShapedQwen3MoeConfig.tiny(), hw=HW)),
        "qwen3moe-tiny-ga2": _hash(
            lower_qwen3moe(replace(ShapedQwen3MoeConfig.tiny(), grad_accum_rounds=2), hw=HW)
        ),
        "dsv3-tiny": _hash(lower_dsv3(ShapedDsv3Config.tiny(), hw=HW)),
        "dsv3-tiny-ga2": _hash(
            lower_dsv3(replace(ShapedDsv3Config.tiny(), grad_accum_rounds=2), hw=HW)
        ),
        "dsv32-tiny": _hash(lower_dsv32(ShapedDsv32Config.tiny(), hw=HW)),
        "dsv32-tiny-ga2": _hash(
            lower_dsv32(replace(ShapedDsv32Config.tiny(), grad_accum_rounds=2), hw=HW)
        ),
        "dsv32-tiny-dense": _hash(
            lower_dsv32(replace(ShapedDsv32Config.tiny(), sparse_mode=False), hw=HW)
        ),
        "glm52-tiny": _hash(lower_glm52(ShapedGlm52Config.tiny(), hw=HW)),
        "glm52-tiny-ga2": _hash(
            lower_glm52(replace(ShapedGlm52Config.tiny(), grad_accum_rounds=2), hw=HW)
        ),
        "glm52-tiny-warmup": _hash(
            lower_glm52(replace(ShapedGlm52Config.tiny(), sparse_mode=False), hw=HW)
        ),
    }
    assert got == EXPECTED, {k: (got[k], EXPECTED[k]) for k in got if got[k] != EXPECTED[k]}
