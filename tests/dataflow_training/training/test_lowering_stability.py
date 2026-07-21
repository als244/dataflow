"""Bit-identical lowering tripwire: generalizing the shared chain builder
(heterogeneous layer kinds, tied embeddings) must NOT change what existing
families emit — same ids, same order, same sizes, same directives-bare
structure. A legitimate lowering change updates these constants in the same
commit, deliberately."""
import hashlib
import json
from dataclasses import replace

from dataflow.core.jsonio import program_to_dict
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, lower_llama3
from dataflow_training.model_families.olmoe import ShapedOlmoeConfig, lower_olmoe
from dataflow_training.model_families.qwen3 import ShapedQwen3Config, lower_qwen3
from dataflow_training.model_families.qwen35 import ShapedQwen35Config, lower_qwen35
from dataflow_training.model_families.qwen35moe import ShapedQwen35MoeConfig, lower_qwen35moe
from dataflow_training.model_families.qwen3moe import ShapedQwen3MoeConfig, lower_qwen3moe
from dataflow_training.model_families.dsv3 import ShapedDsv3Config, lower_dsv3
from dataflow_training.model_families.dsv32 import ShapedDsv32Config, lower_dsv32
from dataflow_training.model_families.glm52 import ShapedGlm52Config, lower_glm52

# Constants last updated DELIBERATELY for the UNIVERSAL round prologue:
# every family now opens each round with prologue_round_{s}_{r} (publishes
# the round's content token count + materializes Segments), and embed_fwd
# carries the current_round edge that chains the round behind it. Every
# family's digest moves in this commit, dense ones included.
EXPECTED = {
    "llama3-tiny-ga2-s2": "d0d6c5ca89511c61",
    "llama3-tiny-tail": "474cc3736f2f335c",
    "qwen3-tiny-ga3": "a333a01170ed8ad2",
    "qwen35-tiny-ga2": "8ff6b4c8e43f4d7f",
    "qwen35-tiny-tied": "c4e3cb643c97bee2",
    "olmoe-tiny": "095800b546ccd6ec",
    "olmoe-tiny-ga2": "799788e364a6c2cf",
    "qwen35moe-tiny-ga2": "171be03fd8d0a69b",
    "qwen3moe-tiny": "36a72e02a174eb2e",
    "qwen3moe-tiny-ga2": "136cd9b787083263",
    "dsv3-tiny": "64eea2e0e216b174",
    "dsv3-tiny-ga2": "597fa49dd95425e2",
    "dsv32-tiny": "e3afd8c8563f6c0b",
    "dsv32-tiny-ga2": "a47186f9873cdf3a",
    "dsv32-tiny-dense": "80c1e7f07a08f93d",
    "glm52-tiny": "11a3ae6e7cf5cb52",
    "glm52-tiny-ga2": "5fea29c45acc7231",
    "glm52-tiny-warmup": "e9983dd1eb231d7f",
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
        "qwen3moe-tiny": _hash(lower_qwen3moe(ShapedQwen3MoeConfig.tiny())),
        "qwen3moe-tiny-ga2": _hash(
            lower_qwen3moe(replace(ShapedQwen3MoeConfig.tiny(), grad_accum_rounds=2))
        ),
        "dsv3-tiny": _hash(lower_dsv3(ShapedDsv3Config.tiny())),
        "dsv3-tiny-ga2": _hash(
            lower_dsv3(replace(ShapedDsv3Config.tiny(), grad_accum_rounds=2))
        ),
        "dsv32-tiny": _hash(lower_dsv32(ShapedDsv32Config.tiny())),
        "dsv32-tiny-ga2": _hash(
            lower_dsv32(replace(ShapedDsv32Config.tiny(), grad_accum_rounds=2))
        ),
        "dsv32-tiny-dense": _hash(
            lower_dsv32(replace(ShapedDsv32Config.tiny(), sparse_mode=False))
        ),
        "glm52-tiny": _hash(lower_glm52(ShapedGlm52Config.tiny())),
        "glm52-tiny-ga2": _hash(
            lower_glm52(replace(ShapedGlm52Config.tiny(), grad_accum_rounds=2))
        ),
        "glm52-tiny-warmup": _hash(
            lower_glm52(replace(ShapedGlm52Config.tiny(), sparse_mode=False))
        ),
    }
    assert got == EXPECTED, {k: (got[k], EXPECTED[k]) for k in got if got[k] != EXPECTED[k]}
