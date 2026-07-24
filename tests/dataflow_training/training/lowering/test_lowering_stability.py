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
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, lower_llama3
from dataflow_training.model_families.olmoe import ShapedOlmoeConfig, lower_olmoe
from dataflow_training.model_families.qwen3 import ShapedQwen3Config, lower_qwen3
from dataflow_training.model_families.qwen35 import ShapedQwen35Config, lower_qwen35
from dataflow_training.model_families.qwen35moe import ShapedQwen35MoeConfig, lower_qwen35moe
from dataflow_training.model_families.qwen3moe import ShapedQwen3MoeConfig, lower_qwen3moe
from dataflow_training.model_families.dsv3 import ShapedDsv3Config, lower_dsv3
from dataflow_training.model_families.dsv32 import ShapedDsv32Config, lower_dsv32
from dataflow_training.model_families.glm52 import ShapedGlm52Config, lower_glm52

# Constants last updated DELIBERATELY when tasks gained a cost_key: the
# geometry two tasks can differ in while their buffers look identical, which
# has to separate their measured costs. Every family's digest moves, because
# every task carries the field. The change was proved additive rather than
# semantic — stripping cost_key from a new program reproduces its previous
# digest exactly — so what moved is the metadata, not what any family lowers to.
EXPECTED = {
    "llama3-tiny-ga2-s2": "79efb2a5a7d8c0c9",
    "llama3-tiny-tail": "8ce346cd75d73ff4",
    "qwen3-tiny-ga3": "8a5a6e974d51127c",
    "qwen35-tiny-ga2": "b64c3b28ab98ad2d",
    "qwen35-tiny-tied": "81b6f4107a58f8b4",
    "olmoe-tiny": "af83b5562241b08f",
    "olmoe-tiny-ga2": "c8d4dff8a934619e",
    "qwen35moe-tiny-ga2": "0e562d6be5bf3930",
    "qwen3moe-tiny": "7e913c7966056b8b",
    "qwen3moe-tiny-ga2": "5a2ef4d525e09055",
    "dsv3-tiny": "9aaa357d40bf96f5",
    "dsv3-tiny-ga2": "1d43c849c3620815",
    "dsv32-tiny": "cb2166632574a82f",
    "dsv32-tiny-ga2": "7b7dc58dee30076e",
    "dsv32-tiny-dense": "417c25eca98562d1",
    "glm52-tiny": "b8730946e7d04fe9",
    "glm52-tiny-ga2": "798e31cd9745656b",
    "glm52-tiny-warmup": "117060a1342ad1df",
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
