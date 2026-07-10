"""Bit-identical lowering tripwire: generalizing the shared chain builder
(heterogeneous layer kinds, tied embeddings) must NOT change what existing
families emit — same ids, same order, same sizes, same directives-bare
structure. A legitimate lowering change updates these constants in the same
commit, deliberately."""
import hashlib
import json
from dataclasses import replace

from dataflow.core.jsonio import program_to_dict
from dataflow.training.models.llama3 import ShapedLlamaConfig, lower_llama3
from dataflow.training.models.olmoe import ShapedOlmoeConfig, lower_olmoe
from dataflow.training.models.qwen3 import ShapedQwen3Config, lower_qwen3
from dataflow.training.models.qwen35 import ShapedQwen35Config, lower_qwen35
from dataflow.training.models.qwen35moe import ShapedQwen35MoeConfig, lower_qwen35moe
from dataflow.training.models.qwen3moe import ShapedQwen3MoeConfig, lower_qwen3moe
from dataflow.training.models.dsv3 import ShapedDsv3Config, lower_dsv3
from dataflow.training.models.dsv32 import ShapedDsv32Config, lower_dsv32
from dataflow.training.models.glm52 import ShapedGlm52Config, lower_glm52

# Constants last updated DELIBERATELY for the aux-object grammar (the
# per-round metadata objects renamed M_{s}_{r}_{i} -> AuxTemp_{s}_{r}_{i}
# and dM_{s}_{r}_{lead} -> dAuxTemp_{s}_{r}_{lead}; structure otherwise
# unchanged). The five DENSE-family constants are UNTOUCHED — verified
# byte-identical in the same commit (no aux objects in those chains).
EXPECTED = {
    "llama3-tiny-ga2-s2": "28cb016ba2779a5c",
    "llama3-tiny-tail": "e06f9a28c9665f46",
    "qwen3-tiny-ga3": "fd8e2305dd04e271",
    "qwen35-tiny-ga2": "3a76b7ddade100eb",
    "qwen35-tiny-tied": "f6dd20935fc2ad10",
    "olmoe-tiny": "213c075480eb6eb4",
    "olmoe-tiny-ga2": "5bd614146a4c76bc",
    "qwen35moe-tiny-ga2": "c158717e023a55b0",
    "qwen3moe-tiny": "38a3f61397a4d91d",
    "qwen3moe-tiny-ga2": "90e08015ff3c8379",
    "dsv3-tiny": "fb3ee48042e5906c",
    "dsv3-tiny-ga2": "5a0f399e32e62be1",
    "dsv32-tiny": "f63908c078f99531",
    "dsv32-tiny-ga2": "de894d6c97f359ae",
    "dsv32-tiny-dense": "3744001828e42a4b",
    "glm52-tiny": "225ea494965c06d0",
    "glm52-tiny-ga2": "9fe67fd5aaf485c0",
    "glm52-tiny-warmup": "6583684c8b8d9993",
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
