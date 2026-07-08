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
from dataflow.training.qwen3moe import ShapedQwen3MoeConfig, lower_qwen3moe
from dataflow.training.dsv3 import ShapedDsv3Config, lower_dsv3
from dataflow.training.dsv32 import ShapedDsv32Config, lower_dsv32
from dataflow.training.glm52 import ShapedGlm52Config, lower_glm52

# Constants last updated DELIBERATELY for the fused head_loss lowering
# (head_fwd/loss_bwd/head_bwd -> ONE token-chunked task; logits/dlogits
# objects removed from the grammar). olmoe rows ADDED with the MoE family
# (existing constants verified unchanged in the same commit).
EXPECTED = {
    "llama3-tiny-ga2-s2": "28cb016ba2779a5c",
    "llama3-tiny-tail": "e06f9a28c9665f46",
    "qwen3-tiny-ga3": "fd8e2305dd04e271",
    "qwen35-tiny-ga2": "3a76b7ddade100eb",
    "qwen35-tiny-tied": "f6dd20935fc2ad10",
    "olmoe-tiny": "6e122f9e55566eb1",
    "olmoe-tiny-ga2": "cace8430b38656c8",
    "qwen35moe-tiny-ga2": "dd63fb7554d28ac5",
    "qwen3moe-tiny": "b0cae7b2f6845f8e",
    "qwen3moe-tiny-ga2": "37be833884048665",
    "dsv3-tiny": "887016aeea7a406d",
    "dsv3-tiny-ga2": "0235b4af1dcd1378",
    "dsv32-tiny": "c8fff3a4b6125f42",
    "dsv32-tiny-ga2": "fc88bcbcd5b922f2",
    "dsv32-tiny-dense": "7d6c6179f0c00e33",
    "glm52-tiny": "02de39829135fecd",
    "glm52-tiny-ga2": "97d566e41cfd5a02",
    "glm52-tiny-warmup": "45d5d022358e5c74",
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
