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
    # qwen3moe: qwen3 attention verbatim + MoE FFN (no shared expert)
    "qwen3moe-tiny": "886848980dd78436",
    "qwen3moe-tiny-ga2": "be8b0be9821dc76c",
    # dsv3: MLA + hybrid dense/MoE depth + sigmoid_noaux_tc
    "dsv3-tiny": "dbf9d95a46cc53a7",
    "dsv3-tiny-ga2": "95af056e117d237a",
    # dsv32/glm52: METADATA grammar (Shein naming iteration 2026-07-07):
    # each layer's never-recompute artifacts (routing pack, dsa
    # selection) live in ONE normal object M_{s}_{r}_{i}; IndexShare
    # followers also consume the producer's M; the cross-layer KL
    # accumulator is dM (backward companion, accumulated like dW).
    # Recompute repopulates ONLY the A objects.
    "dsv32-tiny": "abd87724c1d90494",
    "dsv32-tiny-ga2": "faf81c930445fde4",
    "dsv32-tiny-dense": "e6606d8ef769ae9f",
    "glm52-tiny": "17e45164ff6a8fca",
    "glm52-tiny-ga2": "4df8f99bebe0920b",
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
    }
    assert got == EXPECTED, {k: (got[k], EXPECTED[k]) for k in got if got[k] != EXPECTED[k]}
