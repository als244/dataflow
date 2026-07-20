"""Cross-family preset registry + the locked training configuration.

Per-family preset builders live in their family packages
(``model_families/<family>/presets.py``) and are re-exported here, so
``from dataflow_training.run.presets import <preset>`` stays the one
import surface for tools and tests. This module owns only what is
genuinely cross-family: the locked training-config constants, the preset
name lookup (``resolve_preset``), the wire-spec glue (``resolver_family``
/ ``cfg_dict``), and the study bookkeeping helpers.

The training config is fixed (locked 2026-07-09): sequences of length
2048 (uniform, no ragged packing), 4 per round → 8192 tokens/round, 8
grad-accum rounds → 65,536 (~64K) tokens/step, gpt2 vocab padded to
50304. The scaling ladder reuses this exact config; only the model shape
changes. Family preset builders read these constants back from here
(function-scope imports — this module imports the family preset modules
for the re-exports, so the constants are the one shared anchor).
"""
from __future__ import annotations

from dataflow_training.model_families.llama3 import ShapedLlamaConfig

# -- locked training config ---------------------------------------------------
SEQ_LEN = 2048
BATCH = 4                    # 4 × 2048 = 8192 tokens / round
GRAD_ACCUM_ROUNDS = 8        # 8 × 8192 = 65,536 tokens / step
VOCAB_SIZE = 50304
# Driver loop count = number of run() invocations. The PROGRAM stays
# single-step (num_steps=1, the ShapedLlamaConfig default): the daemon holds
# W/O in the store and evolves them in place across run() calls. Unrolling
# the steps into the program instead builds a ~300k-task monster.
TRAIN_STEPS = 1000

# The tiny infra/parity SMOKE geometry (real 50304 vocab so smoke configs
# consume real fineweb tokens through the whole pipeline; every family's
# smoke twin shares it — the per-family shapes live with the families).
SMOKE_SEQ_LEN = 256
SMOKE_BATCH = 2              # 512 tokens / round
SMOKE_GRAD_ACCUM_ROUNDS = 2  # 1024 tokens / step
SMOKE_STEPS = 100           # driver loop count for the parity smoke gate

# -- family preset re-exports -------------------------------------------------
# (import sites keep saying ``from dataflow_training.run.presets import X``)
from dataflow_training.model_families.dsv3.presets import (  # noqa: E402,F401
    dsv3_2b_preset,
    dsv3_cfg_dict,
    dsv3_smoke_preset,
)
from dataflow_training.model_families.dsv32.presets import (  # noqa: E402,F401
    dsv32_cfg_dict,
    dsv32_smoke_preset,
)
from dataflow_training.model_families.glm52.presets import (  # noqa: E402,F401
    glm52_cfg_dict,
    glm52_smoke_preset,
)
from dataflow_training.model_families.gpt2.presets import (  # noqa: E402,F401
    gpt2_cfg_dict,
    gpt2_preset,
    gpt2_smoke_preset,
)
from dataflow_training.model_families.llama3.presets import (  # noqa: E402,F401
    LADDER,
    LADDER_NAMES,
    SMOKE,
    llama3_cfg_dict,
    preset,
    smoke_preset,
)
from dataflow_training.model_families.olmoe.presets import (  # noqa: E402,F401
    olmoe_cfg_dict,
    olmoe_smoke_preset,
)
from dataflow_training.model_families.qwen3.presets import (  # noqa: E402,F401
    qwen3_cfg_dict,
    qwen3_smoke_preset,
)
from dataflow_training.model_families.qwen3moe.presets import (  # noqa: E402,F401
    qwen3moe_cfg_dict,
    qwen3moe_smoke_preset,
)
from dataflow_training.model_families.qwen35.presets import (  # noqa: E402,F401
    QWEN35_BATCH,
    QWEN35_GRAD_ACCUM_ROUNDS,
    QWEN35_SEQ_LEN,
    qwen35_cfg_dict,
    qwen35_preset,
    qwen35_smoke_preset,
)
from dataflow_training.model_families.qwen35moe.presets import (  # noqa: E402,F401
    qwen35moe_cfg_dict,
    qwen35moe_smoke_preset,
)


def resolve_preset(name: str):
    """Preset name -> config, across families ('qwen35' -> the hybrid preset,
    'dsv3_2b'/'dsv3_2b_nolbl' -> the MoE study preset, otherwise the
    llama3 ladder). Plugin-registered bench configs
    (run.bench_presets.register_bench_config) resolve here too."""
    from dataflow_training.run.bench_presets import EXTRA_CONFIGS

    if name in EXTRA_CONFIGS:
        return EXTRA_CONFIGS[name]
    if name in ("qwen35", "q35"):
        return qwen35_preset()
    if name == "dsv3_2b":
        return dsv3_2b_preset(load_balance=True)
    if name == "dsv3_2b_nolbl":
        return dsv3_2b_preset(load_balance=False)
    if name in ("gpt2_124m", "gpt2"):
        return gpt2_preset()
    return preset(name)


RESOLVER_FAMILY_BY_TYPE = {
    "ShapedDsv3Config": "dsv3",
    "ShapedDsv32Config": "dsv32",
    "ShapedGlm52Config": "glm52",
    "ShapedGpt2Config": "gpt2",
    "ShapedLlamaConfig": "llama3",
    "ShapedOlmoeConfig": "olmoe",
    "ShapedQwen3Config": "qwen3",
    "ShapedQwen3MoeConfig": "qwen3moe",
    "ShapedQwen35Config": "qwen35",
    "ShapedQwen35MoeConfig": "qwen35moe",
}


def resolver_family(cfg) -> str:
    name = type(cfg).__name__
    if name not in RESOLVER_FAMILY_BY_TYPE:
        raise KeyError(f"no resolver family for config type {name}; "
                       f"known: {sorted(RESOLVER_FAMILY_BY_TYPE)}")
    return RESOLVER_FAMILY_BY_TYPE[name]


CFG_DICT_BY_TYPE = {
    "ShapedDsv3Config": dsv3_cfg_dict,
    "ShapedDsv32Config": dsv32_cfg_dict,
    "ShapedGlm52Config": glm52_cfg_dict,
    "ShapedGpt2Config": gpt2_cfg_dict,
    "ShapedLlamaConfig": llama3_cfg_dict,
    "ShapedOlmoeConfig": olmoe_cfg_dict,
    "ShapedQwen3Config": qwen3_cfg_dict,
    "ShapedQwen3MoeConfig": qwen3moe_cfg_dict,
    "ShapedQwen35Config": qwen35_cfg_dict,
    "ShapedQwen35MoeConfig": qwen35moe_cfg_dict,
}


def cfg_dict(cfg) -> dict:
    """JSON-able config for the wire resolver spec. The daemon rebuilds
    ``config_type(**cfg)`` (the service resolver), so every field here must
    be a constructor kwarg; omitted fields take their defaults (all-bf16
    ``dtypes``, ``seq_lens=None`` uniform) — the SAME defaults the planned
    program used, so the rebuilt dims match. A non-default ``opt_policy``
    rides along when it is a string shorthand ("muon", "sgd", ...) — the
    daemon's layouts must size optimizer state with the SAME policy the
    program was lowered with (muon matrices carry m only, half the adamw
    bytes). Policy OBJECTS cannot ride the wire spec."""
    name = type(cfg).__name__
    if name not in CFG_DICT_BY_TYPE:
        raise KeyError(f"no cfg_dict serializer for config type {name}; "
                       f"known: {sorted(CFG_DICT_BY_TYPE)}")
    d = CFG_DICT_BY_TYPE[name](cfg)
    policy = getattr(cfg, "opt_policy", None)
    if policy is not None and policy != "adamw":
        if not isinstance(policy, str):
            raise ValueError(
                f"{name}.opt_policy is a policy object — only string "
                f"shorthands ride the wire resolver spec")
        d["opt_policy"] = policy
    return d


def param_counts(cfg: ShapedLlamaConfig) -> dict:
    """(embed, head, blocks, total, non_embedding) parameter counts."""
    embed = cfg.embed_params
    head = cfg.head_params
    blocks = cfg.n_layers * cfg.block_params
    total = embed + head + blocks
    return dict(embed=embed, head=head, blocks=blocks, total=total,
                non_embedding=blocks)


def tokens_per_step(cfg: ShapedLlamaConfig) -> int:
    return cfg.tokens * cfg.grad_accum_rounds
