"""Per-block isolation gates for the flip-amplified families.

The MODEL-level gradient bands for glm52/qwen35moe are wide (cos 0.87 /
0.89) because discrete routing flips cascade at smoke scale — the wide
band prices documented AMPLIFICATION, not per-op error, and would
indeed be a poor bug detector on its own. THIS gate carries the
bug-catching duty instead: feeding the ENGINE's own block-(N-1) output
into the twin's block N removes every upstream flip, so each block's
math must match at the bf16 FLOOR (median row-rel a few e-3, no
broadband) — a real per-op defect cannot hide here at any band width.
Followers isolate together with their leader (gotcha 9); budgets admit
only countable near-tie hot rows.
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.testing.gradcheck import (  # noqa: E402
    isolated_block_compare,
)

pytestmark = pytest.mark.gpu

# (family, blocks to isolate together, max hot rows). Blocks chosen to
# cover each family's block KINDS: glm52 gml leader (1), the gml+gmf
# leader/follower pair (4,5); qwen35moe lin (0) and gated-attention
# full (3); dsv3's first MoE (1). Floors from the certification runs
# (row-median 2.6-3.5e-3): gate at ~4x.
CASES = [
    ("glm52", (1,), 6),
    ("glm52", (4, 5), 6),
    ("qwen35moe", (0,), 4),
    ("qwen35moe", (3,), 4),
    ("dsv3", (1,), 3),
]

ROW_MEDIAN_MAX = 1.5e-2
EXCL_HOT_MAX = 3e-2


def tiny_ragged_cfg(family: str):
    import importlib

    mod = importlib.import_module(f"dataflow_training.model_families.{family}")
    cfg_cls = next(v for k, v in vars(mod).items()
                   if k.startswith("Shaped") and k.endswith("Config"))
    cfg = cfg_cls.tiny()
    t = cfg.seq_len * cfg.batch
    a = t // 2 + 3
    b = t // 4 + 1
    return replace(cfg, seq_lens=(a, b, t - a - b))


@pytest.mark.parametrize("family,isolate,hot_budget", CASES)
def test_isolated_block_at_floor(family, isolate, hot_budget):
    stats = isolated_block_compare(tiny_ragged_cfg(family), isolate)
    assert stats["row_median"] < ROW_MEDIAN_MAX, stats
    assert stats["rel_excl_hot"] < EXCL_HOT_MAX, stats
    assert len(stats["hot_rows"]) <= hot_budget, stats
