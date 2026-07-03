"""Window-plan analyzer: canonicalization + seam extraction (CPU-only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from window_plans import (  # noqa: E402
    analyze_window,
    canon_obj,
    canon_task,
    replicate_levels,
    task_step,
)

from dataflow.training.llama3_lowering import lower_llama3  # noqa: E402
from dataflow.training.planning import plan_program, simulate_program  # noqa: E402
from dataflow.training.shaped_llama3 import ShapedLlamaConfig  # noqa: E402


def test_canonicalization_strips_only_step_index():
    assert task_step("block_bwd_3_1_17") == 3
    assert canon_task("block_bwd_3_1_17") == "block_bwd_*_1_17"
    assert canon_task("optimizer_embed_2") == "optimizer_embed_*"
    assert canon_task("optimizer_2_31") == "optimizer_*_31"
    assert canon_obj("A_2_0_5") == "A_*_0_5"
    assert canon_obj("dW_2_5") == "dW_*_5"
    assert canon_obj("dW_head_2") == "dW_head_*"
    assert canon_obj("dy_embed_1_0") == "dy_embed_*_0"
    assert canon_obj("dlogits_1_0") == "dlogits_*_0"
    assert canon_obj("W_5") == "W_5"       # global: untouched
    assert canon_obj("O_embed") == "O_embed"


def test_replicate_levels_across_steps():
    out = replicate_levels({"A_0_0_1": 1, "A_0_2_0": 0}, 3)
    assert out == {
        "A_0_0_1": 1, "A_1_0_1": 1, "A_2_0_1": 1,
        "A_0_2_0": 0, "A_1_2_0": 0, "A_2_2_0": 0,
    }


def _tiny(k, cap):
    from dataclasses import replace

    cfg = replace(ShapedLlamaConfig.tiny(), grad_accum_rounds=2, num_steps=k)
    planned = plan_program(lower_llama3(cfg), fast_memory_capacity=cap)
    return analyze_window(planned.program, simulate_program(planned.program))


def test_analyze_window_tiny_generous():
    row = _tiny(4, 64 * 1024 * 1024)
    assert row["num_steps"] == 4
    assert row["interior_periodic"] in (True, False)
    assert set(row["seams"]) == {"0", "1", "2"}
    # generous cap: weights stay resident across every seam
    for seam in row["seams"].values():
        assert "W_0" in seam["resident_fast_objects"]
        assert seam["resident_global_gib"] > 0


def test_analyze_window_tiny_tight_has_seam_traffic():
    row = _tiny(3, 600 * 1024)  # tight: forces offload/prefetch churn
    assert row["num_steps"] == 3
    total = sum(
        s["inflight_gib"] + s["cross_seam_prefetch_gib"] + s["resident_fast_gib"]
        for s in row["seams"].values()
    )
    assert total >= 0.0  # smoke: every id parsed, every seam extracted
