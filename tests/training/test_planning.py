from functools import partial

from dataflow.core import validate_program
from dataflow.training.planning import plan_program, simulate_program
from dataflow.training.shaped_llama3 import ShapedLlamaConfig, build_shaped_llama3

TINY_CAP = 600_000  # bytes; tight enough to force movement on the tiny config


def test_pressurefit_plan_tiny():
    program = build_shaped_llama3(ShapedLlamaConfig.tiny())
    planned = plan_program(program, fast_memory_capacity=TINY_CAP)

    validate_program(planned.program)
    assert planned.program.is_annotated()
    assert planned.peak_fast_bytes <= TINY_CAP
    assert planned.makespan_us > 0

    # the annotated program re-simulates to the planner's makespan
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) == planned.makespan_us
    assert log.peak_fast_memory_bytes == planned.peak_fast_bytes


def test_plan_with_recompute_tiny():
    cfg = ShapedLlamaConfig.tiny()
    program = build_shaped_llama3(cfg)
    build_variant = partial(build_shaped_llama3, cfg)

    planned = plan_program(
        program,
        fast_memory_capacity=TINY_CAP,
        recompute=True,
        build_variant=lambda levels: build_variant(recompute_levels=levels),
    )
    validate_program(planned.program)
    assert planned.program.is_annotated()
    assert planned.peak_fast_bytes <= TINY_CAP
    assert set(planned.recompute_levels) == {f"A_0_0_{i}" for i in range(cfg.n_layers)}

    # chosen variant must contain a recompute task iff its level is 1
    ids = {t.id for t in planned.program.tasks}
    for a_id, level in planned.recompute_levels.items():
        _, s, r, i = a_id.split("_")
        assert (f"block_recompute_{s}_{r}_{i}" in ids) == (level == 1)


def test_recompute_fires_under_starved_interconnect():
    """With PCIe too slow to hide offload round-trips, the planner must
    choose recompute and beat the save-all pressurefit plan. (At healthy
    PCIe the same config correctly chooses zero recompute — transfers hide
    under compute; verified in tools/golden_path.py runs.)"""
    from dataflow.training.shaped_llama3 import ShapedHardware

    cfg = ShapedLlamaConfig.llama3_8b()
    hw = ShapedHardware(pcie_gbs=10.0)
    cap = 8 * 1024**3
    program = build_shaped_llama3(cfg, hw=hw)

    planned = plan_program(
        program,
        fast_memory_capacity=cap,
        recompute=True,
        build_variant=lambda levels: build_shaped_llama3(cfg, hw=hw, recompute_levels=levels),
    )
    chosen = sum(1 for v in planned.recompute_levels.values() if v > 0)
    assert chosen > 0
    assert planned.peak_fast_bytes <= cap

    baseline = plan_program(program, fast_memory_capacity=cap)
    assert planned.makespan_us < baseline.makespan_us


def test_capacity_sweep_monotone_tiny():
    """Looser budgets should never plan slower (sanity of the whole path)."""
    program = build_shaped_llama3(ShapedLlamaConfig.tiny())
    caps = [500_000, 800_000, 2_000_000]
    makespans = [plan_program(program, fast_memory_capacity=c).makespan_us for c in caps]
    assert makespans[0] >= makespans[1] >= makespans[2]
