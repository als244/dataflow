from functools import partial

from dataflow.core import validate_program
from dataflow_training.lowering.planning import plan_program, simulate_program
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, build_shaped_llama3

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
    from dataflow_training.lowering.shaped_program import ShapedHardware

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


def test_backing_capacity_drives_recompute():
    """First-class backing capacity steering the recompute planner.

    The lever exists in the GRAD-ACCUM regime, where saved contexts dominate
    backing demand: recompute variants genuinely shrink the footprint, so a
    cap between the save-all and recompute-all peaks forces recomputation to
    replace offloading. (At ga=1 the footprint is dominated by the W/dW
    round-trip, which no recompute level removes — there the cap is a sharp
    feasibility cliff, not a dial; measured and documented.)"""
    from dataclasses import replace

    from dataflow.runtime import Engine
    from dataflow.runtime.device.fake import FakeBackend

    cfg = ShapedLlamaConfig.llama3_8b(batch=1, grad_accum_rounds=2)
    cap = 12 * 1024**3
    program = build_shaped_llama3(cfg)
    all_levels = {
        rw.object_id: rw.options[-1].level for rw in program.recompute_rewrites
    }

    def plan_at(backing: int | None):
        def variant(levels):
            return replace(
                build_shaped_llama3(cfg, recompute_levels=levels),
                backing_memory_capacity=backing,
            )

        return plan_program(
            replace(program, backing_memory_capacity=backing),
            fast_memory_capacity=cap,
            recompute=True,
            build_variant=variant,
        )

    def peak_backing(planned) -> int:
        dry = Engine(FakeBackend()).execute(planned.program)
        peak = dry.peak_backing_bytes
        dry.close()
        return peak

    unlimited = plan_at(None)
    n_unlimited = sum(1 for v in unlimited.recompute_levels.values() if v > 0)
    # the free planner must stay far from recompute-all (else there is no dial)
    assert n_unlimited < len(all_levels) // 4, n_unlimited

    # measure both ends of the dial EXPLICITLY (the free plan need not be
    # exactly save-all — e.g. optimizer interleaving shifts one layer), cap
    # in between
    save_planned = plan_program(
        replace(program, backing_memory_capacity=None), fast_memory_capacity=cap,
    )
    save_peak = peak_backing(save_planned)
    rc_all_planned = plan_program(
        replace(build_shaped_llama3(cfg, recompute_levels=all_levels),
                backing_memory_capacity=None),
        fast_memory_capacity=cap,
    )
    rc_peak = peak_backing(rc_all_planned)
    assert save_peak > rc_peak, (save_peak, rc_peak)
    tight = plan_at((save_peak + rc_peak) // 2)

    n_tight = sum(1 for v in tight.recompute_levels.values() if v > 0)
    assert n_tight > n_unlimited, (n_unlimited, n_tight)
    # the tight plan must actually simulate green under the cap
    log = simulate_program(tight.program)
    assert max(iv.end for iv in log.task_intervals) == tight.makespan_us
