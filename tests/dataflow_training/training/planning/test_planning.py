"""Planner gates on the shaped Llama3 program: pressure-fit planning,
recompute selection, capacity monotonicity, and backing-capacity-driven
recompute.

Tests:
- test_pressurefit_plan_tiny: the plan is annotated, fits the cap with positive makespan, and re-simulates to exactly the planner's makespan and peak.
- test_plan_program_reports_object_capacity_and_fixed_leeway: planning reports the exact total, workspace, leeway, and derived object-only capacities.
- test_static_extent_feedback_replans_only_residency: a packing deficit reruns PressureFit on the selected task variant and returns an exactly admitted physical extent.
- test_static_extent_feedback_does_not_rerun_search_when_variant_fits: a feasible fixed-variant correction does not invoke the expensive complete-search fallback.
- test_plan_with_recompute_tiny: recompute-enabled planning fits the cap, covers all saved contexts, and each variant carries a recompute task iff its level is 1.
- test_recompute_fires_under_starved_interconnect: slow PCIe drives the planner to pick recompute and beat the save-all plan while staying under cap.
- test_capacity_sweep_monotone_tiny: looser memory budgets never yield a larger makespan.
- test_backing_capacity_drives_recompute: a backing cap between the save-all and recompute-all peaks forces more recomputation than the unbounded plan and still simulates green.
- test_level_pins_cover_every_variant_the_search_prices: the programs the profiler is told to measure carry every cost signature the recompute search can encounter, including the seeds it starts from.
- test_incomplete_cost_table_is_not_reported_as_infeasible: pricing a program against a table that never measured it raises MissingProfileError rather than being absorbed as a plan that does not fit.
- test_burst_sampled_profiles_are_refused_as_a_production_price: pricing production work from profiles taken under less load than the production floor raises UnderSampledProfileError instead of planning on optimistic numbers.
- test_recompute_never_plans_slower_than_saving_everything: offering recompute never yields a slower plan than the save-everything baseline it starts from, at any budget.
- test_measured_programs_are_built_in_one_place: nothing outside the profiling module assembles a measured program itself, so profiled costs and measured bandwidths cannot be applied by halves.
"""
from dataclasses import replace
from functools import partial

import pytest

from dataflow.core import validate_program
from dataflow_training.lowering.planning import (
    fit_static_extent,
    plan_program,
    simulate_program,
)
from dataflow_training.lowering.shaped_program import hardware_preset
from dataflow_training.model_families.llama3 import (ShapedLlamaConfig,
                                                     build_shaped_llama3)
from dataflow_training.run.profiling import (PRODUCTION_SAMPLE_SECONDS,
                                             MissingProfileError, TaskProfile,
                                             UnderSampledProfileError,
                                             _signature, apply_measured_costs,
                                             recompute_level_pins)

# fixture pricing: planner-BEHAVIOR assertions must be
# machine-independent, so lowerings here price with a named preset
HW = hardware_preset("rtx 5090")

TINY_CAP = 600_000  # bytes; tight enough to force movement on the tiny config


def test_pressurefit_plan_tiny():
    program = build_shaped_llama3(ShapedLlamaConfig.tiny(), hw=HW)
    planned = plan_program(program, fast_memory_capacity=TINY_CAP)

    validate_program(planned.program)
    assert planned.program.is_annotated()
    assert planned.peak_fast_bytes <= TINY_CAP
    assert planned.makespan_us > 0

    # the annotated program re-simulates to the planner's makespan
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) == planned.makespan_us
    assert log.peak_fast_memory_bytes == planned.peak_fast_bytes


def test_plan_program_reports_object_capacity_and_fixed_leeway():
    program = build_shaped_llama3(ShapedLlamaConfig.tiny(), hw=HW)
    leeway = 1_024
    planned = plan_program(
        program,
        fast_memory_capacity=TINY_CAP,
        program_leeway_bytes=leeway,
    )

    assert planned.program_leeway_bytes == leeway
    assert planned.object_memory_capacity == (
        TINY_CAP - planned.max_task_workspace_bytes - leeway
    )
    assert planned.peak_fast_bytes + leeway <= TINY_CAP


def test_static_extent_feedback_replans_only_residency():
    program = build_shaped_llama3(ShapedLlamaConfig.tiny(), hw=HW)
    leeway = 1_024
    planned = plan_program(
        program,
        fast_memory_capacity=TINY_CAP,
        program_leeway_bytes=leeway,
    )

    fitted = fit_static_extent(
        planned,
        user_memory_capacity=TINY_CAP,
        fixed_program_leeway_bytes=leeway,
        physical_extent=lambda candidate: (
            candidate.fast_memory_capacity + 10_000
        ),
        minimum_reserve_step_bytes=1,
    )

    assert fitted.packing_reserve_bytes == 10_000
    assert fitted.static_extent_replans == 1
    assert fitted.static_extent_bytes + leeway == TINY_CAP
    assert fitted.program.fast_memory_capacity == TINY_CAP - leeway - 10_000


def test_static_extent_feedback_does_not_rerun_search_when_variant_fits():
    program = build_shaped_llama3(ShapedLlamaConfig.tiny(), hw=HW)
    leeway = 1_024
    capacities = []

    def replan(capacity):
        capacities.append(capacity)
        candidate = plan_program(
            program,
            fast_memory_capacity=capacity,
            program_leeway_bytes=leeway,
        )
        return replace(candidate, recompute_levels={"choice": capacity})

    planned = replan(TINY_CAP)
    fitted = fit_static_extent(
        planned,
        user_memory_capacity=TINY_CAP,
        fixed_program_leeway_bytes=leeway,
        physical_extent=lambda candidate: (
            candidate.fast_memory_capacity + 10_000
        ),
        fallback_replan_for_capacity=replan,
        minimum_reserve_step_bytes=1,
    )

    assert capacities == [TINY_CAP]
    assert fitted.recompute_levels == {"choice": TINY_CAP}
    assert fitted.static_extent_replans == 1


def test_plan_with_recompute_tiny():
    cfg = ShapedLlamaConfig.tiny()
    program = build_shaped_llama3(cfg, hw=HW)
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
    under compute; verified in tools/export_program.py runs.)"""
    from dataclasses import replace
    from dataflow_training.lowering.shaped_program import hardware_preset

    cfg = ShapedLlamaConfig.llama3_8b()
    hw = replace(hardware_preset("rtx 5090"), pcie_gbs=10.0)
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
    program = build_shaped_llama3(ShapedLlamaConfig.tiny(), hw=HW)
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
    program = build_shaped_llama3(cfg, hw=HW)
    all_levels = {
        rw.object_id: rw.options[-1].level for rw in program.recompute_rewrites
    }

    def plan_at(backing: int | None):
        def variant(levels):
            return replace(
                build_shaped_llama3(cfg, recompute_levels=levels, hw=HW),
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
        replace(build_shaped_llama3(cfg, recompute_levels=all_levels, hw=HW),
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


def test_level_pins_cover_every_variant_the_search_prices():
    """Measured costs are looked up by signature, and the recompute search
    prices programs the base lowering does not contain: a block that recomputes
    emits a task with no counterpart there, and its forward stops emitting the
    saved-activation object, which changes that forward's signature too. When a
    signature is missing the variant cannot be priced at all — which is
    indistinguishable, to the search, from a variant that does not fit. The
    pins the profiler is handed have to close that gap."""
    cfg = ShapedLlamaConfig.tiny()
    program = build_shaped_llama3(cfg, hw=HW)

    covered = set()
    for pins in recompute_level_pins(program):
        variant = build_shaped_llama3(cfg, recompute_levels=pins, hw=HW)
        sizes = variant.object_sizes()
        covered |= {_signature(t, sizes, None) for t in variant.tasks}

    top = {rw.object_id: rw.options[-1].level for rw in program.recompute_rewrites}
    # the seeds the search evaluates before its greedy loop, plus the mixed
    # assignment the loop walks through
    for levels in (dict.fromkeys(top, 0),
                   top,
                   {obj: (lvl if n % 2 == 0 else 0)
                    for n, (obj, lvl) in enumerate(top.items())}):
        variant = build_shaped_llama3(cfg, recompute_levels=levels, hw=HW)
        sizes = variant.object_sizes()
        missing = {_signature(t, sizes, None) for t in variant.tasks} - covered
        assert not missing, f"unpriceable tasks in variant: {sorted(missing)}"


def test_incomplete_cost_table_is_not_reported_as_infeasible():
    """A table that cannot price a program is a fault in what was measured, not
    a plan that does not fit. It has to raise as itself: the recompute search
    treats the planner's ValueError as "this variant is infeasible", so a cost
    lookup that failed the same way would be recorded as a variant to discard,
    and a feasible plan would be reported as impossible."""
    cfg = ShapedLlamaConfig.tiny()
    program = build_shaped_llama3(cfg, hw=HW)

    with pytest.raises(MissingProfileError) as excinfo:
        apply_measured_costs(program, {}, None)
    assert not isinstance(excinfo.value, ValueError)
    assert program.tasks[0].id in str(excinfo.value)


def test_burst_sampled_profiles_are_refused_as_a_production_price():
    """The sampling floor is opt-in, so the guard has to be the thing that
    catches a pricing path which forgot to ask for it. A signature timed in a
    short burst reads several percent faster than the same work under
    sustained load (block_fwd: 22.16 ms burst vs 23.56 ms sustained, and
    production reproduces the sustained figure), so a plan built from burst
    numbers is optimistic and entirely plausible-looking — exactly the kind of
    silent wrongness MissingProfileError exists to prevent for absent costs."""
    # Pytest globally selects the zero-floor correctness path so no test pays
    # production's 2.5 seconds per signature.  Keep this contract test pinned
    # to the real production requirement rather than the test-process setting.
    production_floor = 2.5
    assert PRODUCTION_SAMPLE_SECONDS == 0.0

    cfg = ShapedLlamaConfig.tiny()
    program = build_shaped_llama3(cfg, hw=HW)
    sizes = program.object_sizes()

    def table(sampled_seconds):
        return {_signature(t, sizes): TaskProfile(
            runtime_us=100.0, workspace_bytes=0, repeats=9,
            sampled_us=sampled_seconds * 1e6,
            sample_floor_s=sampled_seconds) for t in program.tasks}

    # a cost table is a legitimate use and demands no floor
    apply_measured_costs(program, table(0.0))
    # a profile from a cache predating the field is unknown, never refused
    legacy = {_signature(t, program.object_sizes()): TaskProfile(
        runtime_us=100.0, workspace_bytes=0, repeats=9) for t in program.tasks}
    apply_measured_costs(program, legacy,
                         require_sample_seconds=production_floor)
    # so is a production price built from production-sampled sigs
    apply_measured_costs(program, table(production_floor),
                         require_sample_seconds=production_floor)

    with pytest.raises(UnderSampledProfileError) as excinfo:
        apply_measured_costs(program, table(0.05),
                             require_sample_seconds=production_floor)
    assert program.tasks[0].id in str(excinfo.value)


def test_recompute_never_plans_slower_than_saving_everything():
    """Offering recompute can only help: the save-everything plan is the
    search's own starting point, and it keeps the best plan it has seen. So a
    budget where recompute is available must never plan slower than the same
    budget without it — if it does, the search has lost a variant it was
    holding, which is how a whole sweep once came back reporting feasible
    cells as impossible."""
    cfg = ShapedLlamaConfig.tiny()
    program = build_shaped_llama3(cfg, hw=HW)

    for cap in (500_000, 800_000, 2_000_000):
        saved = plan_program(program, fast_memory_capacity=cap)
        offered = plan_program(
            program,
            fast_memory_capacity=cap,
            recompute=True,
            build_variant=partial(build_variant_at, cfg),
        )
        assert offered.makespan_us <= saved.makespan_us, (
            f"at cap {cap}: recompute planned {offered.makespan_us} us, "
            f"slower than saving everything at {saved.makespan_us} us")


def build_variant_at(cfg, levels):
    return build_shaped_llama3(cfg, recompute_levels=levels, hw=HW)


def test_measured_programs_are_built_in_one_place():
    """Assembling a measured program means two things -- profiled task costs
    and the box's own link bandwidths -- and applying one without the other
    biases every plan built from it. That gap existed twice: a second copy of
    the assembly profiled only the base lowering, so the recompute search could
    price nothing, and the same copy never installed measured bandwidths, so
    its plans came out ~10% optimistic against the identical cell the driver
    planned honestly. Both were invisible because the copies looked right in
    isolation. Only the module that owns the cost table may reach for these."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve()
    while not (root / "src" / "dataflow_training").is_dir():
        root = root.parent
    owner = root / "src" / "dataflow_training" / "run" / "profiling.py"
    guarded = {"apply_measured_costs", "load_or_profile"}

    offenders = []
    for path in list((root / "src").rglob("*.py")) + list((root / "tools").rglob("*.py")):
        if path == owner:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in guarded:
                    offenders.append(f"{path.relative_to(root)}:{node.lineno} "
                                     f"calls {node.func.id}")
    assert not offenders, (
        "measured programs must come from profiling.measured_program / "
        "measured_profile_table so both halves of the measurement travel "
        "together:\n  " + "\n  ".join(offenders))
