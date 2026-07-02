"""M1 gate: the engine on the fake backend must reproduce the simulator
exactly — task intervals, transfer intervals (including naming), and peak
fast memory — on annotated programs from tiny to full 8B scale.
"""
import pytest

from dataflow.core.convert import to_sim_chain
from dataflow.runtime import Engine, compare_to_sim_eventlog
from dataflow.runtime.device.fake import FakeBackend
from dataflow.training.planning import plan_program
from dataflow.training.shaped_llama3 import (
    ShapedHardware,
    ShapedLlamaConfig,
    build_shaped_llama3,
)

GIB = 1024**3


def run_both(annotated_program):
    from dataflow_sim.engine.simulator import run as sim_run

    result = Engine(FakeBackend()).execute(annotated_program)
    log = sim_run(to_sim_chain(annotated_program), snapshots=False)
    diff = compare_to_sim_eventlog(result.trace, log)
    assert diff.ok, (
        f"parity failed: peak sim={diff.peak_sim} runtime={diff.peak_runtime}; "
        f"missing={diff.missing[:5]}; extra={diff.extra[:5]} "
        f"({len(diff.missing)} missing / {len(diff.extra)} extra intervals)"
    )
    return result, log


def plan(cfg, cap, *, hw=None, recompute=False):
    program = build_shaped_llama3(cfg, hw=hw)
    return plan_program(
        program,
        fast_memory_capacity=cap,
        recompute=recompute,
        build_variant=(lambda levels: build_shaped_llama3(cfg, hw=hw, recompute_levels=levels))
        if recompute
        else None,
    ).program


def test_parity_tiny():
    run_both(plan(ShapedLlamaConfig.tiny(), 600_000))


def test_parity_tiny_tighter():
    run_both(plan(ShapedLlamaConfig.tiny(), 500_000))


def test_parity_tiny_recompute_all():
    cfg = ShapedLlamaConfig.tiny()
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    program = build_shaped_llama3(cfg, recompute_levels=levels)
    annotated = plan_program(program, fast_memory_capacity=600_000).program
    run_both(annotated)


def test_parity_tiny_grad_accum():
    cfg = ShapedLlamaConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=160,
        vocab_size=512, seq_len=64, batch=1, grad_accum_rounds=3,
    )
    run_both(plan(cfg, 700_000))


def test_parity_8b_16gib():
    """The headline M1 gate: full llama3-8B-shaped chain."""
    result, log = run_both(plan(ShapedLlamaConfig.llama3_8b(), 16 * GIB))
    assert result.makespan_us == pytest.approx(
        max(iv.end for iv in log.task_intervals), abs=0.0
    )


def test_parity_8b_tight_budget():
    run_both(plan(ShapedLlamaConfig.llama3_8b(), 10 * GIB))


def test_parity_8b_starved_pcie_recompute():
    """Blocked heads + deferred prefetches under pressure, with recompute
    tasks spliced in — the busiest scheduling regime."""
    cfg = ShapedLlamaConfig.llama3_8b()
    hw = ShapedHardware(pcie_gbs=10.0)
    run_both(plan(cfg, 8 * GIB, hw=hw, recompute=True))


def test_parity_grad_accum_8b_scale():
    cfg = ShapedLlamaConfig.llama3_8b(seq_len=2048, grad_accum_rounds=2)
    run_both(plan(cfg, 12 * GIB))


def test_buffer_reuse_happens_at_scale():
    result, _ = run_both(plan(ShapedLlamaConfig.llama3_8b(), 16 * GIB))
    assert result.buffers_reused > 0
