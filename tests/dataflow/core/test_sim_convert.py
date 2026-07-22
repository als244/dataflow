"""Converter tests against the real dataflow_sim package.

Tests:
- test_to_sim_chain_preserves_ids_and_sizes: to_sim_chain keeps task ids, initial-memory ids, and per-output sizes.
- test_annotated_chain_validates: a pressurefit-annotated chain passes the sim's location-aware validate_chain.
- test_from_sim_chain_roundtrip: from_sim_chain then to_sim_chain reproduces the original chain (sim-visible content lossless).
- test_annotation_join_is_lossless: program to chain to pressurefit to joined program preserves annotations and capacity and yields the same chain.
- test_annotated_program_simulates_identically: the joined annotated program simulates to the same task intervals and peak memory as the direct chain.
- test_webapp_export_realizes: the exported DataflowProgram validates, realizes in the sim, and simulates within the memory budget.
"""
import pytest

from dataflow.core import validate_program
from dataflow.core.convert import (
    apply_chain_annotations,
    from_sim_chain,
    to_sim_chain,
    to_webapp_program,
)
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, build_shaped_llama3

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def tiny_program():
    return build_shaped_llama3(ShapedLlamaConfig.tiny())


def test_to_sim_chain_preserves_ids_and_sizes(tiny_program):
    # NOTE: the sim's validate_chain is location-aware and only meaningful for
    # ANNOTATED chains (a bare chain has no prefetches for its backing-source
    # objects yet). Structural fidelity is asserted here; annotated-chain
    # validation happens in test_annotation_join_is_lossless via the policy.
    chain = to_sim_chain(tiny_program)
    assert [t.id for t in chain.tasks] == [t.id for t in tiny_program.tasks]
    assert {o.id for o in chain.initial_memory} == {o.id for o in tiny_program.initial_objects}
    sizes = tiny_program.object_sizes()
    for t in chain.tasks:
        for out in t.outputs:
            assert out.size == sizes[out.id]


def test_annotated_chain_validates(tiny_program):
    from dataflow_sim.core.validate import validate_chain
    from dataflow_sim.policies.pressurefit import apply_pressurefit_policy

    annotated = apply_pressurefit_policy(to_sim_chain(tiny_program), fast_memory_capacity=600_000)
    validate_chain(annotated)


def test_from_sim_chain_roundtrip(tiny_program):
    chain = to_sim_chain(tiny_program)
    back = from_sim_chain(chain, name=tiny_program.name)
    validate_program(back)
    assert to_sim_chain(back) == chain  # sim-visible content is lossless


def test_annotation_join_is_lossless(tiny_program):
    """program -> chain -> pressurefit -> join -> chain must be identical."""
    from dataflow_sim.policies.pressurefit import apply_pressurefit_policy

    cap = 600_000
    annotated_chain = apply_pressurefit_policy(to_sim_chain(tiny_program), fast_memory_capacity=cap)
    annotated_program = apply_chain_annotations(tiny_program, annotated_chain)
    validate_program(annotated_program)
    assert annotated_program.is_annotated()
    assert annotated_program.fast_memory_capacity == cap
    assert to_sim_chain(annotated_program) == annotated_chain


def test_annotated_program_simulates_identically(tiny_program):
    from dataflow_sim.engine.simulator import run
    from dataflow_sim.policies.pressurefit import apply_pressurefit_policy

    annotated_chain = apply_pressurefit_policy(to_sim_chain(tiny_program), fast_memory_capacity=600_000)
    log_direct = run(annotated_chain, snapshots=False)

    annotated_program = apply_chain_annotations(tiny_program, annotated_chain)
    log_joined = run(to_sim_chain(annotated_program), snapshots=False)

    assert log_direct.task_intervals == log_joined.task_intervals
    assert log_direct.peak_fast_memory_bytes == log_joined.peak_fast_memory_bytes


def test_webapp_export_realizes(tiny_program):
    """The exported DataflowProgram v1 must validate + realize in the sim."""
    from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
    from dataflow_sim.workloads.dataflow import DataflowProgram, realize_dataflow_program

    payload = to_webapp_program(tiny_program)
    prog = DataflowProgram.model_validate(payload)
    # any preset serves: the assertions below are structural
    preset_name = sorted(HARDWARE_PRESETS)[0]
    hw = HARDWARE_PRESETS[preset_name]
    workload = realize_dataflow_program(prog, hw)
    assert len(workload.chain.tasks) == len(tiny_program.tasks)

    # what the webapp does server-side: realize -> policy -> simulate
    from dataflow_sim.engine.simulator import run
    from dataflow_sim.policies.pressurefit import apply_pressurefit_policy

    annotated = apply_pressurefit_policy(workload.chain, fast_memory_capacity=600_000)
    log = run(annotated, snapshots=False)
    assert log.peak_fast_memory_bytes <= 600_000
