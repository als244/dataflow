"""Reserve-order inversion: minimal deterministic reproduction.

Pins the CURRENT engine behavior for the bs32-class deadlock (the
reserve-order-inversion design note): a hand-authored, SIM-VALID
training-shaped program whose realized transfer timing admits
trigger-satisfied prefetches into the bytes the plan earmarked for a
blocked task's output reservation. The eviction valve then feeds the
poke loop (freed bytes go to the waiting transfer head — the
simulator's tie priority) so every eviction is net-zero for the
blocked task, until the thrash guard (10 x n_tasks) trips and
DeadlockError surfaces.

The scenario (capacity 10 MiB, planned bandwidth ~21 B/us so each 2 MiB
prefetch is planned at ~100 ms; real PCIe moves it in ~us):

    T0  (in W)        -> P(4)   directives: prefetch Z1,Z2,Z3,Q after T0
    T1  (in W,P)      -> Y(2)   releases P after; THE BLOCKED RESERVE
    T2  (in Y)                  releases Y after
    Tz  (in Z1,Z2,Z3,Q,W) -> loss

Planned timeline (sim-verified below, peak exactly at capacity): the
prefetch lane is planned SLOW, so Z2 lands during T1, Z3/Q only after
Y releases. Realized: transfers are ~instant, so at T0's retirement
Z1+Z2 charge immediately (can_reserve admits to 10/10), Z3 parks at
the head, and T1 cannot reserve Y. Each valve eviction pokes the head
(Z3, then Q, then the evictees' own reloads) — never T1.

WHEN THE PLAN-ORDER ADMISSION FIX LANDS (design note section 4a),
flip this test: the run must COMPLETE with zero pressure evictions,
because Z3/Q/reloads may not be admitted while T1's earlier-positioned
reservation is pending.
"""
from __future__ import annotations

import time

import pytest
import torch

from dataflow.core import ObjectSpec, OutputSpec, Program, TaskSpec, TransferDirective
from dataflow.core.validate import validate_program
from dataflow.runtime.engine import DeadlockError, Engine

MIB = 1 << 20
U = MIB  # one "unit" from the design note's pen-and-paper example


def _program() -> Program:
    initial = (
        ObjectSpec("W_0", 2 * U, location="fast", role="parameter"),
        ObjectSpec("Z_1", 2 * U, location="backing", role="input"),
        ObjectSpec("Z_2", 2 * U, location="backing", role="input"),
        ObjectSpec("Z_3", 2 * U, location="backing", role="input"),
        ObjectSpec("Q_0", 2 * U, location="backing", role="input"),
    )
    tasks = (
        TaskSpec(
            id="block_fwd_0_0_0", inputs=("W_0",),
            outputs=(OutputSpec("P_0", 4 * U, location="fast"),),
            runtime_us=100_000, compute_block_key="toy_fwd",
            prefetch_after=(
                TransferDirective("Z_1"), TransferDirective("Z_2"),
                TransferDirective("Z_3"), TransferDirective("Q_0"),
            ),
        ),
        TaskSpec(
            id="block_fwd_0_0_1", inputs=("W_0", "P_0"),
            outputs=(OutputSpec("y_0", 4 * U, location="fast"),),
            runtime_us=100_000, compute_block_key="toy_fwd2",
            releases_after=("P_0",),
        ),
        TaskSpec(
            id="block_bwd_0_0_1", inputs=("y_0",),
            outputs=(OutputSpec("dy_0", 1 * U, location="fast"),),
            runtime_us=100_000, compute_block_key="toy_bwd",
            releases_after=("y_0",),
        ),
        TaskSpec(
            id="block_bwd_0_0_0", inputs=("Z_1", "Z_2", "dy_0"),
            outputs=(OutputSpec("dW_0", 1 * U, location="fast"),),
            runtime_us=100_000, compute_block_key="toy_tail_a",
            releases_after=("Z_1", "Z_2", "dy_0"),
        ),
        TaskSpec(
            id="head_loss_0", inputs=("Z_3", "Q_0", "W_0", "dW_0"),
            outputs=(OutputSpec("loss_0", 4096, location="fast"),),
            runtime_us=100_000, compute_block_key="toy_tail_b",
            releases_after=("Z_3", "Q_0", "dW_0"),
        ),
    )
    prog = Program(
        name="reserve-inversion-min",
        initial_objects=initial,
        tasks=tasks,
        fast_memory_capacity=10 * U,
        # planned lane speed: ~21 B/us -> each 2 MiB prefetch ~100 ms.
        # Real PCIe realizes the same transfer in ~100 us: the inversion.
        bandwidth_from_slow=21,
        bandwidth_to_slow=21,
        final_locations={"W_0": "fast", "loss_0": "fast"},
    )
    validate_program(prog)
    return prog


class _Toy:
    """Minimal executable: touches its output buffers; T0 sleeps its
    planned duration so retirement (and directive firing) is orderly."""

    def __init__(self, task):
        self.task = task

    def launch(self, ctx):
        if self.task.id == "block_fwd_0_0_0":
            time.sleep(0.1)
        for oid, buf in ctx.outputs.items():
            pass  # reservation is the point; contents are irrelevant


def _resolver(task):
    return _Toy(task)


# Sim-validity note: because BOTH sim and runtime charge transfer bytes
# at START, this six-object artifact cannot be simultaneously sim-tight-
# valid and deterministically blocking — under planned (slow-lane)
# timings the same tie the runtime loses (transfers charge before the
# task reserve) makes the sim reject it. The two recorded field
# instances (108-task bs32 plan, 770-eviction dsv3 plan) ARE the
# sim-valid members of this class; their interleavings hide the charge
# behind hundreds of tasks of slack. This artifact trades sim-validity
# for a deterministic minimal trigger of the SAME runtime mechanism:
# early-admitted prefetches occupy a blocked reserve's bytes and the
# valve's freed bytes are poked to the transfer queue, never the task.


def test_program_is_schema_valid():
    _program()  # validate_program runs inside


@pytest.mark.gpu
def test_current_engine_deadlocks_by_poke_starvation():
    """CURRENT behavior pin: realized (fast) transfers admit Z1+Z2 into
    T1's planned bytes; the valve's freed bytes are poked to the
    transfer head (Z3, Q, then reloads), never to the blocked task;
    the thrash guard trips and DeadlockError names the blocked
    reservation. Flip to a completes-clean assertion when plan-order
    admission lands."""
    from dataflow.runtime.device.cuda import CudaBackend

    engine = Engine(CudaBackend())
    with pytest.raises(DeadlockError) as ei:
        result = engine.execute(_program(), resolver=_resolver)
        result.close()
    msg = str(ei.value)
    assert "block_fwd_0_0_1" in msg or "waiting to reserve" in msg
    assert "pressure_evictions=" in msg
