"""The run failure-mode contract.

A run-level failure (a task raising, an unforeseen device error) or a cancel
comes back as a RunResult carrying a FAILED/CANCELLED outcome — the run never
propagates an exception, so no traceback frame outlives the buffers it
referenced. The engine's OWN invariant violations still raise, scrubbed of any
frame that could reach a view. The ordered cleanup (drain -> invalidate ->
reclaim) runs on every terminal transition: a healthy failed run leaves its
session reusable, while a drain that hits a corrupted context marks the session
unrecoverable.

Tests:
- test_task_raise_returns_failed_outcome: a task raising at launch yields a
  returned FAILED RunResult with copied kind/message/task_id/traceback text and
  no propagated exception.
- test_success_outcome_is_succeeded: a normal run returns a SUCCEEDED outcome.
- test_failed_outcome_carries_full_diagnostics: a FAILED outcome carries kind + message + task_id + the full formatted traceback, and raise_if_failed surfaces all of it.
- test_outcome_serializes_uniformly: a RunOutcome roundtrips through plain JSON (dataclasses.asdict), so the wire and in-process paths carry the same fields.
- test_engine_invariant_raises_scrubbed: an engine-invariant violation (ledger
  over-commit) raises its own type with no __context__/__cause__ chain.
- test_cancel_returns_cancelled_outcome: a set cancel_event yields a CANCELLED
  outcome, not a raise.
- test_inv2_drain_runs_on_failure: the abort-drain step runs exactly once on a
  failed run.
- test_healthy_failed_session_reusable: a session survives a healthy failed run
  and a subsequent run on it succeeds.
- test_corrupted_context_marks_session_unrecoverable: a drain that raises marks
  the session unrecoverable and every later run on it refuses to start.
- test_task_raise_no_crash_on_cuda: on real hardware a task raising yields a
  FAILED outcome and the process survives to run again.
"""
import threading

import pytest

from dataflow.core import ObjectSpec, OutputSpec, Program, TaskSpec
from dataflow.runtime import Engine, ExecutionError
from dataflow.runtime.device.fake import FakeBackend
from dataflow.runtime.engine import RunOutcomeKind, Session
from dataflow.runtime.executable import synthetic_resolver
from dataflow.runtime.ledger import LedgerError


class RaisingExecutable:
    def __init__(self, message):
        self.message = message

    def launch(self, ctx):
        raise RuntimeError(self.message)


class RaisingResolver:
    """Fails one named task at launch and runs every other task normally —
    models a task/kernel raising mid-run."""

    def __init__(self, target_task_id, message):
        self.target = target_task_id
        self.message = message

    def __call__(self, task):
        if task.id == self.target:
            return RaisingExecutable(self.message)
        return synthetic_resolver(task)


class SpyDrainBackend(FakeBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drain_calls = 0

    def drain_aborted(self):
        self.drain_calls += 1
        return super().drain_aborted()


class CorruptDrainBackend(FakeBackend):
    """The abort-drain itself fails — a sticky/corrupted CUDA context."""

    def drain_aborted(self):
        raise RuntimeError("sticky CUDA context (simulated illegal access)")


def one_task_program(name="run", cap=1 << 20):
    return Program(
        name=name,
        initial_objects=(ObjectSpec(id="a", size_bytes=1024, location="fast"),),
        tasks=(TaskSpec(id="t0", inputs=("a",),
                        outputs=(OutputSpec(id="b", size_bytes=1024,
                                            location="fast"),),
                        runtime_us=1.0),),
        fast_memory_capacity=cap,
    )


def over_capacity_program():
    return Program(
        name="over",
        initial_objects=(ObjectSpec(id="a", size_bytes=2_000, location="fast"),),
        tasks=(),
        fast_memory_capacity=1_000,
    )


def test_task_raise_returns_failed_outcome():
    result = Engine(FakeBackend()).execute(
        one_task_program(), resolver=RaisingResolver("t0", "kaboom in t0"))
    # RETURNED, not raised
    assert result.outcome.kind is RunOutcomeKind.FAILED
    assert result.outcome.is_failed
    assert "kaboom in t0" in result.outcome.message
    assert result.outcome.task_id == "t0"
    assert result.outcome.traceback_text          # full formatted text present
    # copied-out diagnostics are plain strings — no device handle escapes
    assert isinstance(result.outcome.message, str)
    assert isinstance(result.outcome.traceback_text, str)


def test_success_outcome_is_succeeded():
    result = Engine(FakeBackend()).execute(one_task_program())
    assert result.outcome.is_success
    assert result.outcome.kind is RunOutcomeKind.SUCCEEDED


def test_failed_outcome_carries_full_diagnostics():
    # the caller must learn WHY, not merely that it failed: kind + message +
    # task_id + the full formatted (view-free) traceback
    result = Engine(FakeBackend()).execute(
        one_task_program(), resolver=RaisingResolver("t0", "why it broke"))
    assert result.outcome.is_failed
    assert result.outcome.task_id == "t0"
    assert "why it broke" in result.outcome.message
    assert "Traceback" in result.outcome.traceback_text
    assert "why it broke" in result.outcome.traceback_text
    # raise_if_failed surfaces all of it (for the in-process parity helpers)
    with pytest.raises(RuntimeError) as excinfo:
        result.raise_if_failed()
    text = str(excinfo.value)
    assert "why it broke" in text
    assert "t0" in text
    assert "Traceback" in text


def test_outcome_serializes_uniformly():
    # the wire carries the outcome as plain JSON data (dataclasses.asdict), so a
    # client reads exactly the fields an in-process caller sees on result.outcome
    import dataclasses
    import json

    result = Engine(FakeBackend()).execute(
        one_task_program(), resolver=RaisingResolver("t0", "boom"))
    wire = json.loads(json.dumps(dataclasses.asdict(result.outcome)))
    assert wire["kind"] == RunOutcomeKind.FAILED
    assert wire["task_id"] == "t0"
    assert "boom" in wire["message"]
    assert "Traceback" in wire["traceback_text"]


def test_engine_invariant_raises_scrubbed():
    with pytest.raises(LedgerError, match="over-commit") as excinfo:
        Engine(FakeBackend()).execute(over_capacity_program())
    # scrubbed: no chain back to the run's frames (no view reachable)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_cancel_returns_cancelled_outcome():
    cancel = threading.Event()
    cancel.set()
    result = Engine(FakeBackend()).execute(one_task_program(),
                                           cancel_event=cancel)
    assert result.outcome.kind is RunOutcomeKind.CANCELLED
    assert not result.outcome.is_success


def test_inv2_drain_runs_on_failure():
    backend = SpyDrainBackend()
    result = Engine(backend).execute(
        one_task_program(), resolver=RaisingResolver("t0", "boom"))
    assert result.outcome.is_failed
    assert backend.drain_calls == 1        # the drain step ran exactly once


def test_healthy_failed_session_reusable():
    session = Session(backend=FakeBackend())
    engine = Engine(session.backend, session=session)
    failed = engine.execute(one_task_program(),
                            resolver=RaisingResolver("t0", "boom"))
    assert failed.outcome.is_failed
    assert session.unrecoverable is False           # healthy failure
    # the session is clean -> a subsequent run on it succeeds
    ok = engine.execute(one_task_program())
    assert ok.outcome.is_success


def test_corrupted_context_marks_session_unrecoverable():
    session = Session(backend=CorruptDrainBackend())
    engine = Engine(session.backend, session=session)
    failed = engine.execute(one_task_program(),
                            resolver=RaisingResolver("t0", "boom"))
    assert failed.outcome.is_failed
    assert session.unrecoverable is True            # the drain itself failed
    # every later run on the poisoned session refuses to start
    with pytest.raises(ExecutionError, match="unrecoverable"):
        engine.execute(one_task_program())


@pytest.mark.gpu
def test_task_raise_no_crash_on_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from dataflow.runtime.device.cuda import CudaBackend

    # a task raising on real hardware -> a returned FAILED outcome, no segfault
    failed = Engine(CudaBackend()).execute(
        one_task_program(), resolver=RaisingResolver("t0", "boom on device"))
    assert failed.outcome.is_failed
    assert "boom on device" in failed.outcome.message
    # the process survived the failure and issues another run that also returns
    # a structured outcome rather than crashing (the success-on-CUDA path is
    # covered by the client fetch-surface gate)
    again = Engine(CudaBackend()).execute(
        one_task_program(), resolver=RaisingResolver("t0", "boom again"))
    assert again.outcome.is_failed
    failed.close()      # release the client-owned pool (slab + overflow arena);
    again.close()       # a failed run's RunResult still owns it until closed
