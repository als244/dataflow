"""Round-prologue mechanism gates (aux refactor): ``round_prologue=True``
opens every round with a ``prologue_round_{s}_{r}`` task whose 4-byte
int32 output IS the object-backed ``current_round`` value, and whose launch
publishes it into the engine's mutable ``run_values`` channel (``run_args``
stays immutable). The flag defaults ON —
EVERY family opens each round with the prologue (it also publishes the
round's content token count and materializes Segments); passing
``round_prologue=False`` builds the bare chain (planner unit tests).

Tests:
- test_prologue_round_structure: each round gets a prologue_round task whose 4-byte current_round output precedes that round's embed_fwd.
- test_flag_off_omits_prologues_and_default_adds_one_per_round: round_prologue=False emits no prologue tasks while the default build adds exactly one per round.
- test_round_prologue_publishes_round_index_via_run_values_and_object: an engine run surfaces each round's index through both the run_values channel and the pinned current_round object.
"""
from dataclasses import dataclass, replace

import pytest

from dataflow.core.validate import validate_program
from dataflow_training.model_families.llama3 import ShapedLlamaConfig
from dataflow_training.lowering.shaped_program import (
    ShapedHardware,
    build_shaped_program,
    roofline_block_kind_spec,
)

TINY = replace(ShapedLlamaConfig.tiny(), grad_accum_rounds=2)


def build_with_prologue(cfg):
    hw = ShapedHardware()
    return build_shaped_program(
        cfg, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(cfg, hw)},
        round_prologue=True,
    )


def test_prologue_round_structure():
    program = build_with_prologue(TINY)
    validate_program(program)
    by_id = {t.id: t for t in program.tasks}
    order = [t.id for t in program.tasks]
    for r in range(TINY.grad_accum_rounds):
        pid = f"prologue_round_0_{r}"
        assert pid in by_id
        t = by_id[pid]
        assert t.compute_block_key == "prologue_round"
        assert t.block_params["round"] == r
        assert [o.id for o in t.outputs] == [f"current_round_0_{r}"]
        assert t.outputs[0].size_bytes == 4
        # the prologue opens its round: it precedes the round's embed_fwd
        assert order.index(pid) < order.index(f"embed_fwd_0_{r}")


def test_flag_off_omits_prologues_and_default_adds_one_per_round():
    """round_prologue=False builds the bare chain — no prologue tasks —
    and the DEFAULT build opens every round with one (the universal-
    prologue contract; the hash tripwire pins the bytes)."""
    hw = ShapedHardware()
    program = build_shaped_program(
        TINY, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(TINY, hw)},
        round_prologue=False,
    )
    assert not [t for t in program.tasks if t.compute_block_key == "prologue_round"]
    default = build_shaped_program(
        TINY, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(TINY, hw)},
    )
    prologues = [t for t in default.tasks
                 if t.compute_block_key == "prologue_round"]
    assert len(prologues) == TINY.grad_accum_rounds


# --- GPU e2e: the value flows through both channels ---------------------------

@dataclass
class RoundValueRecorder:
    """Wraps a block executable; records ctx.run_values["current_round"]
    as seen at launch time, keyed by the task's round index."""

    inner: object
    seen: list

    def launch(self, ctx) -> None:
        r = int(ctx.task.id.rsplit("_", 1)[-1])   # embed_fwd_{s}_{r}
        rv = None if ctx.run_values is None else ctx.run_values.get("current_round")
        self.seen.append((r, rv))
        self.inner.launch(ctx)


@dataclass
class PrologueResolver:
    """The family resolver plus the round-prologue executable, with a
    recorder on embed_fwd (runs once per round, right after the prologue)."""

    base: object
    prologue: object
    seen: list

    def __call__(self, task):
        if task.compute_block_key == "prologue_round":
            return self.prologue
        exe = self.base(task)
        if task.compute_block_key == "embed_fwd":
            return RoundValueRecorder(exe, self.seen)
        return exe


@pytest.mark.gpu
def test_round_prologue_publishes_round_index_via_run_values_and_object():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    pytest.importorskip("cuda.bindings")
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow_training.data.segments import uniform_segments
    from dataflow_training.blocks.base_blocks import RoundPrologue
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.llama3.blocks import build_resolver
    from dataflow_training.lowering.emit import apply_exact_sizes, object_size_factory
    from dataflow_training.model_families.llama3 import family_layouts, initial_values
    from dataflow_training.lowering.planning import plan_program

    cfg = TINY
    dims, fl = family_layouts(cfg)
    shaped = build_with_prologue(cfg)
    program = apply_exact_sizes(shaped, "llama3-exact",
                                object_size=object_size_factory(dims, fl))
    # pin the LAST round's marker so its bytes survive to readback
    program = replace(program, final_locations={
        **program.final_locations, "current_round_0_1": "backing"})
    planned = plan_program(program, fast_memory_capacity=64 * 1024 * 1024)

    backend = CudaBackend()
    values = initial_values(planned.program, cfg, backend, seed=5)
    seen: list = []
    resolver = PrologueResolver(base=build_resolver(dims),
                                prologue=RoundPrologue(dims), seen=seen)
    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=resolver,
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)},
    )
    # run_values channel: each round's tasks saw their own round index
    assert seen == [(0, 0), (1, 1)], seen
    # object channel: the pinned marker holds its round index
    rec = result.objects.get("current_round_0_1")
    slot = rec.backing or rec.fast
    assert int(torch_view(slot.buffer, (1,), torch.int32)[0]) == 1
    result.close()
