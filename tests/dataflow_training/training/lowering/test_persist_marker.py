"""Lowerings assign the persistent marker at emission: parameters,
optimizer state and cross-step accumulators are marked; inputs are
not. Checkpoint selection is exactly this marker — no prefixes.

Tests:
- test_llama3_marks_state_not_data: every W_*/O_* spec in a lowered llama3 program is persistent and every input spec is not.
- test_moe_aux_rides_the_marker: an MoE family's Aux_* cross-step accumulator objects (expert counts among their fields) carry the marker like any other persistent state.
"""
from dataflow_training.model_families.families import resolve_family
from dataflow_training.model_families.llama3 import ShapedLlamaConfig
from dataflow_training.model_families.olmoe.model import ShapedOlmoeConfig


def test_llama3_marks_state_not_data():
    program = resolve_family(ShapedLlamaConfig.tiny()).lower(
        ShapedLlamaConfig.tiny())
    state = [s for s in program.initial_objects
             if s.id.startswith(("W_", "O_"))]
    data = [s for s in program.initial_objects if s.role == "input"]
    assert state and data
    assert all(s.persistent for s in state), \
        [s.id for s in state if not s.persistent]
    assert not any(s.persistent for s in data), \
        [s.id for s in data if s.persistent]


def test_moe_aux_rides_the_marker():
    cfg = ShapedOlmoeConfig.tiny()
    program = resolve_family(cfg).lower(cfg)
    aux = [s for s in program.initial_objects
           if s.id.startswith("Aux_")]
    assert aux, "an MoE lowering emits cross-step accumulator objects"
    assert all(s.persistent for s in aux), \
        [s.id for s in aux if not s.persistent]
