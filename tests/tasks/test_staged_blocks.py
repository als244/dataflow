"""Staged block authoring: the single stage list drives everything.

- completeness: every declared context field is emitted by some stage
  (a field nobody emits would make recompute silently wrong);
- derived truncation: the recompute boundary excludes at least one stage
  (the waste tripwire — recompute == full forward means boundary work
  crept back in);
- the boundary sits exactly after the last context-emitting stage.
"""
from dataflow_training.model_families.llama3_blocks import BlockFwd
from dataflow_training.blocks.layouts import activation_layout
from dataflow_training.model_families.llama3 import dims_of
from dataflow_training.model_families.llama3 import ShapedLlamaConfig


def test_stage_context_completeness():
    dims = dims_of(ShapedLlamaConfig.tiny())
    declared = {f.name for f in activation_layout(dims).fields}
    emitted = BlockFwd.context_fields_emitted()
    assert declared == emitted, declared ^ emitted


def test_derived_recompute_excludes_boundary_work():
    n = BlockFwd.recompute_stage_count()
    assert n < len(BlockFwd.STAGES), (
        "recompute runs EVERY stage — boundary work (y-only stages) has "
        "crept back in; recompute must stop at the last context emission"
    )
    # boundary is exactly after the last emitting stage
    last_emit = max(i for i, (_, _, e) in enumerate(BlockFwd.STAGES) if e)
    assert n == last_emit + 1
