"""Program serialization: dict and JSON round-trips preserve program
equality across variants, and program_from_dict guards schema version and
comm-group typing.

Tests:
- test_roundtrip_equality: a program survives program_to_dict/from_dict and save/load unchanged.
- test_unknown_schema_version_rejected: program_from_dict raises on an unsupported schema_version.
- test_grad_accum_variant_roundtrips: a grad-accum program round-trips through dict form unchanged.
- test_recompute_variant_roundtrips: a recompute-annotated program keeps its recompute tasks and round-trips unchanged.
- test_comm_groups_roundtrip_and_validation: comm_groups serializes only when set, round-trips exactly, and the validator rejects an empty role name.
"""
import pytest

from dataflow.core import (
    load_program,
    program_from_dict,
    program_to_dict,
    save_program,
    validate_program,
)
from dataflow_training.model_families.llama3 import ShapedLlamaConfig, build_shaped_llama3


def test_roundtrip_equality(tmp_path):
    program = build_shaped_llama3(ShapedLlamaConfig.tiny(), fast_memory_capacity=1_000_000)
    validate_program(program)

    again = program_from_dict(program_to_dict(program))
    assert again == program

    path = tmp_path / "p.json"
    save_program(program, path)
    assert load_program(path) == program


def test_unknown_schema_version_rejected():
    program = build_shaped_llama3(ShapedLlamaConfig.tiny())
    d = program_to_dict(program)
    d["schema_version"] = "dataflow-rt/v999"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        program_from_dict(d)


def test_grad_accum_variant_roundtrips():
    cfg = ShapedLlamaConfig.tiny()
    cfg = ShapedLlamaConfig(**{**cfg.__dict__, "grad_accum_rounds": 2})
    program = build_shaped_llama3(cfg)
    validate_program(program)
    assert program_from_dict(program_to_dict(program)) == program


def test_recompute_variant_roundtrips():
    cfg = ShapedLlamaConfig.tiny()
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    program = build_shaped_llama3(cfg, recompute_levels=levels)
    validate_program(program)
    ids = {t.id for t in program.tasks}
    assert f"block_recompute_0_0_{cfg.n_layers - 1}" in ids
    assert program_from_dict(program_to_dict(program)) == program


def test_comm_groups_roundtrip_and_validation():
    """comm_groups is the typed home for group ROLE names (purpose ->
    role): serialized only when set, round-trips exactly, and the
    validator rejects malformed entries."""
    from dataflow.core.validate import ValidationError

    cfg = ShapedLlamaConfig.tiny()
    program = build_shaped_llama3(cfg)
    # solo lowering: no task communicates, nothing serializes
    assert all(not t.comm_groups for t in program.tasks)
    assert all("comm_groups" not in td
               for td in program_to_dict(program)["tasks"])

    from dataflow_training.lowering.shaped_program import (
        ShapedHardware,
        build_shaped_program,
        roofline_block_kind_spec,
    )

    hw = ShapedHardware()
    fleet = build_shaped_program(
        cfg, hw=hw, family="llama3-shaped",
        kinds={"block": roofline_block_kind_spec(cfg, hw)},
        dp_group="dp")
    validate_program(fleet)
    opts = [t for t in fleet.tasks if t.id.startswith("optimizer_")]
    assert opts and all(t.comm_groups == {"dp": "dp"} for t in opts)
    fwd = [t for t in fleet.tasks if t.id.startswith("block_fwd_")]
    assert fwd and all(not t.comm_groups for t in fwd)
    assert program_from_dict(program_to_dict(fleet)) == fleet

    bad = program_from_dict(program_to_dict(fleet))
    object.__setattr__(bad.tasks[0], "comm_groups", {"dp": ""})
    with pytest.raises(ValidationError, match="comm_groups"):
        validate_program(bad)
