import pytest

from dataflow.core import (
    load_program,
    program_from_dict,
    program_to_dict,
    save_program,
    validate_program,
)
from dataflow.training.shaped_llama3 import ShapedLlamaConfig, build_shaped_llama3


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
