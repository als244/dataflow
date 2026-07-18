"""Registration smoke: a bare registry + register_all resolves every
family through the ONE contract key ("kind"); unknown kinds fail loudly
naming what IS registered. CPU-only — resolution builds executables,
it does not launch them."""
import pytest

from dataflow.service.registry import (
    registered_kinds,
    resolver_for,
)
from dataflow.service.wire import ServiceError
from dataflow_training.model_families.families import build_init_program, family
from dataflow_training.register import canonical_spec, register_all

FAMILIES = ("llama3", "qwen3", "olmoe", "dsv3", "dsv32", "glm52",
            "qwen3moe", "qwen35", "qwen35moe")


def smoke_cfg_dict(name: str) -> dict:
    from dataflow_training.run.presets import cfg_dict

    fam = family(name)
    cfg = fam.config_type.tiny()
    return cfg_dict(cfg)


def test_register_all_resolves_every_family():
    register_all()
    assert "model_family" in registered_kinds()
    for name in FAMILIES:
        spec = canonical_spec(name, smoke_cfg_dict(name))
        resolver = resolver_for(spec)
        fam = family(name)
        cfg = fam.config_type(**spec["cfg"])
        program = fam.lower(cfg)
        # every training task resolves...
        for task in program.tasks:
            assert resolver(task) is not None, (name, task.id)
        # ...and so does the init program's one task
        init = build_init_program(fam, cfg, seed=7)
        assert resolver(init.tasks[0]) is not None, name
        assert init.tasks[0].compute_block_key == "family_init"
        assert not init.initial_objects
        assert {o.id for o in init.tasks[0].outputs} == {
            s.id for s in program.initial_objects}


def test_unknown_kind_is_loud():
    register_all()
    with pytest.raises(ServiceError) as e:
        resolver_for({"kind": "nonsense"})
    assert "model_family" in str(e.value)
    with pytest.raises(ServiceError) as e:
        resolver_for({"family": "llama3"})   # no kind at all
    assert "kind" in str(e.value)
