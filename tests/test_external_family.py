"""The external-client gate: a model family defined entirely outside
``src/`` (tests/fixtures/external_family/toy_family.py) loads through the
plugin seam, lowers a structurally clean program, passes the structural
contract validator, and composes with the SERVICE path — registration +
``resolver_for`` build an executable for every task the family emits.

CPU-only: resolution builds executables, never launches them; no CUDA
imports at module scope (dataflow imports live inside the tests, the
test_plugins.py convention).
"""
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "external_family"


def cleanup_toyfam():
    import dataflow_training.model_families.families as F
    from dataflow.service import registry as service_registry

    F._FAMILIES.pop("toyfam", None)
    F._cache.pop("toyfam", None)
    for key in [k for k in service_registry._CACHE if "toyfam" in k[1]]:
        del service_registry._CACHE[key]
    sys.modules.pop("toy_family", None)


def load_toyfam(monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURE_DIR))
    import dataflow_training.model_families.families as F

    F.load_plugins(explicit=["toy_family"])
    return F


def test_external_family_registers_lowers_and_validates(monkeypatch):
    F = load_toyfam(monkeypatch)
    try:
        import toy_family

        from dataflow.core import validate_program

        fam = F.family("toyfam")
        cfg = toy_family.ToyConfig.tiny()
        # config-type dispatch reaches the EXTERNAL family
        assert F.resolve_family(cfg) is fam
        assert fam.config_type is toy_family.ToyConfig

        # the lowering is structurally clean (validate_program raises on
        # any id/edge/size problem) and keeps the repo naming grammar
        prog = fam.lower(cfg)
        validate_program(prog)
        keys = {t.compute_block_key for t in prog.tasks}
        assert {"toy_block_fwd", "toy_block_bwd", "embed_fwd", "head_loss",
                "embed_bwd", "optimizer_block"} <= keys

        # planner-style re-lowering with a recompute level still validates
        relowered = fam.lower(cfg, recompute_levels={"A_0_0_0": 1})
        validate_program(relowered)
        assert any(t.compute_block_key == "toy_block_recompute"
                   for t in relowered.tasks)

        # the structural contract validator reports no broken surface
        assert F.validate_family("toyfam") == []
    finally:
        cleanup_toyfam()


def test_external_family_composes_with_service_path(monkeypatch):
    F = load_toyfam(monkeypatch)
    try:
        import dataclasses

        import toy_family

        from dataflow.service.registry import resolver_for
        from dataflow_training.register import register_all

        assert "model_family" in register_all()
        fam = F.family("toyfam")
        cfg = toy_family.ToyConfig.tiny()
        spec = {"kind": "model_family", "family": "toyfam",
                "cfg": dataclasses.asdict(cfg)}
        resolver = resolver_for(spec)

        # every task of the family's lowered program resolves to an
        # executable (resolution only — nothing launches on CPU), and the
        # block tasks resolve to the family's OWN executables
        prog = fam.lower(cfg)
        own = 0
        for task in prog.task_by_id().values():
            ex = resolver(task)
            assert hasattr(ex, "launch"), task.id
            if task.compute_block_key == "toy_block_fwd":
                assert isinstance(ex, toy_family.ToyBlockFwd)
                own += 1
            if task.compute_block_key == "toy_block_bwd":
                assert isinstance(ex, toy_family.ToyBlockBwd)
                own += 1
        assert own == 2 * cfg.grad_accum_rounds  # fwd+bwd per round

        # init-as-program composes too: the shared "family_init" task
        # resolves through the same registered kind
        init_prog = F.build_init_program(fam, cfg, seed=0)
        init_task = init_prog.task_by_id()["family_init_0"]
        assert hasattr(resolver(init_task), "launch")
    finally:
        cleanup_toyfam()
