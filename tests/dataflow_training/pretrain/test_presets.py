"""Gates for the model presets + training config: shapes, param counts,
the locked token budget, cfg_dict round-trip, and that every preset lowers
(planning of the full ladder is exercised by the runs / scaling tool)."""
import pytest

from dataflow_training.run import presets as P
from dataflow_training.model_families.families import resolve_family
from dataflow_training.model_families.llama3 import ShapedLlamaConfig
from dataflow_training.lowering.planning import plan_program

GIB = 1024**3


def test_locked_token_budget():
    cfg = P.preset("l3_1b")
    assert cfg.seq_len == 2048 and cfg.batch == 4
    assert cfg.max_tokens == 8192                       # tokens / round
    assert cfg.grad_accum_rounds == 8
    assert P.tokens_per_step(cfg) == 65536          # ~64K tokens / step
    assert cfg.vocab_size == 50304


@pytest.mark.parametrize("name", P.LADDER_NAMES)
def test_preset_shapes_consistent(name):
    cfg = P.preset(name)
    # head_dim 64 throughout; GQA divides; heads span d_model
    assert cfg.head_dim == 64
    assert cfg.n_heads % cfg.n_kv_heads == 0
    assert cfg.n_heads * cfg.head_dim == cfg.d_model
    assert cfg.d_ff == 4 * cfg.d_model
    assert cfg.vocab_size == 50304


def test_param_counts_monotone_and_1b_is_1b():
    counts = [P.param_counts(P.preset(n))["total"] for n in P.LADDER_NAMES]
    assert counts == sorted(counts)                 # strictly increasing ladder
    pc = P.param_counts(P.preset("l3_1b"))
    assert 1.0e9 < pc["total"] < 1.3e9              # ~1.18B total
    assert 0.9e9 < pc["non_embedding"] < 1.0e9


def test_cfg_dict_round_trip_matches_dims():
    for name in P.LADDER_NAMES:
        cfg = P.preset(name)
        rebuilt = ShapedLlamaConfig(**P.cfg_dict(cfg))
        d1 = resolve_family(cfg).derive_dims(cfg)
        d2 = resolve_family(rebuilt).derive_dims(rebuilt)
        for attr in ("d_model", "n_heads", "n_kv_heads", "d_ff",
                     "vocab_size", "max_tokens", "seq_len"):
            assert getattr(d1, attr) == getattr(d2, attr)


@pytest.mark.parametrize("name", P.LADDER_NAMES)
def test_preset_lowers(name):
    """Lowering exercises the family layouts + exact packed sizes."""
    cfg = P.preset(name)
    prog = resolve_family(cfg).lower(cfg)
    assert len(prog.tasks) > 0
    assert prog.initial_objects  # W/O/tokens/targets declared


def test_smoke_preset_lowers_and_plans():
    cfg = P.smoke_preset()
    assert cfg.vocab_size == 50304  # real vocab -> consumes fineweb ids
    prog = resolve_family(cfg).lower(cfg)
    planned = plan_program(prog, fast_memory_capacity=8 * GIB)
    assert planned.peak_fast_bytes <= 8 * GIB


def test_resolve_preset_bare_unique_names_across_families():
    assert type(P.resolve_preset("gpt2_124m")).__name__ == "ShapedGpt2Config"
    assert type(P.resolve_preset("l3_1b")).__name__ == "ShapedLlamaConfig"
    assert type(P.resolve_preset("llama3_8b")).__name__ == "ShapedLlamaConfig"
    assert type(P.resolve_preset("olmoe_7b")).__name__ == "ShapedOlmoeConfig"
    assert (type(P.resolve_preset("qwen35moe_20l")).__name__
            == "ShapedQwen35MoeConfig")


def test_resolve_preset_ambiguous_name_lists_qualified_forms():
    with pytest.raises(KeyError, match="gpt2:tiny"):
        P.resolve_preset("tiny")


def test_resolve_preset_qualified_name_disambiguates():
    assert type(P.resolve_preset("gpt2:tiny")).__name__ == "ShapedGpt2Config"
    assert (type(P.resolve_preset("llama3:tiny")).__name__
            == "ShapedLlamaConfig")


def test_resolve_preset_unknown_name_points_at_the_table():
    with pytest.raises(KeyError, match="builtin_models"):
        P.resolve_preset("nope_123")
