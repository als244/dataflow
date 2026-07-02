from dataflow.core import validate_program
from dataflow.training.shaped_llama3 import ShapedLlamaConfig, build_shaped_llama3

GIB = 1024**3


def test_tiny_validates():
    validate_program(build_shaped_llama3(ShapedLlamaConfig.tiny()))


def test_8b_shape_totals():
    cfg = ShapedLlamaConfig.llama3_8b()
    program = build_shaped_llama3(cfg, fast_memory_capacity=16 * GIB)
    validate_program(program)

    sizes = program.object_sizes()
    param_bytes = sizes["W_embed"] + sizes["W_head"] + sum(sizes[f"W_{i}"] for i in range(32))
    # Llama3-8B bf16 weights are ~15 GiB (8.03B params x 2 bytes)
    assert 14.5 * GIB < param_bytes < 15.5 * GIB

    # optimizer state is exactly 2x params here
    opt_bytes = sizes["O_embed"] + sizes["O_head"] + sum(sizes[f"O_{i}"] for i in range(32))
    assert opt_bytes == 2 * param_bytes

    # one full round of saved context alone approaches the 16 GiB budget, and
    # params + saved context together far exceed it -> recompute matters
    a_bytes = sum(sizes[f"A_0_0_{i}"] for i in range(32))
    assert a_bytes > 12 * GIB
    assert param_bytes + a_bytes > 24 * GIB

    # task chain structure: embed + 32 fwd + head + loss + head_bwd + 32 bwd
    # + embed_bwd + 34 optimizer tasks
    assert len(program.tasks) == 1 + 32 + 1 + 1 + 1 + 32 + 1 + 34


def test_grad_accum_mutation_pattern():
    cfg = ShapedLlamaConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=160,
        vocab_size=512, seq_len=64, batch=1, grad_accum_rounds=2,
    )
    program = build_shaped_llama3(cfg)
    validate_program(program)
    by_id = program.task_by_id()
    # round 0 creates dW, round 1 mutates it
    assert any(o.id == "dW_0_1" for o in by_id["block_bwd_0_0_1"].outputs)
    assert "dW_0_1" in by_id["block_bwd_0_1_1"].inputs
    assert by_id["block_bwd_0_1_1"].mutates == ("dW_0_1",)
    # optimizer mutates W and O
    assert set(by_id["optimizer_0_1"].mutates) == {"W_1", "O_1"}


def test_recompute_variant_moves_A_production():
    cfg = ShapedLlamaConfig.tiny()
    levels = {"A_0_0_1": 1}
    program = build_shaped_llama3(cfg, recompute_levels=levels)
    validate_program(program)
    by_id = program.task_by_id()
    assert all(o.id != "A_0_0_1" for o in by_id["block_fwd_0_0_1"].outputs)
    assert any(o.id == "A_0_0_1" for o in by_id["block_recompute_0_0_1"].outputs)
    # layer 0 unchanged
    assert any(o.id == "A_0_0_0" for o in by_id["block_fwd_0_0_0"].outputs)
    assert "block_recompute_0_0_0" not in by_id


def test_rewrites_cover_all_saved_contexts():
    cfg = ShapedLlamaConfig.tiny()
    program = build_shaped_llama3(cfg)
    ids = {rw.object_id for rw in program.recompute_rewrites}
    assert ids == {f"A_0_0_{i}" for i in range(cfg.n_layers)}
