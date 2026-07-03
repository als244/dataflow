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


def test_interleaved_optimizer_fires_at_final_grad_mutation():
    """Default placement: optimizer_i sits right after the LAST round's
    block_bwd_i (its dW's final mutation), not in a tail phase; ids and the
    task SET are identical to tail mode so plans/profiles stay comparable."""
    from dataclasses import replace as dc_replace

    cfg = dc_replace(ShapedLlamaConfig.tiny(), grad_accum_rounds=3)
    inter = build_shaped_llama3(cfg)
    tail = build_shaped_llama3(dc_replace(cfg, optimizer_placement="tail"))
    validate_program(inter)
    validate_program(tail)
    assert {t.id for t in inter.tasks} == {t.id for t in tail.tasks}

    order = [t.id for t in inter.tasks]
    idx = {tid: k for k, tid in enumerate(order)}
    last = cfg.grad_accum_rounds - 1
    for i in range(cfg.n_layers):
        assert idx[f"optimizer_0_{i}"] == idx[f"block_bwd_0_{last}_{i}"] + 1
    assert idx[f"optimizer_head_0"] == idx[f"head_bwd_0_{last}"] + 1
    assert idx[f"optimizer_embed_0"] == idx[f"embed_bwd_0_{last}"] + 1
    # interleaved: no optimizer task may follow embed's (the chain's closer)
    assert order[-1] == "optimizer_embed_0"

    # tail mode keeps the legacy suffix: every optimizer after every backward
    tail_order = [t.id for t in tail.tasks]
    first_opt = min(k for k, tid in enumerate(tail_order) if tid.startswith("optimizer"))
    assert all(tid.startswith("optimizer") for tid in tail_order[first_opt:])


def test_interleaved_optimizer_respects_reader_ordering():
    """No task after optimizer_i may READ W_i or O_i within the step (the
    mutation must be the last touch), and dW_i must not be referenced after
    its optimizer consumes it — the properties interleaving relies on."""
    from dataclasses import replace as dc_replace

    cfg = dc_replace(ShapedLlamaConfig.tiny(), grad_accum_rounds=2, num_steps=2)
    program = build_shaped_llama3(cfg)
    idx = {t.id: k for k, t in enumerate(program.tasks)}
    for s in range(cfg.num_steps):
        step_end = idx[f"optimizer_embed_{s}"]  # interleaved: the step's closer
        for i in range(cfg.n_layers):
            k_opt = idx[f"optimizer_{s}_{i}"]
            for t in program.tasks[k_opt + 1 : step_end + 1]:
                assert f"W_{i}" not in t.inputs, (t.id, f"reads W_{i} after optimizer_{s}_{i}")
                assert f"dW_{s}_{i}" not in t.inputs, (t.id, "reads consumed grad")


def test_tied_embeddings_chain_structure():
    """Config-gated tied embeddings: one W_embed/O_embed pair serves embed
    AND head; head_bwd (which runs first in the round) creates the shared
    dW_embed, embed_bwd accumulates; no head objects or optimizer_head."""
    from dataclasses import dataclass
    from dataflow.training.planning import plan_program, simulate_program

    @dataclass(frozen=True)
    class TiedCfg(ShapedLlamaConfig):
        tied_embeddings: bool = True

    from dataclasses import replace as dc_replace

    cfg = dc_replace(TiedCfg(**vars(ShapedLlamaConfig.tiny())), grad_accum_rounds=2)
    program = build_shaped_llama3(cfg)
    validate_program(program)

    ids = {o.id for o in program.initial_objects}
    assert "W_head" not in ids and "O_head" not in ids
    by_id = {t.id: t for t in program.tasks}
    assert "optimizer_head_0" not in by_id
    assert by_id["head_fwd_0_0"].inputs[1] == "W_embed"
    # round 0: head_bwd creates dW_embed, embed_bwd mutates it
    assert any(o.id == "dW_embed_0" for o in by_id["head_bwd_0_0"].outputs)
    assert by_id["embed_bwd_0_0"].mutates == ("dW_embed_0",)
    # round 1: both accumulate
    assert by_id["head_bwd_0_1"].mutates == ("dW_embed_0",)
    assert by_id["embed_bwd_0_1"].mutates == ("dW_embed_0",)
    # optimizer_embed consumes the shared gradient and is the only embed/head opt
    assert by_id["optimizer_embed_0"].inputs == ("W_embed", "dW_embed_0", "O_embed")

    # the annotated chain plans + simulates green
    planned = plan_program(program, fast_memory_capacity=4 * 1024 * 1024)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


def test_heterogeneous_kinds_emit_per_kind_keys_and_sizes():
    """LayerKindSpec table: task IDS stay uniform (tooling contract) while
    compute_block_keys, W/A sizes, and rewrite keys follow the layer kind."""
    from dataflow.training.shaped_llama3 import LayerKindSpec

    cfg = ShapedLlamaConfig.tiny()  # 2 layers
    sub = [{"kind": "roofline", "name": "x", "flops": 1, "memory_bytes": 1, "efficiency": "matmul"}]
    kinds = {
        "lin": LayerKindSpec(
            key_prefix="linattn", w_bytes=1000, a_bytes=2000, fwd_us=10.0,
            bwd_us=20.0, recompute_us=9.0, optimizer_us=3.0,
            fwd_subops=sub, bwd_subops=sub, recompute_subops=sub, optimizer_subops=sub,
        ),
        "full": LayerKindSpec(
            key_prefix="gattn", w_bytes=3000, a_bytes=4000, fwd_us=11.0,
            bwd_us=21.0, recompute_us=8.0, optimizer_us=4.0,
            fwd_subops=sub, bwd_subops=sub, recompute_subops=sub, optimizer_subops=sub,
        ),
    }
    program = build_shaped_llama3(
        cfg, kinds=kinds, kind_of=lambda i: "lin" if i == 0 else "full",
    )
    validate_program(program)
    by_id = {t.id: t for t in program.tasks}
    assert by_id["block_fwd_0_0_0"].compute_block_key == "linattn_fwd"
    assert by_id["block_fwd_0_0_1"].compute_block_key == "gattn_fwd"
    assert by_id["block_bwd_0_0_1"].compute_block_key == "gattn_bwd"
    sizes = {o.id: o.size_bytes for o in program.initial_objects}
    assert sizes["W_0"] == 1000 and sizes["W_1"] == 3000
    assert sizes["O_0"] == 2000 and sizes["O_1"] == 6000
    rw = {r.object_id: r for r in program.recompute_rewrites}
    assert rw["A_0_0_0"].r_compute_block_key == "linattn_recompute"
    assert rw["A_0_0_1"].r_compute_block_key == "gattn_recompute"
    assert rw["A_0_0_0"].options[0].saved_bytes == 2000
    assert rw["A_0_0_1"].options[0].saved_bytes == 4000
