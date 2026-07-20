# qwen35moe / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen35MoeConfig.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_docs/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (4 layers): `lin lin lin full`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (lin)` | layer | 1.95 MiB |
| `dW_i (lin)` | layer/step | 1.95 MiB |
| `O_i (lin)` | layer | 3.90 MiB |
| `A (lin)` | layer × round | 228.62 MiB (3.57 KiB/token) |
| `W_i (full)` | layer | 2.19 MiB |
| `dW_i (full)` | layer/step | 2.19 MiB |
| `O_i (full)` | layer | 4.39 MiB |
| `A (full)` | layer × round | 260.12 MiB (4.06 KiB/token) |
| `W_head` | run | 256.50 KiB |
| `W_embed` | run | 256.00 KiB |
| `O_embed` | run | 512.00 KiB |
| `O_head` | run | 513.00 KiB |
| `hidden state (y)` | boundary buffer | 32.00 MiB (512.0 B/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 6 | 8.54 MiB |
| dW (all gradients, per step) | 6 | 8.54 MiB |
| O (all optimizer state) | 6 | 17.08 MiB |
| A (all saved activations, one round) | 4 | 946.00 MiB (14.78 KiB/token) |
| M (all metadata, one round) | 4 | 5.00 MiB (80.0 B/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 256 |
| `n_layers` | 4 |
| `full_attention_interval` | 4 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim` | 64 |
| `partial_rotary_factor` | 0.25 |
| `lin_k_heads` | 2 |
| `lin_v_heads` | 4 |
| `lin_k_head_dim` | 32 |
| `lin_v_head_dim` | 32 |
| `lin_conv_kernel` | 4 |
| `d_ff` | 128 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000000.0 |
| `opt_policy` | adamw |
| `kinds` | ('lin', 'lin', 'lin', 'full') |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `lin` (e.g. layer 0)

**`W_0` weights** — 1.95 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 B |
| `w_qkvz` | bf16 | (256, 384) | 192.00 KiB |
| `w_ba` | bf16 | (256, 8) | 4.00 KiB |
| `w_conv` | bf16 | (256, 4) | 2.00 KiB |
| `A_log` | bf16 | (4,) | 8 B |
| `dt_bias` | bf16 | (4,) | 8 B |
| `lin_norm_w` | bf16 | (32,) | 64 B |
| `w_out` | bf16 | (128, 256) | 64.00 KiB |
| `ffn_norm_w` | bf16 | (256,) | 512 B |
| `w_router` | bf16 | (256, 8) | 4.00 KiB |
| `w13_experts` | bf16 | (8, 256, 256) | 1.00 MiB |
| `w2_experts` | bf16 | (8, 128, 256) | 512.00 KiB |
| `w_shared_gate` | bf16 | (256, 1) | 512 B |
| `w_s13` | bf16 | (256, 256) | 128.00 KiB |
| `w_s2` | bf16 | (128, 256) | 64.00 KiB |

**`A_.._0` saved context** — 228.62 MiB = **3.57 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qkvz` | bf16 | (65536, 384) | 48.00 MiB |
| `ba` | bf16 | (65536, 8) | 1.00 MiB |
| `g_post` | fp32 | (65536, 4) | 1.00 MiB |
| `A_int` | bf16 | (65536, 4, 64) | 32.00 MiB |
| `core_out` | bf16 | (65536, 4, 32) | 16.00 MiB |
| `rstd_gate` | fp32 | (262144,) | 1.00 MiB |
| `xo` | bf16 | (65536, 256) | 32.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 8) | 1.00 MiB |
| `h13` | bf16 | (131072, 256) | 64.00 MiB |
| `gate_pre` | bf16 | (65536, 1) | 128.00 KiB |
| `s13` | bf16 | (65536, 256) | 32.00 MiB |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 2.19 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 B |
| `wq` | bf16 | (256, 512) | 256.00 KiB |
| `wk` | bf16 | (256, 128) | 64.00 KiB |
| `wv` | bf16 | (256, 128) | 64.00 KiB |
| `q_norm_w` | bf16 | (64,) | 128 B |
| `k_norm_w` | bf16 | (64,) | 128 B |
| `wo` | bf16 | (256, 256) | 128.00 KiB |
| `ffn_norm_w` | bf16 | (256,) | 512 B |
| `w_router` | bf16 | (256, 8) | 4.00 KiB |
| `w13_experts` | bf16 | (8, 256, 256) | 1.00 MiB |
| `w2_experts` | bf16 | (8, 128, 256) | 512.00 KiB |
| `w_shared_gate` | bf16 | (256, 1) | 512 B |
| `w_s13` | bf16 | (256, 256) | 128.00 KiB |
| `w_s2` | bf16 | (128, 256) | 64.00 KiB |

**`A_.._3` saved context** — 260.12 MiB = **4.06 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 256) | 32.00 MiB |
| `km` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_q` | fp32 | (262144,) | 1.00 MiB |
| `rstd_k` | fp32 | (131072,) | 512.00 KiB |
| `gate` | bf16 | (65536, 256) | 32.00 MiB |
| `v` | bf16 | (65536, 128) | 16.00 MiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 256) | 32.00 MiB |
| `xo` | bf16 | (65536, 256) | 32.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 8) | 1.00 MiB |
| `h13` | bf16 | (131072, 256) | 64.00 MiB |
| `gate_pre` | bf16 | (65536, 1) | 128.00 KiB |
| `s13` | bf16 | (65536, 256) | 32.00 MiB |

**`W_head`** — 256.50 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 256) | 256.00 KiB |
| `final_norm_w` | bf16 | (256,) | 512 B |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_0` (512 B), `Aux_1` (512 B), `Aux_2` (512 B), `Aux_3` (512 B)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_0`, `Aux_1`, `Aux_2`, `Aux_3`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (256.00 KiB)
- outputs: `y_embed_0_0` (32.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `linmoe_fwd` — `Qwen35MoeLinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (32.00 MiB), `W_0` (1.95 MiB), `current_round_0_0` (4 B), `Aux_0` (512 B)
- outputs: `y_0_0_0` (32.00 MiB), `A_0_0_0` (228.62 MiB), `AuxTemp_0_0_0` (1.25 MiB)
- mutates: `Aux_0`
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `proj` — qkvz, ba
    2. `conv` — —
    3. `heads_l2norm` — —
    4. `fla` — g_post, A_int, core_out
    5. `norm_out` — rstd_gate, xo
    6. `ffn_norm` — rstd_ffn
    7. `moe_route` — router_logits
    8. `moe_dispatch` — —
    9. `moe_experts13` — h13
    10. `moe_shared` — s13, gate_pre  ← derived recompute boundary
    11. `moe_experts2_combine` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `proj`:
        1. `mm ×2`
    - `conv`:
        2. `causal_conv1d_silu_fwd`
    - `heads_l2norm`:
        3. `fla::l2norm_fwd ×2`
    - `fla`:
        4. `fla::chunk_gated_delta_rule_fwd`
    - `norm_out`:
        5. `gated_rmsnorm_fwd`
        6. `addmm`
    - `ffn_norm`:
        7. `rmsnorm_fwd`
    - `moe_route`:
        8. `mm`
        9. `moe_topk_softmax`
        10. `scatter_add_ ×2`
    - `moe_dispatch`:
        11. `moe_sort`
        12. `moe_dispatch_fwd`
    - `moe_experts13`:
        13. `moe_grouped_mm_fwd`
    - `moe_shared`:
        14. `mm ×2`
    - `moe_experts2_combine`:
        15. `swiglu_packed_fwd`
        16. `moe_grouped_mm_fwd`
        17. `swiglu_packed_fwd`
        18. `mm`
        19. `moe_scale_rows`
        20. `moe_combine_fwd`

### `gattnmoe_fwd` — `Qwen35MoeAttnBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (32.00 MiB), `W_3` (2.19 MiB), `current_round_0_0` (4 B), `Aux_3` (512 B)
- outputs: `y_0_0_3` (32.00 MiB), `A_0_0_3` (260.12 MiB), `AuxTemp_0_0_3` (1.25 MiB)
- mutates: `Aux_3`
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_gate` — qm, km, gate, v
    2. `qknorm_rope` — rstd_q, rstd_k
    3. `attn` — lse, attn_out
    4. `gate_o` — xo
    5. `ffn_norm` — rstd_ffn
    6. `moe_route` — router_logits
    7. `moe_dispatch` — —
    8. `moe_experts13` — h13
    9. `moe_shared` — s13, gate_pre  ← derived recompute boundary
    10. `moe_experts2_combine` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `qkv_gate`:
        1. `mm ×3`
    - `qknorm_rope`:
        2. `rmsnorm_fwd ×2`
        3. `rope_fwd ×2`
    - `gate_o`:
        4. `addmm`
    - `ffn_norm`:
        5. `rmsnorm_fwd`
    - `moe_route`:
        6. `mm`
        7. `moe_topk_softmax`
        8. `scatter_add_ ×2`
    - `moe_dispatch`:
        9. `moe_sort`
        10. `moe_dispatch_fwd`
    - `moe_experts13`:
        11. `moe_grouped_mm_fwd`
    - `moe_shared`:
        12. `mm ×2`
    - `moe_experts2_combine`:
        13. `swiglu_packed_fwd`
        14. `moe_grouped_mm_fwd`
        15. `swiglu_packed_fwd`
        16. `mm`
        17. `moe_scale_rows`
        18. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_3` (32.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (256.50 KiB)
- outputs: `dy_0_0_3` (32.00 MiB), `loss_0_0` (4 B), `dW_head_0` (256.50 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (256.50 KiB), `dW_head_0` (256.50 KiB), `O_head` (513.00 KiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `gattnmoe_bwd` — `Qwen35MoeAttnBlockBwd`

- example task: `block_bwd_0_0_3`
- inputs: `dy_0_0_3` (32.00 MiB), `A_0_0_3` (260.12 MiB), `y_0_0_2` (32.00 MiB), `W_3` (2.19 MiB), `AuxTemp_0_0_3` (1.25 MiB), `Aux_3` (512 B)
- outputs: `dy_0_0_2` (32.00 MiB), `dW_0_3` (2.19 MiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `moe_dispatch_fwd ×2`
    2. `swiglu_packed_fwd`
    3. `moe_grouped_mm_dgrad`
    4. `moe_rowdot`
    5. `moe_scale_rows`
    6. `moe_grouped_mm_wgrad`
    7. `moe_scale_rows`
    8. `swiglu_packed_bwd`
    9. `moe_grouped_mm_wgrad`
    10. `moe_grouped_mm_dgrad`
    11. `moe_dispatch_bwd`
    12. `moe_router_bwd`
    13. `moe_aux_lb_grad`
    14. `mm`
    15. `swiglu_packed_fwd`
    16. `mm`
    17. `moe_rowdot`
    18. `moe_scale_rows`
    19. `mm ×3`
    20. `swiglu_packed_bwd`
    21. `mm`
    22. `rmsnorm_bwd`
    23. `mm ×2`
    24. `rmsnorm_apply ×2`
    25. `rope_fwd ×2`
    26. `_flash_attention_backward`
    27. `rope_bwd ×2`
    28. `rmsnorm_bwd ×2`
    29. `rmsnorm_apply`
    30. `mm ×4`
    31. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_3`
- inputs: `W_3` (2.19 MiB), `dW_0_3` (2.19 MiB), `O_3` (4.39 MiB)
- outputs: —
- mutates: `W_3`, `O_3`
- kernel calls:
    0. `adamw_step ×14`

### `linmoe_bwd` — `Qwen35MoeLinBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (32.00 MiB), `A_0_0_2` (228.62 MiB), `y_0_0_1` (32.00 MiB), `W_2` (1.95 MiB), `AuxTemp_0_0_2` (1.25 MiB), `Aux_2` (512 B)
- outputs: `dy_0_0_1` (32.00 MiB), `dW_0_2` (1.95 MiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `moe_dispatch_fwd ×2`
    2. `swiglu_packed_fwd`
    3. `moe_grouped_mm_dgrad`
    4. `moe_rowdot`
    5. `moe_scale_rows`
    6. `moe_grouped_mm_wgrad`
    7. `moe_scale_rows`
    8. `swiglu_packed_bwd`
    9. `moe_grouped_mm_wgrad`
    10. `moe_grouped_mm_dgrad`
    11. `moe_dispatch_bwd`
    12. `moe_router_bwd`
    13. `moe_aux_lb_grad`
    14. `mm`
    15. `swiglu_packed_fwd`
    16. `mm`
    17. `moe_rowdot`
    18. `moe_scale_rows`
    19. `mm ×3`
    20. `swiglu_packed_bwd`
    21. `mm`
    22. `rmsnorm_bwd`
    23. `mm`
    24. `gated_rmsnorm_bwd`
    25. `mm`
    26. `causal_conv1d_silu_fwd`
    27. `fla::l2norm_fwd ×2`
    28. `fla::chunk_gated_delta_rule_bwd`
    29. `fla::l2norm_bwd ×2`
    30. `causal_conv1d_silu_bwd`
    31. `rmsnorm_apply`
    32. `mm ×3`
    33. `rmsnorm_bwd`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (32.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (256.00 KiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (256.00 KiB), `dW_embed_0` (256.00 KiB), `O_embed` (512.00 KiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

