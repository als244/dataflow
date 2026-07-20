# qwen35moe / `qwen35moe_35b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen35MoeConfig.qwen35moe_35b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset qwen35moe_35b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (40 layers): `lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (lin)` | layer | 1.57 GiB |
| `dW_i (lin)` | layer/step | 1.57 GiB |
| `O_i (lin)` | layer | 3.14 GiB |
| `A (lin)` | layer × round | 3.68 GiB (58.88 KiB/token) |
| `W_i (full)` | layer | 1.56 GiB |
| `dW_i (full)` | layer/step | 1.56 GiB |
| `O_i (full)` | layer | 3.12 GiB |
| `A (full)` | layer × round | 3.04 GiB (48.64 KiB/token) |
| `W_head` | run | 970.00 MiB |
| `W_embed` | run | 970.00 MiB |
| `O_embed` | run | 1.89 GiB |
| `O_head` | run | 1.89 GiB |
| `hidden state (y)` | boundary buffer | 256.00 MiB (4.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 42 | 64.56 GiB |
| dW (all gradients, per step) | 42 | 64.56 GiB |
| O (all optimizer state) | 42 | 129.12 GiB |
| A (all saved activations, one round) | 40 | 140.81 GiB (2.20 MiB/token) |
| M (all metadata, one round) | 40 | 200.05 MiB (3.13 KiB/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 2048 |
| `n_layers` | 40 |
| `full_attention_interval` | 4 |
| `n_heads` | 16 |
| `n_kv_heads` | 2 |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 |
| `lin_k_heads` | 16 |
| `lin_v_heads` | 32 |
| `lin_k_head_dim` | 128 |
| `lin_v_head_dim` | 128 |
| `lin_conv_kernel` | 4 |
| `d_ff` | 512 |
| `vocab_size` | 248320 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `lin` (e.g. layer 0)

**`W_0` weights** — 1.57 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_qkvz` | bf16 | (2048, 12288) | 48.00 MiB |
| `w_ba` | bf16 | (2048, 64) | 256.00 KiB |
| `w_conv` | bf16 | (8192, 4) | 64.00 KiB |
| `A_log` | bf16 | (32,) | 64 B |
| `dt_bias` | bf16 | (32,) | 64 B |
| `lin_norm_w` | bf16 | (128,) | 256 B |
| `w_out` | bf16 | (4096, 2048) | 16.00 MiB |
| `ffn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_router` | bf16 | (2048, 256) | 1.00 MiB |
| `w13_experts` | bf16 | (256, 2048, 1024) | 1.00 GiB |
| `w2_experts` | bf16 | (256, 512, 2048) | 512.00 MiB |
| `w_shared_gate` | bf16 | (2048, 1) | 4.00 KiB |
| `w_s13` | bf16 | (2048, 1024) | 4.00 MiB |
| `w_s2` | bf16 | (512, 2048) | 2.00 MiB |

**`A_.._0` saved context** — 3.68 GiB = **58.88 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qkvz` | bf16 | (65536, 12288) | 1.50 GiB |
| `ba` | bf16 | (65536, 64) | 8.00 MiB |
| `g_post` | fp32 | (65536, 32) | 8.00 MiB |
| `A_int` | bf16 | (65536, 32, 64) | 256.00 MiB |
| `core_out` | bf16 | (65536, 32, 128) | 512.00 MiB |
| `rstd_gate` | fp32 | (2097152,) | 8.00 MiB |
| `xo` | bf16 | (65536, 2048) | 256.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 256) | 32.00 MiB |
| `h13` | bf16 | (524288, 1024) | 1.00 GiB |
| `gate_pre` | bf16 | (65536, 1) | 128.00 KiB |
| `s13` | bf16 | (65536, 1024) | 128.00 MiB |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 1.56 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `wq` | bf16 | (2048, 8192) | 32.00 MiB |
| `wk` | bf16 | (2048, 512) | 2.00 MiB |
| `wv` | bf16 | (2048, 512) | 2.00 MiB |
| `q_norm_w` | bf16 | (256,) | 512 B |
| `k_norm_w` | bf16 | (256,) | 512 B |
| `wo` | bf16 | (4096, 2048) | 16.00 MiB |
| `ffn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_router` | bf16 | (2048, 256) | 1.00 MiB |
| `w13_experts` | bf16 | (256, 2048, 1024) | 1.00 GiB |
| `w2_experts` | bf16 | (256, 512, 2048) | 512.00 MiB |
| `w_shared_gate` | bf16 | (2048, 1) | 4.00 KiB |
| `w_s13` | bf16 | (2048, 1024) | 4.00 MiB |
| `w_s2` | bf16 | (512, 2048) | 2.00 MiB |

**`A_.._3` saved context** — 3.04 GiB = **48.64 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 4096) | 512.00 MiB |
| `km` | bf16 | (65536, 512) | 64.00 MiB |
| `rstd_q` | fp32 | (1048576,) | 4.00 MiB |
| `rstd_k` | fp32 | (131072,) | 512.00 KiB |
| `gate` | bf16 | (65536, 4096) | 512.00 MiB |
| `v` | bf16 | (65536, 512) | 64.00 MiB |
| `lse` | fp32 | (16, 65536) | 4.00 MiB |
| `attn_out` | bf16 | (65536, 4096) | 512.00 MiB |
| `xo` | bf16 | (65536, 2048) | 256.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 256) | 32.00 MiB |
| `h13` | bf16 | (524288, 1024) | 1.00 GiB |
| `gate_pre` | bf16 | (65536, 1) | 128.00 KiB |
| `s13` | bf16 | (65536, 1024) | 128.00 MiB |

**`W_head`** — 970.00 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (248320, 2048) | 970.00 MiB |
| `final_norm_w` | bf16 | (2048,) | 4.00 KiB |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_0` (3.00 KiB), `Aux_1` (3.00 KiB), `Aux_2` (3.00 KiB), `Aux_3` (3.00 KiB), `Aux_4` (3.00 KiB), `Aux_5` (3.00 KiB), `Aux_6` (3.00 KiB), `Aux_7` (3.00 KiB), `Aux_8` (3.00 KiB), `Aux_9` (3.00 KiB), `Aux_10` (3.00 KiB), `Aux_11` (3.00 KiB), `Aux_12` (3.00 KiB), `Aux_13` (3.00 KiB), `Aux_14` (3.00 KiB), `Aux_15` (3.00 KiB), `Aux_16` (3.00 KiB), `Aux_17` (3.00 KiB), `Aux_18` (3.00 KiB), `Aux_19` (3.00 KiB), `Aux_20` (3.00 KiB), `Aux_21` (3.00 KiB), `Aux_22` (3.00 KiB), `Aux_23` (3.00 KiB), `Aux_24` (3.00 KiB), `Aux_25` (3.00 KiB), `Aux_26` (3.00 KiB), `Aux_27` (3.00 KiB), `Aux_28` (3.00 KiB), `Aux_29` (3.00 KiB), `Aux_30` (3.00 KiB), `Aux_31` (3.00 KiB), `Aux_32` (3.00 KiB), `Aux_33` (3.00 KiB), `Aux_34` (3.00 KiB), `Aux_35` (3.00 KiB), `Aux_36` (3.00 KiB), `Aux_37` (3.00 KiB), `Aux_38` (3.00 KiB), `Aux_39` (3.00 KiB)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_0`, `Aux_1`, `Aux_2`, `Aux_3`, `Aux_4`, `Aux_5`, `Aux_6`, `Aux_7`, `Aux_8`, `Aux_9`, `Aux_10`, `Aux_11`, `Aux_12`, `Aux_13`, `Aux_14`, `Aux_15`, `Aux_16`, `Aux_17`, `Aux_18`, `Aux_19`, `Aux_20`, `Aux_21`, `Aux_22`, `Aux_23`, `Aux_24`, `Aux_25`, `Aux_26`, `Aux_27`, `Aux_28`, `Aux_29`, `Aux_30`, `Aux_31`, `Aux_32`, `Aux_33`, `Aux_34`, `Aux_35`, `Aux_36`, `Aux_37`, `Aux_38`, `Aux_39`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (970.00 MiB)
- outputs: `y_embed_0_0` (256.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `linmoe_fwd` — `Qwen35MoeLinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (256.00 MiB), `W_0` (1.57 GiB), `current_round_0_0` (4 B), `Aux_0` (3.00 KiB)
- outputs: `y_0_0_0` (256.00 MiB), `A_0_0_0` (3.68 GiB), `AuxTemp_0_0_0` (5.00 MiB)
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
- inputs: `y_0_0_2` (256.00 MiB), `W_3` (1.56 GiB), `current_round_0_0` (4 B), `Aux_3` (3.00 KiB)
- outputs: `y_0_0_3` (256.00 MiB), `A_0_0_3` (3.04 GiB), `AuxTemp_0_0_3` (5.00 MiB)
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
- inputs: `y_0_0_39` (256.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (970.00 MiB)
- outputs: `dy_0_0_39` (256.00 MiB), `loss_0_0` (4 B), `dW_head_0` (970.00 MiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (970.00 MiB), `dW_head_0` (970.00 MiB), `O_head` (1.89 GiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `gattnmoe_bwd` — `Qwen35MoeAttnBlockBwd`

- example task: `block_bwd_0_0_39`
- inputs: `dy_0_0_39` (256.00 MiB), `A_0_0_39` (3.04 GiB), `y_0_0_38` (256.00 MiB), `W_39` (1.56 GiB), `AuxTemp_0_0_39` (5.00 MiB), `Aux_39` (3.00 KiB)
- outputs: `dy_0_0_38` (256.00 MiB), `dW_0_39` (1.56 GiB)
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

- example task: `optimizer_0_39`
- inputs: `W_39` (1.56 GiB), `dW_0_39` (1.56 GiB), `O_39` (3.12 GiB)
- outputs: —
- mutates: `W_39`, `O_39`
- kernel calls:
    0. `adamw_step ×14`

### `linmoe_bwd` — `Qwen35MoeLinBlockBwd`

- example task: `block_bwd_0_0_38`
- inputs: `dy_0_0_38` (256.00 MiB), `A_0_0_38` (3.68 GiB), `y_0_0_37` (256.00 MiB), `W_38` (1.57 GiB), `AuxTemp_0_0_38` (5.00 MiB), `Aux_38` (3.00 KiB)
- outputs: `dy_0_0_37` (256.00 MiB), `dW_0_38` (1.57 GiB)
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
- inputs: `dy_embed_0_0` (256.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (970.00 MiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (970.00 MiB), `dW_embed_0` (970.00 MiB), `O_embed` (1.89 GiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

