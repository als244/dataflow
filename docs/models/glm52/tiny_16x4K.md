# glm52 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedGlm52Config.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (6 layers): `gdl gml gmf gmf gml gmf`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (gdl)` | layer | 299.50 KiB |
| `dW_i (gdl)` | layer/step | 299.50 KiB |
| `O_i (gdl)` | layer | 599.00 KiB |
| `A (gdl)` | layer × round | 103.00 MiB (1.61 KiB/token) |
| `M (gdl)` | layer × round | 6.00 MiB (96.0 B/token) |
| `W_i (gml)` | layer | 325.75 KiB |
| `dW_i (gml)` | layer/step | 325.75 KiB |
| `O_i (gml)` | layer | 651.00 KiB |
| `A (gml)` | layer × round | 64.00 MiB (1.00 KiB/token) |
| `M (gml)` | layer × round | 7.25 MiB (116.0 B/token) |
| `W_i (gmf)` | layer | 281.25 KiB |
| `dW_i (gmf)` | layer/step | 281.25 KiB |
| `O_i (gmf)` | layer | 562.00 KiB |
| `A (gmf)` | layer × round | 64.00 MiB (1.00 KiB/token) |
| `M (gmf)` | layer × round | 1.25 MiB (20.0 B/token) |
| `W_head` | run | 128.25 KiB |
| `W_embed` | run | 128.00 KiB |
| `O_embed` | run | 256.00 KiB |
| `O_head` | run | 256.50 KiB |
| `hidden state (y)` | boundary buffer | 16.00 MiB (256.0 B/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 8 | 2.00 MiB |
| dW (all gradients, per step) | 10 | 14.00 MiB |
| O (all optimizer state) | 8 | 4.00 MiB |
| A (all saved activations, one round) | 6 | 423.00 MiB (6.61 KiB/token) |
| M (all metadata, one round) | 6 | 24.25 MiB (388.0 B/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `q_lora_rank` | 64 |
| `kv_lora_rank` | 32 |
| `qk_nope_dim` | 16 |
| `qk_rope_dim` | 8 |
| `v_head_dim` | 16 |
| `d_ff` | 256 |
| `first_k_dense` | 1 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 8000000.0 |
| `kinds` | ('gdl', 'gml', 'gmf', 'gmf', 'gml', 'gmf') |
| `index_n_heads` | 8 |
| `index_head_dim` | 32 |
| `index_topk` | 24 |
| `sparse_mode` | True |
| `train_indexer` | True |
| `indexer_types` | ('full', 'full', 'shared', 'shared', 'full', 'shared') |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `gdl` (e.g. layer 0)

**`W_0` weights** — 299.50 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 B |
| `w_q_a` | bf16 | (128, 64) | 16.00 KiB |
| `q_a_norm_w` | bf16 | (64,) | 128 B |
| `w_q_b` | bf16 | (64, 96) | 12.00 KiB |
| `w_kv_a` | bf16 | (128, 40) | 10.00 KiB |
| `kv_a_norm_w` | bf16 | (32,) | 64 B |
| `w_kv_b` | bf16 | (32, 128) | 8.00 KiB |
| `wo` | bf16 | (64, 128) | 16.00 KiB |
| `w_idx_q` | bf16 | (64, 256) | 32.00 KiB |
| `w_idx_k` | bf16 | (128, 32) | 8.00 KiB |
| `idx_k_ln_w` | bf16 | (32,) | 64 B |
| `idx_k_ln_b` | bf16 | (32,) | 64 B |
| `w_idx_w` | fp32 | (128, 8) | 4.00 KiB |
| `ffn_norm_w` | bf16 | (128,) | 256 B |
| `w1` | bf16 | (128, 256) | 64.00 KiB |
| `w3` | bf16 | (128, 256) | 64.00 KiB |
| `w2` | bf16 | (256, 128) | 64.00 KiB |

**`A_.._0` saved context** — 103.00 MiB = **1.61 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 64) | 8.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 40) | 5.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 64) | 8.00 MiB |
| `h_mid` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 256) | 32.00 MiB |
| `x3` | bf16 | (65536, 256) | 32.00 MiB |

**`M_.._0` metadata** — 6.00 MiB = **96.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 24) | 6.00 MiB |

### kind `gml` (e.g. layer 1)

**`W_1` weights** — 325.75 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 B |
| `w_q_a` | bf16 | (128, 64) | 16.00 KiB |
| `q_a_norm_w` | bf16 | (64,) | 128 B |
| `w_q_b` | bf16 | (64, 96) | 12.00 KiB |
| `w_kv_a` | bf16 | (128, 40) | 10.00 KiB |
| `kv_a_norm_w` | bf16 | (32,) | 64 B |
| `w_kv_b` | bf16 | (32, 128) | 8.00 KiB |
| `wo` | bf16 | (64, 128) | 16.00 KiB |
| `w_idx_q` | bf16 | (64, 256) | 32.00 KiB |
| `w_idx_k` | bf16 | (128, 32) | 8.00 KiB |
| `idx_k_ln_w` | bf16 | (32,) | 64 B |
| `idx_k_ln_b` | bf16 | (32,) | 64 B |
| `w_idx_w` | fp32 | (128, 8) | 4.00 KiB |
| `ffn_norm_w` | bf16 | (128,) | 256 B |
| `w_router` | bf16 | (128, 8) | 2.00 KiB |
| `w_router_bias` | fp32 | (8,) | 32 B |
| `w13_experts` | bf16 | (8, 128, 64) | 128.00 KiB |
| `w2_experts` | bf16 | (8, 32, 128) | 64.00 KiB |
| `w_s13` | bf16 | (128, 64) | 16.00 KiB |
| `w_s2` | bf16 | (32, 128) | 8.00 KiB |

**`A_.._1` saved context** — 64.00 MiB = **1.00 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 64) | 8.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 40) | 5.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 64) | 8.00 MiB |
| `h_mid` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 8) | 1.00 MiB |
| `h13` | bf16 | (131072, 64) | 16.00 MiB |
| `s13` | bf16 | (65536, 64) | 8.00 MiB |

**`M_.._1` metadata** — 7.25 MiB = **116.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 24) | 6.00 MiB |
| `route_w` | bf16 | (65536, 2) | 256.00 KiB |
| `route_ids` | int32 | (65536, 2) | 512.00 KiB |
| `route_order` | int32 | (131072,) | 512.00 KiB |
| `route_offsets` | int32 | (9,) | 36 B |

### kind `gmf` (e.g. layer 2)

**`W_2` weights** — 281.25 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 B |
| `w_q_a` | bf16 | (128, 64) | 16.00 KiB |
| `q_a_norm_w` | bf16 | (64,) | 128 B |
| `w_q_b` | bf16 | (64, 96) | 12.00 KiB |
| `w_kv_a` | bf16 | (128, 40) | 10.00 KiB |
| `kv_a_norm_w` | bf16 | (32,) | 64 B |
| `w_kv_b` | bf16 | (32, 128) | 8.00 KiB |
| `wo` | bf16 | (64, 128) | 16.00 KiB |
| `ffn_norm_w` | bf16 | (128,) | 256 B |
| `w_router` | bf16 | (128, 8) | 2.00 KiB |
| `w_router_bias` | fp32 | (8,) | 32 B |
| `w13_experts` | bf16 | (8, 128, 64) | 128.00 KiB |
| `w2_experts` | bf16 | (8, 32, 128) | 64.00 KiB |
| `w_s13` | bf16 | (128, 64) | 16.00 KiB |
| `w_s2` | bf16 | (32, 128) | 8.00 KiB |

**`A_.._2` saved context** — 64.00 MiB = **1.00 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 64) | 8.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 40) | 5.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 64) | 8.00 MiB |
| `h_mid` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 8) | 1.00 MiB |
| `h13` | bf16 | (131072, 64) | 16.00 MiB |
| `s13` | bf16 | (65536, 64) | 8.00 MiB |

**`M_.._2` metadata** — 1.25 MiB = **20.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 2) | 256.00 KiB |
| `route_ids` | int32 | (65536, 2) | 512.00 KiB |
| `route_order` | int32 | (131072,) | 512.00 KiB |
| `route_offsets` | int32 | (9,) | 36 B |

**`W_head`** — 128.25 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 128) | 128.00 KiB |
| `final_norm_w` | bf16 | (128,) | 256 B |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_1` (512 B), `Aux_2` (512 B), `Aux_3` (512 B), `Aux_4` (512 B), `Aux_5` (512 B)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_1`, `Aux_2`, `Aux_3`, `Aux_4`, `Aux_5`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (128.00 KiB)
- outputs: `y_embed_0_0` (16.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `gdl_fwd` — `Glm52DlBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (16.00 MiB), `W_0` (299.50 KiB)
- outputs: `y_0_0_0` (16.00 MiB), `A_0_0_0` (103.00 MiB), `AuxTemp_0_0_0` (6.00 MiB)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `mla_q` — q_a, rstd_qa
    2. `mla_kv` — kv_a, rstd_kva
    3. `dsa_select` — — [meta: never recomputed]
    4. `dsa_attn` — lse, attn_out
    5. `resid1_norm2` — h_mid, rstd_ffn
    6. `up_proj` — x1, x3  ← derived recompute boundary
    7. `swiglu` — —
    8. `down_resid` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
        5. `rmsnorm_apply`
    - `mla_kv`:
        6. `mm`
        7. `rope_fwd`
        8. `mm`
        9. `rope_fwd`
        10. `mm ×2`
        11. `rmsnorm_fwd`
        12. `rope_fwd`
        13. `mm`
    - `dsa_select`:
        14. `dsa_index_scores`
        15. `dsa_topk`
    - `dsa_attn`:
        16. `dsa_sparse_attn_fwd`
    - `resid1_norm2`:
        17. `addmm`
        18. `rmsnorm_fwd`
    - `up_proj`:
        19. `mm ×2`
    - `swiglu`:
        20. `swiglu_fwd_out`
    - `down_resid`:
        21. `addmm`

### `gml_fwd` — `Glm52MlBlockFwd`

- example task: `block_fwd_0_0_1`
- inputs: `y_0_0_0` (16.00 MiB), `W_1` (325.75 KiB), `current_round_0_0` (4 B), `Aux_1` (512 B)
- outputs: `y_0_0_1` (16.00 MiB), `A_0_0_1` (64.00 MiB), `AuxTemp_0_0_1` (7.25 MiB)
- mutates: `Aux_1`
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `mla_q` — q_a, rstd_qa
    2. `mla_kv` — kv_a, rstd_kva
    3. `dsa_select` — — [meta: never recomputed]
    4. `dsa_attn` — lse, attn_out
    5. `resid1_norm2` — h_mid, rstd_ffn
    6. `moe_route` — router_logits
    7. `moe_dispatch` — —
    8. `moe_experts13` — h13
    9. `moe_shared` — s13  ← derived recompute boundary
    10. `moe_experts2_combine` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
        5. `rmsnorm_apply`
    - `mla_kv`:
        6. `mm`
        7. `rope_fwd`
        8. `mm`
        9. `rope_fwd`
        10. `mm ×2`
        11. `rmsnorm_fwd`
        12. `rope_fwd`
        13. `mm`
    - `dsa_select`:
        14. `dsa_index_scores`
        15. `dsa_topk`
    - `dsa_attn`:
        16. `dsa_sparse_attn_fwd`
    - `resid1_norm2`:
        17. `addmm`
        18. `rmsnorm_fwd`
    - `moe_route`:
        19. `mm`
        20. `moe_topk_sigmoid_noaux`
        21. `scatter_add_ ×2`
    - `moe_dispatch`:
        22. `moe_sort`
        23. `moe_dispatch_fwd`
    - `moe_experts13`:
        24. `moe_grouped_mm_fwd`
    - `moe_shared`:
        25. `mm`
    - `moe_experts2_combine`:
        26. `swiglu_packed_fwd`
        27. `moe_grouped_mm_fwd`
        28. `swiglu_packed_fwd`
        29. `mm`
        30. `moe_combine_fwd`

### `gmf_fwd` — `Glm52MfBlockFwd`

- example task: `block_fwd_0_0_2`
- inputs: `y_0_0_1` (16.00 MiB), `W_2` (281.25 KiB), `AuxTemp_0_0_1` (7.25 MiB), `current_round_0_0` (4 B), `Aux_2` (512 B)
- outputs: `y_0_0_2` (16.00 MiB), `A_0_0_2` (64.00 MiB), `AuxTemp_0_0_2` (1.25 MiB)
- mutates: `Aux_2`
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `mla_q` — q_a, rstd_qa
    2. `mla_kv` — kv_a, rstd_kva
    3. `dsa_attn` — lse, attn_out
    4. `resid1_norm2` — h_mid, rstd_ffn
    5. `moe_route` — router_logits
    6. `moe_dispatch` — —
    7. `moe_experts13` — h13
    8. `moe_shared` — s13  ← derived recompute boundary
    9. `moe_experts2_combine` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
    - `mla_kv`:
        5. `mm`
        6. `rmsnorm_fwd`
        7. `rope_fwd`
        8. `mm`
    - `dsa_attn`:
        9. `dsa_sparse_attn_fwd`
    - `resid1_norm2`:
        10. `addmm`
        11. `rmsnorm_fwd`
    - `moe_route`:
        12. `mm`
        13. `moe_topk_sigmoid_noaux`
        14. `scatter_add_ ×2`
    - `moe_dispatch`:
        15. `moe_sort`
        16. `moe_dispatch_fwd`
    - `moe_experts13`:
        17. `moe_grouped_mm_fwd`
    - `moe_shared`:
        18. `mm`
    - `moe_experts2_combine`:
        19. `swiglu_packed_fwd`
        20. `moe_grouped_mm_fwd`
        21. `swiglu_packed_fwd`
        22. `mm`
        23. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_5` (16.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (128.25 KiB)
- outputs: `dy_0_0_5` (16.00 MiB), `loss_0_0` (4 B), `dW_head_0` (128.25 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (128.25 KiB), `dW_head_0` (128.25 KiB), `O_head` (256.50 KiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `gmf_bwd` — `Glm52MfBlockBwd`

- example task: `block_bwd_0_0_5`
- inputs: `dy_0_0_5` (16.00 MiB), `A_0_0_5` (64.00 MiB), `y_0_0_4` (16.00 MiB), `W_5` (281.25 KiB), `AuxTemp_0_0_5` (1.25 MiB), `AuxTemp_0_0_4` (7.25 MiB), `Aux_5` (512 B)
- outputs: `dy_0_0_4` (16.00 MiB), `dW_0_5` (281.00 KiB), `dAuxTemp_0_0_4` (6.00 MiB)
- mutates: `W_5`
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
    12. `moe_router_bwd_sigmoid`
    13. `moe_seq_aux_grad`
    14. `mm`
    15. `swiglu_packed_fwd`
    16. `mm ×2`
    17. `swiglu_packed_bwd`
    18. `mm`
    19. `rmsnorm_bwd`
    20. `mm ×2`
    21. `rmsnorm_apply`
    22. `mm`
    23. `rope_fwd`
    24. `rmsnorm_apply`
    25. `rope_fwd`
    26. `mm`
    27. `sort`
    28. `scatter_add_`
    29. `dsa_sparse_attn_bwd`
    30. `rmsnorm_apply`
    31. `dsa_probs_sum`
    32. `rope_bwd`
    33. `mm ×2`
    34. `rmsnorm_bwd`
    35. `rope_bwd`
    36. `mm ×2`
    37. `rmsnorm_bwd`
    38. `mm ×3`
    39. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_5`
- inputs: `W_5` (281.25 KiB), `dW_0_5` (281.00 KiB), `O_5` (562.00 KiB)
- outputs: —
- mutates: `W_5`, `O_5`
- kernel calls:
    0. `adamw_step ×14`

### `gml_bwd` — `Glm52MlBlockBwd`

- example task: `block_bwd_0_0_4`
- inputs: `dy_0_0_4` (16.00 MiB), `A_0_0_4` (64.00 MiB), `y_0_0_3` (16.00 MiB), `W_4` (325.75 KiB), `AuxTemp_0_0_4` (7.25 MiB), `dAuxTemp_0_0_4` (6.00 MiB), `Aux_4` (512 B)
- outputs: `dy_0_0_3` (16.00 MiB), `dW_0_4` (325.50 KiB)
- mutates: `W_4`
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
    12. `moe_router_bwd_sigmoid`
    13. `moe_seq_aux_grad`
    14. `mm`
    15. `swiglu_packed_fwd`
    16. `mm ×2`
    17. `swiglu_packed_bwd`
    18. `mm`
    19. `rmsnorm_bwd`
    20. `mm ×2`
    21. `rmsnorm_apply`
    22. `mm`
    23. `rope_fwd`
    24. `rmsnorm_apply`
    25. `rope_fwd`
    26. `mm`
    27. `sort`
    28. `scatter_add_`
    29. `dsa_sparse_attn_bwd`
    30. `rmsnorm_apply`
    31. `mm`
    32. `rope_fwd`
    33. `mm`
    34. `rope_fwd`
    35. `mm`
    36. `dsa_index_scores`
    37. `dsa_probs_sum`
    38. `scatter_add_`
    39. `logsumexp`
    40. `dsa_index_bwd`
    41. `rope_bwd`
    42. `mm`
    43. `rope_bwd`
    44. `mm ×3`
    45. `rope_bwd`
    46. `mm ×2`
    47. `rmsnorm_bwd`
    48. `rope_bwd`
    49. `mm ×2`
    50. `rmsnorm_bwd`
    51. `mm ×3`
    52. `rmsnorm_bwd`

### `gdl_bwd` — `Glm52DlBlockBwd`

- example task: `block_bwd_0_0_0`
- inputs: `dy_0_0_0` (16.00 MiB), `A_0_0_0` (103.00 MiB), `y_embed_0_0` (16.00 MiB), `W_0` (299.50 KiB), `AuxTemp_0_0_0` (6.00 MiB)
- outputs: `dy_embed_0_0` (16.00 MiB), `dW_0_0` (299.50 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `swiglu_fwd_out`
    2. `mm ×2`
    3. `swiglu_bwd`
    4. `mm ×3`
    5. `rmsnorm_bwd`
    6. `mm ×2`
    7. `rmsnorm_apply`
    8. `mm`
    9. `rope_fwd`
    10. `rmsnorm_apply`
    11. `rope_fwd`
    12. `mm`
    13. `sort`
    14. `scatter_add_`
    15. `dsa_sparse_attn_bwd`
    16. `rmsnorm_apply`
    17. `mm`
    18. `rope_fwd`
    19. `mm`
    20. `rope_fwd`
    21. `mm`
    22. `dsa_index_scores`
    23. `_softmax`
    24. `dsa_probs_sum`
    25. `dsa_index_bwd`
    26. `rope_bwd`
    27. `mm`
    28. `rope_bwd`
    29. `mm ×3`
    30. `rope_bwd`
    31. `mm ×2`
    32. `rmsnorm_bwd`
    33. `rope_bwd`
    34. `mm ×2`
    35. `rmsnorm_bwd`
    36. `mm ×3`
    37. `rmsnorm_bwd`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (16.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (128.00 KiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (128.00 KiB), `dW_embed_0` (128.00 KiB), `O_embed` (256.00 KiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

