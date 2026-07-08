# glm52 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedGlm52Config.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (6 layers): `gdl gml gmf gmf gml gmf`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (gdl)` | layer | 306,688 |
| `dW_i (gdl)` | layer/step | 306,688 |
| `O_i (gdl)` | layer | 613,376 |
| `A (gdl)` | layer × round | 108,003,328 (1,648.0/token) |
| `M (gdl)` | layer × round | 6,291,456 (96.0/token) |
| `W_i (gml)` | layer | 333,568 |
| `dW_i (gml)` | layer/step | 333,568 |
| `O_i (gml)` | layer | 667,136 |
| `A (gml)` | layer × round | 67,108,864 (1,024.0/token) |
| `M (gml)` | layer × round | 7,602,432 (116.0/token) |
| `W_i (gmf)` | layer | 288,000 |
| `dW_i (gmf)` | layer/step | 288,000 |
| `O_i (gmf)` | layer | 576,000 |
| `A (gmf)` | layer × round | 67,108,864 (1,024.0/token) |
| `M (gmf)` | layer × round | 1,310,976 (20.0/token) |
| `W_head` | run | 131,328 |
| `W_embed` | run | 131,072 |
| `O_embed` | run | 262,144 |
| `O_head` | run | 262,656 |
| `hidden state (y)` | boundary buffer | 16,777,216 (256.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 8 | 2,100,224 |
| dW (all gradients, per step) | 10 | 14,683,136 |
| O (all optimizer state) | 8 | 4,200,448 |
| A (all saved activations, one round) | 6 | 443,547,648 (6,768.0/token) |
| M (all metadata, one round) | 6 | 25,429,248 (388.0/token) |

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
| `opt_policy` | adamw |
| `index_n_heads` | 8 |
| `index_head_dim` | 32 |
| `index_topk` | 24 |
| `sparse_mode` | True |
| `train_indexer` | True |
| `indexer_types` | ('full', 'full', 'shared', 'shared', 'full', 'shared') |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `gdl` (e.g. layer 0)

**`W_0` weights** — 306,688 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 |
| `w_q_a` | bf16 | (128, 64) | 16,384 |
| `q_a_norm_w` | bf16 | (64,) | 128 |
| `w_q_b` | bf16 | (64, 96) | 12,288 |
| `w_kv_a` | bf16 | (128, 40) | 10,240 |
| `kv_a_norm_w` | bf16 | (32,) | 64 |
| `w_kv_b` | bf16 | (32, 128) | 8,192 |
| `wo` | bf16 | (64, 128) | 16,384 |
| `w_idx_q` | bf16 | (64, 256) | 32,768 |
| `w_idx_k` | bf16 | (128, 32) | 8,192 |
| `idx_k_ln_w` | bf16 | (32,) | 64 |
| `idx_k_ln_b` | bf16 | (32,) | 64 |
| `w_idx_w` | fp32 | (128, 8) | 4,096 |
| `ffn_norm_w` | bf16 | (128,) | 256 |
| `w1` | bf16 | (128, 256) | 65,536 |
| `w3` | bf16 | (128, 256) | 65,536 |
| `w2` | bf16 | (256, 128) | 65,536 |

**`A_.._0` saved context** — 108,003,328 bytes = **1,648.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 64) | 8,388,608 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 40) | 5,242,880 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 64) | 8,388,608 |
| `h_mid` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 256) | 33,554,432 |
| `x3` | bf16 | (65536, 256) | 33,554,432 |

**`M_.._0` metadata** — 6,291,456 bytes = **96.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 24) | 6,291,456 |

### kind `gml` (e.g. layer 1)

**`W_1` weights** — 333,568 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 |
| `w_q_a` | bf16 | (128, 64) | 16,384 |
| `q_a_norm_w` | bf16 | (64,) | 128 |
| `w_q_b` | bf16 | (64, 96) | 12,288 |
| `w_kv_a` | bf16 | (128, 40) | 10,240 |
| `kv_a_norm_w` | bf16 | (32,) | 64 |
| `w_kv_b` | bf16 | (32, 128) | 8,192 |
| `wo` | bf16 | (64, 128) | 16,384 |
| `w_idx_q` | bf16 | (64, 256) | 32,768 |
| `w_idx_k` | bf16 | (128, 32) | 8,192 |
| `idx_k_ln_w` | bf16 | (32,) | 64 |
| `idx_k_ln_b` | bf16 | (32,) | 64 |
| `w_idx_w` | fp32 | (128, 8) | 4,096 |
| `ffn_norm_w` | bf16 | (128,) | 256 |
| `w_router` | bf16 | (128, 8) | 2,048 |
| `w_router_bias` | fp32 | (8,) | 32 |
| `w13_experts` | bf16 | (8, 128, 64) | 131,072 |
| `w2_experts` | bf16 | (8, 32, 128) | 65,536 |
| `w_s13` | bf16 | (128, 64) | 16,384 |
| `w_s2` | bf16 | (32, 128) | 8,192 |

**`A_.._1` saved context** — 67,108,864 bytes = **1,024.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 64) | 8,388,608 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 40) | 5,242,880 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 64) | 8,388,608 |
| `h_mid` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 8) | 1,048,576 |
| `h13` | bf16 | (131072, 64) | 16,777,216 |
| `s13` | bf16 | (65536, 64) | 8,388,608 |

**`M_.._1` metadata** — 7,602,432 bytes = **116.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 24) | 6,291,456 |
| `route_w` | bf16 | (65536, 2) | 262,144 |
| `route_ids` | int32 | (65536, 2) | 524,288 |
| `route_order` | int32 | (131072,) | 524,288 |
| `route_offsets` | int32 | (9,) | 36 |

### kind `gmf` (e.g. layer 2)

**`W_2` weights** — 288,000 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 |
| `w_q_a` | bf16 | (128, 64) | 16,384 |
| `q_a_norm_w` | bf16 | (64,) | 128 |
| `w_q_b` | bf16 | (64, 96) | 12,288 |
| `w_kv_a` | bf16 | (128, 40) | 10,240 |
| `kv_a_norm_w` | bf16 | (32,) | 64 |
| `w_kv_b` | bf16 | (32, 128) | 8,192 |
| `wo` | bf16 | (64, 128) | 16,384 |
| `ffn_norm_w` | bf16 | (128,) | 256 |
| `w_router` | bf16 | (128, 8) | 2,048 |
| `w_router_bias` | fp32 | (8,) | 32 |
| `w13_experts` | bf16 | (8, 128, 64) | 131,072 |
| `w2_experts` | bf16 | (8, 32, 128) | 65,536 |
| `w_s13` | bf16 | (128, 64) | 16,384 |
| `w_s2` | bf16 | (32, 128) | 8,192 |

**`A_.._2` saved context** — 67,108,864 bytes = **1,024.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 64) | 8,388,608 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 40) | 5,242,880 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 64) | 8,388,608 |
| `h_mid` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 8) | 1,048,576 |
| `h13` | bf16 | (131072, 64) | 16,777,216 |
| `s13` | bf16 | (65536, 64) | 8,388,608 |

**`M_.._2` metadata** — 1,310,976 bytes = **20.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 2) | 262,144 |
| `route_ids` | int32 | (65536, 2) | 524,288 |
| `route_order` | int32 | (131072,) | 524,288 |
| `route_offsets` | int32 | (9,) | 36 |

**`W_head`** — 131,328 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 128) | 131,072 |
| `final_norm_w` | bf16 | (128,) | 256 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (131,072B)
- outputs: `y_embed_0_0` (16,777,216B)
- mutates: —
- kernel calls:
    0. `index_select`

### `gdl_fwd` — `Glm52DlBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (16,777,216B), `W_0` (306,688B)
- outputs: `y_0_0_0` (16,777,216B), `A_0_0_0` (108,003,328B), `M_0_0_0` (6,291,456B)
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
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `rmsnorm_fwd`
    3. `mm`
    4. `rope_fwd`
    5. `rmsnorm_apply`
    6. `mm`
    7. `rope_fwd`
    8. `mm`
    9. `rope_fwd`
    10. `mm ×2`
    11. `rmsnorm_fwd`
    12. `rope_fwd`
    13. `mm`
    14. `dsa_index_scores`
    15. `dsa_topk`
    16. `dsa_sparse_attn_fwd`
    17. `addmm`
    18. `rmsnorm_fwd`
    19. `mm ×2`
    20. `swiglu_fwd_out`
    21. `addmm`

### `gml_fwd` — `Glm52MlBlockFwd`

- example task: `block_fwd_0_0_1`
- inputs: `y_0_0_0` (16,777,216B), `W_1` (333,568B)
- outputs: `y_0_0_1` (16,777,216B), `A_0_0_1` (67,108,864B), `M_0_0_1` (7,602,432B)
- mutates: —
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
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `rmsnorm_fwd`
    3. `mm`
    4. `rope_fwd`
    5. `rmsnorm_apply`
    6. `mm`
    7. `rope_fwd`
    8. `mm`
    9. `rope_fwd`
    10. `mm ×2`
    11. `rmsnorm_fwd`
    12. `rope_fwd`
    13. `mm`
    14. `dsa_index_scores`
    15. `dsa_topk`
    16. `dsa_sparse_attn_fwd`
    17. `addmm`
    18. `rmsnorm_fwd`
    19. `mm`
    20. `moe_topk_sigmoid_noaux`
    21. `moe_sort`
    22. `moe_dispatch_fwd`
    23. `moe_grouped_mm_fwd`
    24. `mm`
    25. `swiglu_packed_fwd`
    26. `moe_grouped_mm_fwd`
    27. `swiglu_packed_fwd`
    28. `mm`
    29. `moe_combine_fwd`

### `gmf_fwd` — `Glm52MfBlockFwd`

- example task: `block_fwd_0_0_2`
- inputs: `y_0_0_1` (16,777,216B), `W_2` (288,000B), `M_0_0_1` (7,602,432B)
- outputs: `y_0_0_2` (16,777,216B), `A_0_0_2` (67,108,864B), `M_0_0_2` (1,310,976B)
- mutates: —
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
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `rmsnorm_fwd`
    3. `mm`
    4. `rope_fwd`
    5. `mm`
    6. `rmsnorm_fwd`
    7. `rope_fwd`
    8. `mm`
    9. `dsa_sparse_attn_fwd`
    10. `addmm`
    11. `rmsnorm_fwd`
    12. `mm`
    13. `moe_topk_sigmoid_noaux`
    14. `moe_sort`
    15. `moe_dispatch_fwd`
    16. `moe_grouped_mm_fwd`
    17. `mm`
    18. `swiglu_packed_fwd`
    19. `moe_grouped_mm_fwd`
    20. `swiglu_packed_fwd`
    21. `mm`
    22. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_5` (16,777,216B), `targets_0_0` (262,144B), `W_head` (131,328B)
- outputs: `dy_0_0_5` (16,777,216B), `loss_0_0` (4B), `dW_head_0` (131,328B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (131,328B), `dW_head_0` (131,328B), `O_head` (262,656B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `gmf_bwd` — `Glm52MfBlockBwd`

- example task: `block_bwd_0_0_5`
- inputs: `dy_0_0_5` (16,777,216B), `A_0_0_5` (67,108,864B), `y_0_0_4` (16,777,216B), `W_5` (288,000B), `M_0_0_5` (1,310,976B), `M_0_0_4` (7,602,432B)
- outputs: `dy_0_0_4` (16,777,216B), `dW_0_5` (288,000B), `dM_0_0_4` (6,291,456B)
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
- inputs: `W_5` (288,000B), `dW_0_5` (288,000B), `O_5` (576,000B)
- outputs: —
- mutates: `W_5`, `O_5`
- kernel calls:
    0. `adamw_step ×14`

### `gml_bwd` — `Glm52MlBlockBwd`

- example task: `block_bwd_0_0_4`
- inputs: `dy_0_0_4` (16,777,216B), `A_0_0_4` (67,108,864B), `y_0_0_3` (16,777,216B), `W_4` (333,568B), `M_0_0_4` (7,602,432B), `dM_0_0_4` (6,291,456B)
- outputs: `dy_0_0_3` (16,777,216B), `dW_0_4` (333,568B)
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
- inputs: `dy_0_0_0` (16,777,216B), `A_0_0_0` (108,003,328B), `y_embed_0_0` (16,777,216B), `W_0` (306,688B), `M_0_0_0` (6,291,456B)
- outputs: `dy_embed_0_0` (16,777,216B), `dW_0_0` (306,688B)
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
- inputs: `dy_embed_0_0` (16,777,216B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (131,072B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (131,072B), `dW_embed_0` (131,072B), `O_embed` (262,144B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

